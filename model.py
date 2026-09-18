"""
CMALA-Net: Cross-Modal Optimal Transport for Weakly Supervised Breast Cancer Subtyping
核心模型实现 - 数值稳定 Sinkhorn + 保留 patch 信息避免 OT 退化
"""
import os
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import Dinov2Model, AutoConfig

HF_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]


def _configure_hf_endpoint():
    if os.environ.get("HF_ENDPOINT"):
        return
    os.environ["HF_ENDPOINT"] = HF_MIRRORS[0]
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


_configure_hf_endpoint()


def load_dinov2_with_fallback(pretrained=True):
    model_name = 'facebook/dinov2-large'

    if not pretrained:
        config = AutoConfig.from_pretrained(model_name)
        model = Dinov2Model(config)
        print("[INFO] Using randomly initialized DINOv2 (pretrained=False)")
        return model

    for i, endpoint in enumerate(HF_MIRRORS):
        try:
            os.environ["HF_ENDPOINT"] = endpoint
            print(f"[INFO] Loading DINOv2 from: {endpoint}")
            model = Dinov2Model.from_pretrained(
                model_name,
                resume_download=True,
                local_files_only=False,
            )
            print(f"[OK] DINOv2 loaded successfully from {endpoint}")
            return model
        except Exception as e:
            print(f"[WARNING] Failed to load from {endpoint}: {type(e).__name__}")
            if i < len(HF_MIRRORS) - 1:
                print(f"[INFO] Trying next mirror...")
            continue

    print("\n" + "=" * 70)
    print("[WARNING] 自动下载失败，将使用随机初始化 DINOv2 进行测试")
    print("=" * 70 + "\n")
    config = AutoConfig.from_pretrained(model_name)
    return Dinov2Model(config)


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=16, alpha=32):
        super().__init__()
        self.original = original_linear
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            rank=rank,
            alpha=alpha
        )

    def forward(self, x):
        return self.original(x) + self.lora(x)


def apply_lora_to_model(model, rank=16, alpha=32, target_modules=None):
    if target_modules is None:
        target_modules = ['qkv', 'proj', 'fc1', 'fc2',
                          'query', 'key', 'value', 'dense']

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            apply_lora_to_model(module, rank, alpha, target_modules)
        if isinstance(module, nn.Linear):
            should_apply = any(t in name for t in target_modules)
            if should_apply:
                setattr(model, name, LoRALinear(module, rank=rank, alpha=alpha))
    return model


# ===================== Sinkhorn 最优传输对齐模块 =====================
class SinkhornAlignment(nn.Module):
    """
    Cross-Modal Alignment via Sinkhorn Optimal Transport

    本版核心修复：
      用 concat(mean, max) 替代单纯的 mean，保留 patch 级的 max 响应，
      防止 global 特征退化为"所有 patch 的均值"（样本间无区分性）。
    """

    def __init__(self, eps=0.06, max_iters=50, init_scale=10.0):
        super().__init__()
        self.eps = eps
        self.max_iters = max_iters
        # 冻结的 scale，防止模型通过缩小 scale 让 OT 退化成均匀解
        self.register_buffer('log_scale', torch.tensor(math.log(init_scale)))

    def _log_sinkhorn(self, cost):
        eps = max(float(self.eps), 1e-3)
        log_K = -cost / eps
        log_K = log_K - log_K.amax(dim=(-2, -1), keepdim=True).detach()

        B, N, M = log_K.shape
        device = log_K.device
        dtype = log_K.dtype

        log_a = -math.log(N)
        log_b = -math.log(M)

        log_u = torch.full((B, N), log_a, device=device, dtype=dtype)
        log_v = torch.full((B, M), log_b, device=device, dtype=dtype)

        for _ in range(self.max_iters):
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(-2), dim=-1)
            log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=-2)

        log_T = log_K + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        T = torch.exp(log_T)
        T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
        return T

    def forward(self, feat_us, feat_path):
        B, N_us, D = feat_us.shape
        B, N_path, D = feat_path.shape
        device = feat_us.device

        # ---- 直接用原始特征算 cosine cost ----
        feat_us_norm = F.normalize(feat_us, p=2, dim=-1)
        feat_path_norm = F.normalize(feat_path, p=2, dim=-1)
        cos_sim = torch.bmm(feat_us_norm, feat_path_norm.transpose(1, 2))

        # ---- 冻结的 scale ----
        scale = self.log_scale.exp().clamp(min=5.0, max=200.0)
        cost = -cos_sim * scale
        cost = torch.clamp(cost, min=-100.0, max=100.0)

        # ---- log-domain Sinkhorn ----
        T = self._log_sinkhorn(cost)

        # ---- 对齐特征 ----
        aligned_us_to_path = N_us * torch.bmm(T, feat_path)          # [B, N_us, D]
        aligned_path_to_us = N_path * torch.bmm(T.transpose(1, 2), feat_us)  # [B, N_path, D]

        # ===== 关键修复：concat(mean, max) 保留更多信息 =====
        global_us = torch.cat([
            aligned_us_to_path.mean(dim=1),
            aligned_us_to_path.max(dim=1)[0],
        ], dim=-1)   # [B, 2D]

        global_path = torch.cat([
            aligned_path_to_us.mean(dim=1),
            aligned_path_to_us.max(dim=1)[0],
        ], dim=-1)   # [B, 2D]

        return T, global_us, global_path


# ===================== CMALA-Net 主模型 =====================
class CMALANet(nn.Module):
    def __init__(
        self,
        num_subtypes=5,
        num_biomarkers=4,
        embed_dim=1024,
        lora_rank=16,
        lora_alpha=32,
        sinkhorn_eps=0.06,
        sinkhorn_iters=50,
        pretrained=True,
        dinov2_path=None,
        swin_pretrained=True
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ===== 超声分支: Swin-Transformer Base =====
        print(f"[INFO] Loading Swin-Transformer Base backbone...")
        try:
            self.us_backbone = timm.create_model(
                'swin_base_patch4_window7_224',
                pretrained=pretrained and swin_pretrained,
                features_only=True,
                out_indices=(3,),
            )
        except Exception as e:
            print(f"[WARNING] Swin pretrained download failed ({type(e).__name__}), "
                  f"falling back to random init.")
            self.us_backbone = timm.create_model(
                'swin_base_patch4_window7_224',
                pretrained=False,
                features_only=True,
                out_indices=(3,),
            )
        us_feat_dim = self.us_backbone.feature_info.channels()[-1]

        self.us_proj = nn.Sequential(
            nn.LayerNorm(us_feat_dim),
            nn.Linear(us_feat_dim, embed_dim)
        )

        # ===== 病理分支: DINOv2 =====
        if dinov2_path is not None and os.path.exists(dinov2_path):
            print(f"[INFO] Loading DINOv2 from local path: {dinov2_path}")
            self.path_backbone = Dinov2Model.from_pretrained(dinov2_path)
        else:
            self.path_backbone = load_dinov2_with_fallback(pretrained=pretrained)

        path_feat_dim = self.path_backbone.config.hidden_size

        self.path_proj = nn.Sequential(
            nn.LayerNorm(path_feat_dim),
            nn.Linear(path_feat_dim, embed_dim)
        )

        # ===== 跨模态对齐模块 =====
        self.cmal = SinkhornAlignment(
            eps=sinkhorn_eps,
            max_iters=sinkhorn_iters,
            init_scale=10.0,
        )

        # ===== 对比学习专用投影头（256 维） =====
        # 注意：输入变成 2D（因为 global_us 现在是 concat(mean, max)）
        self.contrast_proj_us = nn.Sequential(
            nn.Linear(2 * embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )
        self.contrast_proj_path = nn.Sequential(
            nn.Linear(2 * embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # ===== 多任务分类头 =====
        # 注意：fusion_dim 从 2D 变成 4D
        fusion_dim = 4 * embed_dim

        self.subtype_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_subtypes)
        )

        self.biomarker_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ) for _ in range(num_biomarkers)
        ])

        # ===== LoRA + 冻结 =====
        self._apply_lora(lora_rank, lora_alpha)
        self._freeze_backbone()

    def _apply_lora(self, rank, alpha):
        apply_lora_to_model(self.us_backbone, rank=rank, alpha=alpha)
        apply_lora_to_model(self.path_backbone, rank=rank, alpha=alpha)

    def _freeze_backbone(self):
        for name, param in self.us_backbone.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        for name, param in self.path_backbone.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False

    def extract_us_patches(self, x):
        features = self.us_backbone(x)[0]
        B, H, W, C = features.shape
        patch_features = features.reshape(B, H * W, C)
        patch_features = self.us_proj(patch_features)
        return patch_features

    def extract_path_patches(self, x):
        outputs = self.path_backbone(x)
        patch_features = outputs.last_hidden_state[:, 1:, :]
        patch_features = self.path_proj(patch_features)
        return patch_features

    def forward(self, us_img, path_img, return_features=False):
        us_patches = self.extract_us_patches(us_img)
        path_patches = self.extract_path_patches(path_img)

        T, global_us, global_path = self.cmal(us_patches, path_patches)

        # global_us / global_path 现在是 [B, 2D]
        fused = torch.cat([global_us, global_path], dim=-1)   # [B, 4D]

        subtype_logits = self.subtype_classifier(fused)
        biomarker_logits = [head(fused) for head in self.biomarker_classifiers]

        if return_features:
            contrast_us = self.contrast_proj_us(global_us)
            contrast_path = self.contrast_proj_path(global_path)
            return {
                'subtype_logits': subtype_logits,
                'biomarker_logits': biomarker_logits,
                'transport_plan': T,
                'global_us': global_us,
                'global_path': global_path,
                'fused_features': fused,
                'us_patches': us_patches,
                'path_patches': path_patches,
                'contrast_us': contrast_us,
                'contrast_path': contrast_path,
            }

        return subtype_logits, biomarker_logits, T

    def get_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total