# -*- coding: utf-8 -*-
"""
病理基础模型 (Pathology Foundation Model) backbone 注册表
==========================================================
为 CMALA-Net 的病理分支提供统一的加载接口，用于审稿人要求的
backbone 对比实验:

    dinov2      DINOv2-L/14        (facebook/dinov2-large,   1024-d, 代码原始默认)
    uni         UNI                (MahmoodLab/UNI,          1024-d, ViT-L/16)
    uni2h       UNI2-h             (MahmoodLab/UNI2-h,       1536-d, ViT-H/14)
    virchow2    Virchow2           (paige-ai/Virchow2,       1280-d, ViT-H/14)
    gigapath    Prov-GigaPath      (prov-gigapath/prov-gigapath, 1536-d, ViT-g/14)
    hoptimus1   H-optimus-1        (bioptimus/H-optimus-1,   1536-d, ViT-g/14)
    chief       CHIEF              (Xiyue-Wang/TransPath-CHIEF, 768-d, CTransPath/Swin-T)

所有 wrapper 暴露统一接口:
    .forward_patches(x)  -> [B, N_patches, D]   (仅 patch tokens, 去掉 CLS/寄存器 token)
    .feat_dim            -> int
    .lora_targets        -> list[str]           (LoRA 需要注入的 Linear 层名)
    .display_name        -> str

权重获取 (全部为 gated / 需申请权限的模型, 详见 README):
    1. 在 HuggingFace 网页申请权限 (需机构邮箱)
    2. 设置环境变量 HF_TOKEN=<your_token>
    3. 国内网络可设 HF_ENDPOINT=https://hf-mirror.com
    4. 完全离线时: 用 huggingface-cli download <repo> --local-dir <dir>
       然后传 --path_ckpt <dir> (目录需含 config.json + 权重文件)
       CHIEF 传 --path_ckpt <CHIEF.pth 文件路径>
"""
import os
import math
import warnings

import torch
import torch.nn as nn
import timm

HF_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# H-optimus-1 官方推荐的归一化统计量 (见 bioptimus/H-optimus-1 model card)
HOPTIMUS1_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS1_STD = (0.211883, 0.230117, 0.177517)

VIT_LORA_TARGETS = ['qkv', 'proj', 'fc1', 'fc2']
SWIGLU_LORA_TARGETS = ['qkv', 'proj', 'w12', 'w3', 'fc1', 'fc2']
# transformers 旧版: query/key/value/dense; 新版 (>=5.x): q_proj/k_proj/v_proj/o_proj
DINOV2_LORA_TARGETS = ['query', 'key', 'value', 'dense',
                       'q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'fc1', 'fc2']

# UNI2-h 官方 timm_kwargs (见 MahmoodLab/UNI2-h model card)
UNI2H_TIMM_KWARGS = {
    'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
    'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667 * 2,
    'num_classes': 0, 'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
    'reg_tokens': 8, 'dynamic_img_size': True,
}


def _configure_hf_endpoint():
    """配置 HF 镜像端点 (国内网络)"""
    if os.environ.get("HF_ENDPOINT"):
        return
    os.environ["HF_ENDPOINT"] = HF_MIRRORS[0]
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass


_configure_hf_endpoint()


def _load_state_dict_file(path):
    """手动加载单个权重文件 -> state_dict.

    不走 timm 的 checkpoint_path: 其按扩展名判断格式, Windows 下路径末尾
    多一个 '.' (文件系统会忽略但扩展名判断失效) 会误走 torch.load;
    且 torch>=2.6 默认 weights_only=True 会拒绝部分 checkpoint.
    """
    clean = path.rstrip('. ')
    if clean.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'weights', 'teacher'):
            v = ckpt.get(key)
            if isinstance(v, dict) and any(k.endswith('.weight') for k in v):
                return v
    return ckpt


def _timm_create_with_mirror(hub_id, pretrained, timm_kwargs):
    """带镜像 fallback 的 timm hub 模型创建"""
    last_err = None
    for endpoint in HF_MIRRORS:
        try:
            os.environ["HF_ENDPOINT"] = endpoint
            print(f"[INFO] Loading {hub_id} from: {endpoint} "
                  f"(pretrained={pretrained})")
            model = timm.create_model(hub_id, pretrained=pretrained, **timm_kwargs)
            print(f"[OK] {hub_id} loaded successfully from {endpoint}")
            return model
        except Exception as e:
            print(f"[WARNING] Failed to load {hub_id} from {endpoint}: "
                  f"{type(e).__name__}: {str(e)[:200]}")
            last_err = e
            continue
    raise RuntimeError(
        f"无法加载 {hub_id}。\n"
        f"若为 gated 模型, 请先在 HF 网页申请权限并设置 HF_TOKEN 环境变量;\n"
        f"或使用 huggingface-cli download 下载到本地目录后传 --path_ckpt <dir>。\n"
        f"最后错误: {last_err}"
    )


# =====================================================================
# timm ViT 系 wrapper (UNI / UNI2-h / Virchow2 / GigaPath / H-optimus-1)
# =====================================================================
class TimmPathBackbone(nn.Module):
    """
    timm ViT 病理基础模型统一封装.
    forward_features -> [B, T, D], 其中 T = 前缀 token 数 (CLS+reg) + patch 数.
    直接取最后 n_patches 个 token 作为 patch 特征, 无需关心各模型前缀数量.
    """

    def __init__(self, model, feat_dim, patch_size, display_name, lora_targets):
        super().__init__()
        self.model = model
        self.feat_dim = feat_dim
        self.patch_size = patch_size
        self.display_name = display_name
        self.lora_targets = lora_targets

    def forward_patches(self, x):
        feats = self.model.forward_features(x)      # [B, T, D]
        n_patches = (x.shape[-1] // self.patch_size) ** 2
        return feats[:, -n_patches:, :]             # [B, N, D]

    def forward(self, x):
        return self.forward_patches(x)


# =====================================================================
# DINOv2 (HuggingFace transformers) wrapper —— 代码原始默认 backbone
# =====================================================================
class DinoV2Backbone(nn.Module):
    def __init__(self, pretrained=True, ckpt_path=None):
        super().__init__()
        from transformers import Dinov2Model, AutoConfig
        model_name = 'facebook/dinov2-large'

        if ckpt_path is not None and os.path.exists(ckpt_path):
            print(f"[INFO] Loading DINOv2 from local path: {ckpt_path}")
            self.model = Dinov2Model.from_pretrained(ckpt_path)
        elif not pretrained:
            config = AutoConfig.from_pretrained(model_name)
            self.model = Dinov2Model(config)
            print("[INFO] Using randomly initialized DINOv2 (pretrained=False)")
        else:
            self.model = self._load_with_mirror(model_name)

        self.feat_dim = self.model.config.hidden_size
        self.patch_size = 14
        self.display_name = 'DINOv2-L'
        self.lora_targets = DINOV2_LORA_TARGETS

    @staticmethod
    def _load_with_mirror(model_name):
        from transformers import Dinov2Model, AutoConfig
        for endpoint in HF_MIRRORS:
            try:
                os.environ["HF_ENDPOINT"] = endpoint
                print(f"[INFO] Loading DINOv2 from: {endpoint}")
                model = Dinov2Model.from_pretrained(
                    model_name, local_files_only=False)
                print(f"[OK] DINOv2 loaded successfully from {endpoint}")
                return model
            except Exception as e:
                print(f"[WARNING] Failed to load from {endpoint}: {type(e).__name__}")
                continue
        print("[WARNING] DINOv2 自动下载失败, 使用随机初始化 (仅用于流程测试)")
        config = AutoConfig.from_pretrained(model_name)
        return Dinov2Model(config)

    def forward_patches(self, x):
        return self.model(x).last_hidden_state[:, 1:, :]

    def forward(self, x):
        return self.forward_patches(x)


# =====================================================================
# CHIEF (CTransPath: Swin-T + ConvStem, 768-d)
# =====================================================================
class ConvStem(nn.Module):
    """CTransPath 的卷积 patch stem (来自 Xiyue-Wang/TransPath ctran.py)"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=768, norm_layer=None, flatten=True,
                 output_fmt='BNC', strict_img_size=True, **_ignored):
        # timm>=1.0 的 SwinTransformer 会向 embed_layer 透传
        # strict_img_size / output_fmt 等额外参数
        super().__init__()
        from timm.layers import to_2tuple
        assert patch_size == 4
        assert embed_dim % 8 == 0
        assert output_fmt in ('BNC', 'NHWC', 'BCHW')

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten if output_fmt == 'BNC' else False
        self.output_fmt = output_fmt

        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3,
                                  stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)                     # BCHW
        if self.output_fmt == 'NHWC':
            x = x.permute(0, 2, 3, 1)        # BCHW -> BHWC
        elif self.flatten:
            x = x.flatten(2).transpose(1, 2)   # BCHW -> BNC
        x = self.norm(x)
        return x


class CTransPathBackbone(nn.Module):
    """CHIEF = CTransPath 架构 + CHIEF 预训练权重 (Swin-T, 768-d)"""

    HF_REPO = 'Xiyue-Wang/TransPath-CHIEF'
    CKPT_CANDIDATES = ['CHIEF.pth', 'chief.pth', 'pytorch_model.bin', 'model.pth']

    def __init__(self, pretrained=True, ckpt_path=None):
        super().__init__()
        model = timm.create_model('swin_tiny_patch4_window7_224',
                                  embed_layer=ConvStem, pretrained=False)
        model.head = nn.Identity()
        self.model = model
        self.feat_dim = 768
        self.patch_size = 4          # Swin patch_size; 224 -> 49 patches
        self.display_name = 'CHIEF'
        self.lora_targets = VIT_LORA_TARGETS

        if ckpt_path is None and pretrained:
            ckpt_path = self._try_download()
        if ckpt_path is not None:
            self._load_weights(ckpt_path)
        elif pretrained:
            print("[WARNING] CHIEF 权重未提供且自动下载失败, "
                  "使用随机初始化 (仅用于流程测试)。\n"
                  "          请下载 CHIEF.pth 后传 --path_ckpt <路径>")

    def _try_download(self):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return None
        for fname in self.CKPT_CANDIDATES:
            try:
                for endpoint in HF_MIRRORS:
                    os.environ["HF_ENDPOINT"] = endpoint
                    path = hf_hub_download(self.HF_REPO, fname)
                    print(f"[OK] CHIEF weights downloaded: {path}")
                    return path
            except Exception:
                continue
        return None

    def _load_weights(self, ckpt_path):
        print(f"[INFO] Loading CHIEF weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt
        if isinstance(ckpt, dict):
            for key in ('model', 'state_dict', 'weights', 'teacher'):
                v = ckpt.get(key) if hasattr(ckpt, 'get') else None
                if isinstance(v, dict) and any(
                        k.endswith('.weight') or k.endswith('.bias') for k in v):
                    sd = v
                    break
        # 清理常见前缀
        for prefix in ('module.', 'model.', 'backbone.'):
            if all(k.startswith(prefix) for k in sd):
                sd = {k[len(prefix):]: v for k, v in sd.items()}
        # timm>=1.0 Swin 的 PatchMerging 在 stage 开头 (layers.1/2/3.downsample),
        # TransPath 官方实现在 stage 末尾 (layers.0/1/2.downsample), 需 +1 重映射
        model_keys = set(self.model.state_dict().keys())
        if ('layers.0.downsample.reduction.weight' not in model_keys
                and 'layers.1.downsample.reduction.weight' in model_keys):
            import re
            sd = {re.sub(r'^layers\.(\d+)\.downsample\.',
                         lambda m: f'layers.{int(m.group(1)) + 1}.downsample.',
                         k): v for k, v in sd.items()}
            print("[INFO] CHIEF: downsample 键已按 timm>=1.0 布局重映射 (+1)")
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(unexpected)
        print(f"[OK] CHIEF loaded: {n_loaded} tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if len(missing) > len(sd) // 2:
            raise RuntimeError(
                f"CHIEF checkpoint 与 CTransPath 架构不匹配, "
                f"missing keys 过多 ({len(missing)}), 请确认权重文件正确。")

    def train(self, mode=True):
        # 冻结 backbone 时 BN running stats 不能更新, 否则破坏预训练特征
        super().train(mode)
        for m in self.model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def forward_patches(self, x):
        # timm>=1.0 Swin forward_features -> [B, H, W, C] (NHWC)
        feats = self.model.forward_features(x)
        if feats.ndim == 4:
            B, H, W, C = feats.shape
            feats = feats.reshape(B, H * W, C)   # [B, 49, 768]
        return feats

    def forward(self, x):
        return self.forward_patches(x)


# =====================================================================
# Backbone 注册表
# =====================================================================
BACKBONE_REGISTRY = {
    'dinov2': dict(
        display='DINOv2-L', type='dinov2',
        hub='facebook/dinov2-large', arch=None, kwargs={},
        feat_dim=1024, patch=14,
        norm=(IMAGENET_MEAN, IMAGENET_STD),
        params_m=304,
    ),
    'uni': dict(
        display='UNI', type='timm',
        hub='hf-hub:MahmoodLab/UNI', arch='vit_large_patch16_224',
        # 官方 model card: num_classes=0 (无分类头, 单文件加载必须一致)
        kwargs={'init_values': 1e-5, 'num_classes': 0,
                'dynamic_img_size': True},
        feat_dim=1024, patch=16,
        norm=(IMAGENET_MEAN, IMAGENET_STD),
        params_m=307,
    ),
    'uni2h': dict(
        display='UNI2-h', type='timm',
        hub='hf-hub:MahmoodLab/UNI2-h', arch='vit_giant_patch14_224',
        kwargs=UNI2H_TIMM_KWARGS,
        feat_dim=1536, patch=14,
        norm=(IMAGENET_MEAN, IMAGENET_STD),
        params_m=681,
        lora_targets=SWIGLU_LORA_TARGETS,
    ),
    'virchow2': dict(
        display='Virchow2', type='timm',
        hub='hf-hub:paige-ai/Virchow2', arch='vit_huge_patch14_224',
        kwargs={'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU},
        # 离线文件模式 (实测 checkpoint): pos_embed 261=1+4+256 (cls+reg 嵌入位置,
        # 即 no_embed_class=False); fc1=6832=2x3416 -> mlp_ratio=5.3375
        file_kwargs={'init_values': 1e-5, 'mlp_ratio': 5.3375,
                     'num_classes': 0,
                     'mlp_layer': timm.layers.SwiGLUPacked,
                     'act_layer': torch.nn.SiLU, 'reg_tokens': 4,
                     'dynamic_img_size': True},
        feat_dim=1280, patch=14,
        norm=(IMAGENET_MEAN, IMAGENET_STD),
        params_m=632,
        lora_targets=SWIGLU_LORA_TARGETS,
    ),
    'gigapath': dict(
        display='Prov-GigaPath', type='timm',
        hub='hf_hub:prov-gigapath/prov-gigapath', arch='vit_giant_patch14_224',
        kwargs={},
        # 离线文件模式 (实测 checkpoint): ViT-g/16 (patch=16, 196 patches);
        # fc1=8192 -> mlp_ratio=2.66667*2; LayerScale + 无分类头
        file_kwargs={'embed_dim': 1536, 'depth': 40, 'num_heads': 24,
                     'patch_size': 16, 'mlp_ratio': 2.66667 * 2,
                     'init_values': 1e-5, 'num_classes': 0,
                     'mlp_layer': timm.layers.SwiGLUPacked,
                     'act_layer': torch.nn.SiLU},
        feat_dim=1536, patch=16,
        norm=(IMAGENET_MEAN, IMAGENET_STD),
        params_m=1100,
        lora_targets=SWIGLU_LORA_TARGETS,
    ),
    'hoptimus1': dict(
        display='H-optimus-1', type='timm',
        hub='hf-hub:bioptimus/H-optimus-1', arch='vit_giant_patch14_224',
        kwargs={'init_values': 1e-5, 'dynamic_img_size': False},
        # 离线文件模式 (实测 checkpoint): 4 寄存器 token + no_embed_class;
        # fc1=8192 -> mlp_ratio=2.66667*2
        file_kwargs={'embed_dim': 1536, 'depth': 40, 'num_heads': 24,
                     'mlp_ratio': 2.66667 * 2, 'init_values': 1e-5,
                     'num_classes': 0, 'no_embed_class': True,
                     'reg_tokens': 4,
                     'mlp_layer': timm.layers.SwiGLUPacked,
                     'act_layer': torch.nn.SiLU},
        feat_dim=1536, patch=14,
        norm=(HOPTIMUS1_MEAN, HOPTIMUS1_STD),
        params_m=1100,
        lora_targets=SWIGLU_LORA_TARGETS,
    ),
    'chief': dict(
        display='CHIEF', type='chief',
        hub='Xiyue-Wang/TransPath-CHIEF', arch=None, kwargs={},
        feat_dim=768, patch=4,
        norm=(IMAGENET_MEAN, IMAGENET_STD),
        params_m=28,
    ),
}

BACKBONE_NAMES = list(BACKBONE_REGISTRY.keys())


def create_path_backbone(name, pretrained=True, ckpt_path=None, img_size=224):
    """
    创建病理 backbone wrapper.

    Args:
        name:       BACKBONE_REGISTRY 中的键
        pretrained: 是否加载预训练权重
        ckpt_path:  None=从 HF hub 下载;
                    目录 (含 config.json+权重, 来自 huggingface-cli download --local-dir)
                        = 离线加载 timm hub 模型;
                    文件 (.pth/.bin/.safetensors) = 离线权重文件
        img_size:   输入图像尺寸 (用于校验)

    Returns:
        wrapper (nn.Module): 含 .forward_patches(x)->[B,N,D], .feat_dim,
                             .lora_targets, .display_name
    """
    name = name.lower()
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"未知 backbone: {name}. "
                         f"可选: {BACKBONE_NAMES}")
    cfg = BACKBONE_REGISTRY[name]
    btype = cfg['type']

    if btype == 'dinov2':
        return DinoV2Backbone(pretrained=pretrained, ckpt_path=ckpt_path)

    if btype == 'chief':
        return CTransPathBackbone(pretrained=pretrained, ckpt_path=ckpt_path)

    # ---- timm ViT 系 ----
    kwargs = dict(cfg['kwargs'])
    file_kwargs = dict(cfg.get('file_kwargs', cfg['kwargs']))
    if ckpt_path is not None:
        if os.path.isdir(ckpt_path):
            # 本地 HF snapshot 目录 (含 config.json): timm 直接读取, 配置精确
            hub_id = 'hf-hub:' + os.path.abspath(ckpt_path)
            model = timm.create_model(hub_id, pretrained=True, **kwargs)
        elif os.path.isfile(ckpt_path):
            # 单个权重文件: 用 arch + 已知 kwargs 构建, 手动加载 state_dict
            print(f"[INFO] 使用单文件离线加载 {cfg['display']}: {ckpt_path}")
            model = timm.create_model(cfg['arch'], pretrained=False,
                                      **file_kwargs)
            sd = _load_state_dict_file(ckpt_path)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[OK] {cfg['display']} 权重加载: {len(sd) - len(unexpected)}"
                  f"/{len(sd)} tensors (missing={len(missing)})")
            if len(missing) > len(sd) // 2:
                raise RuntimeError(
                    f"{cfg['display']} checkpoint 与架构不匹配, "
                    f"missing keys 过多 ({len(missing)}), 请确认权重文件正确。")
        else:
            raise FileNotFoundError(f"--path_ckpt 路径不存在: {ckpt_path}")
    elif not pretrained:
        # 随机初始化: 直接用本地 arch 构建, 不访问网络
        # (gated 仓库即使拉取 config.json 也需要权限)
        print(f"[INFO] {cfg['display']}: 随机初始化 (pretrained=False), "
              f"arch={cfg['arch']}")
        model = timm.create_model(cfg['arch'], pretrained=False, **file_kwargs)
    else:
        model = _timm_create_with_mirror(cfg['hub'], pretrained, kwargs)

    lora_targets = cfg.get('lora_targets', VIT_LORA_TARGETS)
    wrapper = TimmPathBackbone(
        model, feat_dim=cfg['feat_dim'], patch_size=cfg['patch'],
        display_name=cfg['display'], lora_targets=lora_targets)

    # 形状自检 (一次前向, CPU, 不耗显存)
    return wrapper


def get_backbone_norm(name):
    """返回该 backbone 病理图像的归一化 (mean, std)"""
    return BACKBONE_REGISTRY[name.lower()]['norm']


def get_backbone_display_name(name):
    return BACKBONE_REGISTRY[name.lower()]['display']


def print_backbone_table():
    """打印支持的 backbone 列表"""
    print("\n支持的病理 backbone:")
    print("-" * 78)
    print(f"{'名称':<12} {'显示名':<16} {'维度':<6} {'参数量':<10} {'HF Repo'}")
    print("-" * 78)
    for k, v in BACKBONE_REGISTRY.items():
        print(f"{k:<12} {v['display']:<16} {v['feat_dim']:<6} "
              f"{v['params_m']}M{'':<4} {v['hub']}")
    print("-" * 78)


if __name__ == '__main__':
    # 快速形状自检: python backbones.py [backbone_name]
    import sys
    print_backbone_table()
    names = [sys.argv[1]] if len(sys.argv) > 1 else BACKBONE_NAMES
    for n in names:
        print(f"\n=== {n} (pretrained=False 随机权重, 仅验证形状) ===")
        bb = create_path_backbone(n, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = bb.forward_patches(x)
        print(f"  forward_patches: {tuple(out.shape)}  "
              f"(期望 [1, N, {bb.feat_dim}])")
        del bb
