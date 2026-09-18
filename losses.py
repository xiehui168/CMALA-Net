"""
CMALA-Net 混合损失函数实现
包含: Cross-Entropy, Focal Loss, Cross-Modal Contrastive Loss, Auxiliary BCE Loss, MMD Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CrossModalContrastiveLoss(nn.Module):
    """
    Cross-Modal Contrastive Loss
    输入是已经投影到 256 维的 contrast_us / contrast_path，
    比直接在 1024 维上对比更容易学习。
    """
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat_us, feat_path):
        B = feat_us.shape[0]
        device = feat_us.device

        feat_us = F.normalize(feat_us, dim=-1)
        feat_path = F.normalize(feat_path, dim=-1)

        logits = torch.mm(feat_us, feat_path.t()) / self.temperature
        labels = torch.arange(B, device=device)

        loss_us_to_path = F.cross_entropy(logits, labels)
        loss_path_to_us = F.cross_entropy(logits.t(), labels)

        return (loss_us_to_path + loss_path_to_us) / 2.0


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def guassian_kernel(self, source, target):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)),
                                            int(total.size(0)),
                                            int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)),
                                            int(total.size(0)),
                                            int(total.size(1)))

        L2_distance = ((total0 - total1) ** 2).sum(2)

        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth = max(bandwidth.item(), 1e-8)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i)
                          for i in range(self.kernel_num)]

        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(XX + YY - XY - YX)
        return loss


class CMALALoss(nn.Module):
    """
    L_total = w_ce * L_ce + w_focal * L_focal + w_cont * L_cont
              + w_aux * L_aux + w_mmd * L_mmd
    """

    def __init__(
        self,
        w_ce=1.0,
        w_focal=0.35,
        w_cont=0.3,
        w_aux=1.0,
        w_mmd=0.08,
        focal_gamma=2.0,
        cont_temperature=0.2,
        class_weights=None
    ):
        super().__init__()
        self.w_ce = w_ce
        self.w_focal = w_focal
        self.w_cont = w_cont
        self.w_aux = w_aux
        self.w_mmd = w_mmd

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        self.contrastive_loss = CrossModalContrastiveLoss(temperature=cont_temperature)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mmd_loss = MMDLoss()

    def forward(self, outputs, labels, biomarkers, domain_labels=None):
        subtype_logits = outputs['subtype_logits']
        biomarker_logits = outputs['biomarker_logits']

        # 1. CE
        l_ce = self.ce_loss(subtype_logits, labels)

        # 2. Focal
        l_focal = self.focal_loss(subtype_logits, labels)

        # 3. Cross-Modal Contrastive（用投影头输出）
        if 'contrast_us' in outputs and 'contrast_path' in outputs:
            l_cont = self.contrastive_loss(outputs['contrast_us'],
                                           outputs['contrast_path'])
        else:
            # 兼容旧接口
            l_cont = self.contrastive_loss(outputs['global_us'],
                                           outputs['global_path'])

        # 4. Auxiliary Biomarker BCE
        l_aux = 0.0
        for i, logit in enumerate(biomarker_logits):
            l_aux = l_aux + self.bce_loss(logit.squeeze(-1), biomarkers[:, i].float())
        l_aux = l_aux / len(biomarker_logits)

        # 5. MMD
        l_mmd = torch.tensor(0.0, device=subtype_logits.device)
        if domain_labels is not None:
            uniq = torch.unique(domain_labels)
            if len(uniq) > 1:
                fused = outputs['fused_features']
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        src = fused[domain_labels == uniq[i]]
                        tgt = fused[domain_labels == uniq[j]]
                        if len(src) > 1 and len(tgt) > 1:
                            l_mmd = l_mmd + self.mmd_loss(src, tgt)

        total_loss = (
            self.w_ce * l_ce +
            self.w_focal * l_focal +
            self.w_cont * l_cont +
            self.w_aux * l_aux +
            self.w_mmd * l_mmd
        )

        loss_dict = {
            'total': total_loss.item(),
            'ce': l_ce.item(),
            'focal': l_focal.item(),
            'contrastive': l_cont.item(),
            'auxiliary': l_aux.item(),
            'mmd': l_mmd.item(),
        }

        return total_loss, loss_dict