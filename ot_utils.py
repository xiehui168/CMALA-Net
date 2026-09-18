# ot_utils.py
import torch
import torch.nn as nn


def sinkhorn_log_from_cost(cost, epsilon=0.1, n_iters=50):
    """
    数值稳定的 log-domain Sinkhorn。
    cost: [B, N, M]
    return: transport_plan [B, N, M]
    """
    # 1) 清洗输入
    cost = torch.nan_to_num(cost, nan=0.0, posinf=1e4, neginf=-1e4)
    cost = torch.clamp(cost, min=-1e4, max=1e4)

    # 2) 到 log 域
    eps = max(float(epsilon), 1e-3)   # eps 太小必然溢出，强制下限
    log_alpha = -cost / eps
    # 减去最大值，保证 exp 不溢出
    log_alpha = log_alpha - log_alpha.amax(dim=(-2, -1), keepdim=True)

    log_u = torch.zeros_like(log_alpha[..., :, 0])   # [B, N]
    log_v = torch.zeros_like(log_alpha[..., 0, :])   # [B, M]

    for _ in range(n_iters):
        log_u = -torch.logsumexp(log_alpha + log_v.unsqueeze(-2), dim=-1)
        log_v = -torch.logsumexp(log_alpha + log_u.unsqueeze(-1), dim=-2)

    plan = torch.exp(log_alpha + log_u.unsqueeze(-1) + log_v.unsqueeze(-2))
    plan = torch.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    return plan


def sinkhorn_log_from_log_alpha(log_alpha, n_iters=50):
    """
    如果模型内部已经有 log_alpha = -cost / epsilon，用这个。
    log_alpha: [B, N, M]
    """
    log_alpha = torch.nan_to_num(log_alpha, nan=0.0, posinf=1e4, neginf=-1e4)
    log_alpha = torch.clamp(log_alpha, min=-1e4, max=1e4)
    log_alpha = log_alpha - log_alpha.amax(dim=(-2, -1), keepdim=True)

    log_u = torch.zeros_like(log_alpha[..., :, 0])
    log_v = torch.zeros_like(log_alpha[..., 0, :])

    for _ in range(n_iters):
        log_u = -torch.logsumexp(log_alpha + log_v.unsqueeze(-2), dim=-1)
        log_v = -torch.logsumexp(log_alpha + log_u.unsqueeze(-1), dim=-2)

    plan = torch.exp(log_alpha + log_u.unsqueeze(-1) + log_v.unsqueeze(-2))
    plan = torch.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    return plan


class LogSinkhorn(nn.Module):
    """
    直接替换原来 Sinkhorn 模块。
    plan = LogSinkhorn(epsilon=0.1, n_iters=50)(cost)
    """
    def __init__(self, epsilon=0.1, n_iters=50):
        super().__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters

    def forward(self, cost):
        # 无论外部是否在 autocast，这里强制 FP32，避免 FP16 exp 溢出
        cost = cost.float()
        return sinkhorn_log_from_cost(cost, self.epsilon, self.n_iters)