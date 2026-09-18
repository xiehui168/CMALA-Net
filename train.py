"""
CMALA-Net 训练脚本（修复版）
关键修复：
  1. grad_clip 从 1.0 提到 5.0（原值把有效学习率砍到 1/40）
  2. last_good_state 只在 NaN 时保存（避免每个 batch 复制 1.6GB）
  3. 添加梯度范数日志 + 类别分布诊断
  4. 添加 warmup + 提高对比温度
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, confusion_matrix
from collections import Counter

from model import CMALANet
from losses import CMALALoss
from dataset import create_dataloaders, SUBTYPE_NAMES, BIOMARKER_NAMES


# =========================================================
# 兼容 PyTorch 新旧 AMP API
# =========================================================
def make_autocast(enabled=True):
    try:
        return torch.amp.autocast('cuda', enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled=True):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CMALA-Net')
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument('--train_csv', type=str, default=None)
    parser.add_argument('--val_csv', type=str, default=None)
    parser.add_argument('--test_B_csv', type=str, default=None)
    parser.add_argument('--test_C_csv', type=str, default=None)
    parser.add_argument('--us_dir', type=str, default='')
    parser.add_argument('--path_dir', type=str, default='')

    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--sinkhorn_eps', type=float, default=0.06)
    parser.add_argument('--sinkhorn_iters', type=int, default=50)

    parser.add_argument('--w_ce', type=float, default=1.0)
    parser.add_argument('--w_focal', type=float, default=0.35)
    parser.add_argument('--w_cont', type=float, default=0.3)
    parser.add_argument('--w_mmd', type=float, default=0.08)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='LoRA 微调推荐 3e-4；原 1e-4 太保守')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)

    # ==== 关键修复：grad_clip 从 1.0 提到 5.0 ====
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='梯度裁剪 max_norm；太小会把有效学习率砍到 1/40')

    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='warmup epoch 数')

    parser.add_argument('--amp', action='store_true', default=False,
                        help='是否启用 AMP（建议先关，稳定后再开）')

    parser.add_argument('--nan_lr_decay', type=float, default=0.5)

    parser.add_argument('--output_dir', type=str, default='checkpoints/')
    parser.add_argument('--save_name', type=str, default='cmalanet_best.pth')
    parser.add_argument('--last_name', type=str, default='cmalanet_last.pth')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--dinov2_path', type=str, default=None)
    parser.add_argument('--demo', action='store_true')

    return parser.parse_args()


# =========================================================
# 工具
# =========================================================
def get_cpu_state_dict(model):
    """把模型 state_dict 复制到 CPU（只在 NaN 时调用，不要每 batch 调用）"""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_cpu_state_dict(model, state, device):
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def check_outputs_nan(outputs):
    if not isinstance(outputs, dict):
        return None
    for k, v in outputs.items():
        if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
            return k
    return None


def print_outputs_debug(outputs, loss_dict, tag=""):
    print(f"\n[DEBUG {tag}] outputs:")
    for k, v in outputs.items():
        if torch.is_tensor(v):
            print(f"  out[{k}]: shape={tuple(v.shape)} "
                  f"has_nan={torch.isnan(v).any().item()} "
                  f"min={v.nan_to_num().min().item():.3e} "
                  f"max={v.nan_to_num().max().item():.3e}")
    print(f"[DEBUG {tag}] loss_dict:")
    for k, v in loss_dict.items():
        vt = v if torch.is_tensor(v) else torch.tensor(float(v))
        print(f"  {k}: value={float(vt)} "
              f"has_nan={torch.isnan(vt).any().item()}")


def get_grad_norm(model):
    """计算可训练参数的总梯度范数"""
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_norm += p.grad.detach().norm(2).item() ** 2
    return total_norm ** 0.5


# =========================================================
# Warmup + Cosine Scheduler
# =========================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 base_lr, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = base_lr * min_lr_ratio

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']


# =========================================================
# 训练
# =========================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device,
                    grad_clip=5.0, use_amp=False, nan_lr_decay=0.5,
                    debug_first_n=2):
    model.train()
    total_loss = 0.0
    n_valid = 0
    n_skipped = 0
    all_preds = []
    all_labels = []
    loss_meter = {k: 0.0 for k in
                  ['ce', 'focal', 'contrastive', 'auxiliary', 'mmd']}

    # ==== 关键修复：只在 NaN 时保存，不每 batch 复制 ====
    last_good_state = None
    need_save_state = True   # 第一个 batch 前保存一次

    pbar = tqdm(dataloader, desc='Training')
    for step, batch in enumerate(pbar):
        us = batch['us_img'].to(device, non_blocking=True)
        path = batch['path_img'].to(device, non_blocking=True)
        labels = batch['subtype'].to(device, non_blocking=True)
        biomarkers = batch['biomarkers'].to(device, non_blocking=True)
        domains = batch['domain'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with make_autocast(use_amp):
            outputs = model(us, path, return_features=True)
            loss, loss_dict = criterion(outputs, labels, biomarkers, domains)

        if step < debug_first_n and n_skipped == 0:
            print_outputs_debug(outputs, loss_dict, tag=f"epoch step={step}")

        # ---- NaN 检测 ----
        if torch.isnan(loss) or torch.isinf(loss):
            first_nan = check_outputs_nan(outputs)
            n_skipped += 1
            print(f"\n[NaN Loss] step={step}, first_nan_output={first_nan}")
            # 只有在需要时才保存 / 加载
            if last_good_state is None:
                last_good_state = get_cpu_state_dict(model)
            load_cpu_state_dict(model, last_good_state, device)
            optimizer.zero_grad(set_to_none=True)
            for pg in optimizer.param_groups:
                pg['lr'] = max(pg['lr'] * nan_lr_decay, 1e-7)
            pbar.set_postfix({'loss': 'NaN(skip)', 'skip': n_skipped})
            continue

        # ---- 反向 ----
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = get_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = get_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=grad_clip
            )
            optimizer.step()

        # 只在第一次成功更新后保存一次 state（用于后续 NaN 回滚）
        if last_good_state is None:
            last_good_state = get_cpu_state_dict(model)

        # ---- 统计 ----
        v_total = loss_dict['total']
        total_loss += float(v_total) if not torch.is_tensor(v_total) else v_total.item()
        n_valid += 1
        for k in loss_meter:
            if k in loss_dict:
                v = loss_dict[k]
                v = v.item() if torch.is_tensor(v) else float(v)
                if not (np.isnan(v) or np.isinf(v)):
                    loss_meter[k] += v

        preds = outputs['subtype_logits'].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'skip': n_skipped,
            'gn': f"{grad_norm:.2f}",
        })

    n_valid = max(n_valid, 1)
    avg_loss = total_loss / n_valid
    if len(all_labels) > 0:
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
    else:
        macro_f1 = 0.0
        accuracy = 0.0

    avg_losses = {k: v / n_valid for k, v in loss_meter.items()}

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'n_skipped': n_skipped,
        **avg_losses
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    n_valid = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_biomarker_preds = [[] for _ in range(4)]
    all_biomarker_labels = [[] for _ in range(4)]

    for batch in tqdm(dataloader, desc='Evaluating'):
        us = batch['us_img'].to(device, non_blocking=True)
        path = batch['path_img'].to(device, non_blocking=True)
        labels = batch['subtype'].to(device, non_blocking=True)
        biomarkers = batch['biomarkers'].to(device, non_blocking=True)
        domains = batch['domain'].to(device, non_blocking=True)

        with make_autocast(use_amp):
            outputs = model(us, path, return_features=True)
            loss, loss_dict = criterion(outputs, labels, biomarkers, domains)

        loss_val = loss_dict['total']
        loss_val = float(loss_val) if not torch.is_tensor(loss_val) else loss_val.item()
        if not (np.isnan(loss_val) or np.isinf(loss_val)):
            total_loss += loss_val
            n_valid += 1

        probs = torch.softmax(outputs['subtype_logits'], dim=1)
        preds = probs.argmax(dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i, logit in enumerate(outputs['biomarker_logits']):
            b_pred = (torch.sigmoid(logit) > 0.5).long().squeeze(-1)
            b_true = (biomarkers[:, i] > 0.5).long()
            all_biomarker_preds[i].extend(b_pred.cpu().numpy())
            all_biomarker_labels[i].extend(b_true.cpu().numpy())

    n_valid = max(n_valid, 1)
    avg_loss = total_loss / n_valid
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    if len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        kappa = cohen_kappa_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        per_class_f1 = f1_score(all_labels, all_preds, average=None,
                                labels=list(range(len(SUBTYPE_NAMES))),
                                zero_division=0)
    else:
        accuracy = macro_f1 = kappa = 0.0
        cm = np.zeros((len(SUBTYPE_NAMES), len(SUBTYPE_NAMES)), dtype=int)
        per_class_f1 = np.zeros(len(SUBTYPE_NAMES))

    per_class_recall = []
    for i in range(len(SUBTYPE_NAMES)):
        mask = all_labels == i
        rec = (all_preds[mask] == i).mean() if mask.sum() > 0 else 0.0
        per_class_recall.append(rec)

    biomarker_acc = []
    for i in range(4):
        try:
            b_acc = accuracy_score(all_biomarker_labels[i],
                                   all_biomarker_preds[i]) \
                    if len(all_biomarker_labels[i]) > 0 else 0.0
        except Exception:
            b_acc = 0.0
        biomarker_acc.append(b_acc)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'kappa': kappa,
        'per_class_f1': per_class_f1.tolist(),
        'per_class_recall': per_class_recall,
        'biomarker_accuracy': biomarker_acc,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


# =========================================================
# 主流程
# =========================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"AMP enabled: {args.amp}")
    print(f"grad_clip: {args.grad_clip}")
    print(f"lr: {args.lr}")

    # ===== 数据 =====
    print("\n" + "=" * 60)
    print("Preparing datasets...")
    print("=" * 60)

    if args.demo:
        from dataset import generate_demo_csv
        generate_demo_csv(os.path.join(args.data_root, 'demo_annotations.csv'))
        args.train_csv = os.path.join(args.data_root, 'demo_annotations_train.csv')
        args.val_csv = os.path.join(args.data_root, 'demo_annotations_val.csv')
        args.test_B_csv = os.path.join(args.data_root, 'demo_annotations_test_B.csv')
        args.test_C_csv = os.path.join(args.data_root, 'demo_annotations_test_C.csv')

    test_csvs = {}
    if args.test_B_csv:
        test_csvs['test_B'] = args.test_B_csv
    if args.test_C_csv:
        test_csvs['test_C'] = args.test_C_csv

    dataloaders = create_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csvs=test_csvs if test_csvs else None,
        us_dir=args.us_dir,
        path_dir=args.path_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.2 if args.val_csv is None else 0
    )

    print(f"Train batches: {len(dataloaders['train'])}")
    if dataloaders.get('val') is not None:
        print(f"Val batches: {len(dataloaders['val'])}")

    # ===== 打印训练集类别分布（诊断不平衡） =====
    print("\n" + "=" * 60)
    print("Training set class distribution:")
    print("=" * 60)
    train_labels = []
    for batch in dataloaders['train']:
        train_labels.extend(batch['subtype'].numpy().tolist())
    counter = Counter(train_labels)
    total = len(train_labels)
    for i, name in enumerate(SUBTYPE_NAMES):
        n = counter.get(i, 0)
        print(f"  {name:20s}: {n:5d}  ({100*n/total:.1f}%)")

    # 根据分布自动生成 class_weights（逆频）
    class_counts = np.array([counter.get(i, 1) for i in range(len(SUBTYPE_NAMES))],
                            dtype=np.float32)
    inv_freq = 1.0 / np.maximum(class_counts, 1)
    inv_freq = inv_freq / inv_freq.mean()    # 归一化到均值 1
    class_weights = torch.tensor(inv_freq, dtype=torch.float32).to(device)
    print(f"\nAuto class_weights (inverse freq, normalized): "
          f"{class_weights.cpu().numpy().round(3)}")

    # ===== 模型 =====
    print("\n" + "=" * 60)
    print("Building CMALA-Net model...")
    print("=" * 60)

    model = CMALANet(
        num_subtypes=5,
        num_biomarkers=4,
        embed_dim=args.embed_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        pretrained=args.pretrained and not args.demo,
        dinov2_path=args.dinov2_path
    ).to(device)

    trainable, total = model.get_trainable_params()
    print(f"Total parameters: {total/1e6:.2f}M")
    print(f"Trainable parameters (LoRA + heads): {trainable/1e6:.2f}M "
          f"({100*trainable/total:.1f}%)")

    # ===== 损失 / 优化器 =====
    criterion = CMALALoss(
        w_ce=args.w_ce,
        w_focal=args.w_focal,
        w_cont=args.w_cont,
        w_mmd=args.w_mmd,
        class_weights=class_weights
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr
    )

    scaler = make_grad_scaler(args.amp)

    # ===== 训练 =====
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        # 先更新 LR（warmup）
        cur_lr = scheduler.step(epoch)
        print(f"  LR (after warmup step): {cur_lr:.2e}")

        train_metrics = train_one_epoch(
            model, dataloaders['train'], criterion,
            optimizer, scaler, device,
            grad_clip=args.grad_clip,
            use_amp=args.amp,
            nan_lr_decay=args.nan_lr_decay,
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Macro-F1: {train_metrics['macro_f1']:.4f}, "
              f"Skipped: {train_metrics['n_skipped']}")
        print(f"  [loss parts] ce={train_metrics['ce']:.4f} "
              f"focal={train_metrics['focal']:.4f} "
              f"cont={train_metrics['contrastive']:.4f} "
              f"aux={train_metrics['auxiliary']:.4f} "
              f"mmd={train_metrics['mmd']:.4f}")

        if train_metrics['n_skipped'] >= len(dataloaders['train']):
            print("\n[FATAL] 整个 epoch 全部 NaN，权重已损坏。")
            break

        if dataloaders.get('val') is not None:
            val_metrics = evaluate(model, dataloaders['val'], criterion,
                                   device, use_amp=args.amp)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"Macro-F1: {val_metrics['macro_f1']:.4f}, "
                  f"Kappa: {val_metrics['kappa']:.4f}")

            for i, name in enumerate(SUBTYPE_NAMES):
                print(f"  {name:20s}: F1={val_metrics['per_class_f1'][i]:.3f}, "
                      f"Recall={val_metrics['per_class_recall'][i]:.3f}")

            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'args': vars(args)
                }, os.path.join(args.output_dir, args.save_name))
                print(f"  -> Saved best model (Macro-F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{args.patience})")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'args': vars(args)
            }, os.path.join(args.output_dir, args.last_name))

            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # ===== 最终测试 =====
    print("\n" + "=" * 60)
    print(f"Loading best model from epoch {best_epoch+1} (Macro-F1: {best_f1:.4f})")
    print("=" * 60)

    ckpt_path = os.path.join(args.output_dir, args.save_name)
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    if dataloaders.get('test'):
        results = {}
        for test_name in dataloaders['test']:
            print(f"\nEvaluating on {test_name}...")
            test_metrics = evaluate(model, dataloaders['test'][test_name],
                                    criterion, device, use_amp=args.amp)
            results[test_name] = test_metrics
            print(f"{test_name} Results:")
            print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"  Macro-F1:  {test_metrics['macro_f1']:.4f}")
            for i, name in enumerate(SUBTYPE_NAMES):
                print(f"  {name:20s}: F1={test_metrics['per_class_f1'][i]:.3f}, "
                      f"Recall={test_metrics['per_class_recall'][i]:.3f}")
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump({
                'best_epoch': best_epoch,
                'best_val_f1': best_f1,
                'test_results': {
                    k: {kk: vv for kk, vv in v.items()
                        if kk not in ['predictions', 'labels', 'probabilities']}
                    for k, v in results.items()
                }
            }, f, indent=2)

    print(f"\nTraining complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()