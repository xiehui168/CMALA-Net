"""
CMALA-Net 评估与可视化脚本
包含: 混淆矩阵、ROC曲线、Grad-CAM、跨模态对齐可视化
"""
import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    cohen_kappa_score, f1_score, accuracy_score
)
from sklearn.manifold import TSNE

from model import CMALANet
# 修复导入：直接导入真实存在的 PathologyDataset，同时保留原名称兼容
from dataset import PathologyDataset, SUBTYPE_NAMES, BIOMARKER_NAMES, create_dataloaders
# 原地别名，不修改 dataset.py 任何内容
BreastCancerDataset = PathologyDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CMALA-Net')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--test_csv', type=str, required=True, help='测试集CSV')
    parser.add_argument('--us_dir', type=str, default='')
    parser.add_argument('--path_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--dinov2_path', type=str, default=None,
                        help='本地DINOv2模型路径（离线使用）')
    return parser.parse_args()


@torch.no_grad()
def get_predictions(model, dataloader, device, debug_shapes=False):
    """获取所有预测结果"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_us_feats = []
    all_path_feats = []
    all_transport_plans = []

    for step, batch in enumerate(dataloader):
        us = batch['us_img'].to(device)
        path = batch['path_img'].to(device)
        labels = batch['subtype']

        outputs = model(us, path, return_features=True)

        if debug_shapes and step == 0:
            print("[debug] global_us shape:     ", outputs['global_us'].shape)
            print("[debug] global_path shape:   ", outputs['global_path'].shape)
            print("[debug] transport_plan shape:", outputs['transport_plan'].shape)

        probs = torch.softmax(outputs['subtype_logits'], dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_us_feats.append(outputs['global_us'].cpu().numpy())
        all_path_feats.append(outputs['global_path'].cpu().numpy())
        all_transport_plans.append(outputs['transport_plan'].cpu().numpy())

    # 处理 transport_plan：
    # - 若模型返回 [B, N, M]，直接 concat -> [num_samples, N, M]
    # - 若模型返回 [N, M]（batch 内聚合），则每个 batch 只有一份，需特殊处理
    plans = all_transport_plans
    if len(plans) > 0 and plans[0].ndim == 2:
        # 每个 batch 一份聚合矩阵，堆成 [num_batches, N, M]
        transport_plans = np.stack(plans, axis=0)
    else:
        transport_plans = np.concatenate(plans, axis=0)

    return {
        'preds': np.array(all_preds),
        'labels': np.array(all_labels),
        'probs': np.array(all_probs),
        'us_feats': np.concatenate(all_us_feats, axis=0),
        'path_feats': np.concatenate(all_path_feats, axis=0),
        'transport_plans': transport_plans
    }


def plot_confusion_matrix(results, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(results['labels'], results['preds'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=SUBTYPE_NAMES, yticklabels=SUBTYPE_NAMES, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('CMALA-Net Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curves(results, save_path):
    """绘制ROC曲线（One-vs-Rest）"""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (name, color) in enumerate(zip(SUBTYPE_NAMES, colors)):
        y_true = (results['labels'] == i).astype(int)
        y_score = results['probs'][:, i]

        # 若某一类在测试集中完全不存在，跳过，避免 roc_curve 报错
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            print(f"  [skip] {name}: 测试集中无正/负样本，跳过 ROC")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {save_path}")


def plot_tsne(results, save_path):
    """t-SNE特征可视化"""
    us = results['us_feats']
    path = results['path_feats']
    labels = results['labels']
    n = len(labels)

    # perplexity 必须 < n，且通常要求 n > 3*perplexity
    perplexity = min(30, max(5, (n - 1) // 3))
    if n < 6:
        print(f"[warn] 样本数过少 (n={n})，跳过 t-SNE 可视化")
        return

    # 共享嵌入：US 和 Path 放在同一坐标系中
    combined = np.concatenate([us, path], axis=0)
    emb = TSNE(n_components=2, random_state=42, perplexity=perplexity,
               init='pca', learning_rate='auto').fit_transform(combined)
    us_2d, path_2d = emb[:n], emb[n:]

    # 融合特征用于按亚型着色
    fused = np.concatenate([us, path], axis=1)
    fused_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                    init='pca', learning_rate='auto').fit_transform(fused)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 按真实标签着色
    ax = axes[0]
    for i, name in enumerate(SUBTYPE_NAMES):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(fused_2d[mask, 0], fused_2d[mask, 1],
                   c=colors[i % len(colors)], label=name, alpha=0.6, s=20)
    ax.set_title('t-SNE by True Subtype (fused features)')
    ax.legend()

    # 按模态着色（共享嵌入坐标系）
    ax = axes[1]
    ax.scatter(us_2d[:, 0], us_2d[:, 1], c='blue',
               label='Ultrasound', alpha=0.5, s=20)
    ax.scatter(path_2d[:, 0], path_2d[:, 1], c='red',
               label='Pathology', alpha=0.5, s=20)
    ax.set_title('t-SNE by Modality (shared embedding)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t-SNE visualization saved to {save_path}")


def plot_transport_plan(results, save_path, num_samples=4):
    """可视化跨模态最优传输对齐矩阵"""
    plans = results['transport_plans']
    if len(plans) == 0:
        print("[warn] 无 transport plan 可绘制，跳过")
        return

    num_samples = min(num_samples, len(plans))

    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        T = plans[i]
        if T.ndim == 3:
            T = T.squeeze(0)
        ax = axes[i]
        im = ax.imshow(T[:16, :16], cmap='viridis', aspect='auto')
        ax.set_title(f'Sample {i+1}\n(True: {SUBTYPE_NAMES[results["labels"][i]]})')
        ax.set_xlabel('Path Patches')
        ax.set_ylabel('US Patches')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('Sinkhorn Optimal Transport Alignment Matrix', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Transport plan visualization saved to {save_path}")


def print_metrics(results):
    """打印详细评估指标"""
    preds = results['preds']
    labels = results['labels']
    probs = results['probs']

    print("\n" + "=" * 60)
    print("CMALA-Net Evaluation Results")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:      {accuracy_score(labels, preds):.4f}")
    print(f"  Macro-F1:      {f1_score(labels, preds, average='macro'):.4f}")
    print(f"  Cohen's Kappa: {cohen_kappa_score(labels, preds):.4f}")

    # 只在测试集中出现的类别才作为 target_names，避免 classification_report 报错
    present_labels = sorted(np.unique(labels).tolist())
    present_names = [SUBTYPE_NAMES[i] for i in present_labels if i < len(SUBTYPE_NAMES)]
    print(f"\nPer-Class Metrics:")
    print(classification_report(labels, preds,
                                labels=present_labels,
                                target_names=present_names,
                                digits=3, zero_division=0))

    # 高置信度病例比例 (confidence >= 90%)
    max_probs = probs.max(axis=1)
    high_conf_mask = max_probs >= 0.9
    print(f"\nClinical Utility Analysis:")
    if high_conf_mask.sum() > 0:
        high_conf_ratio = high_conf_mask.mean()
        high_conf_acc = accuracy_score(labels[high_conf_mask], preds[high_conf_mask])
        print(f"  High-confidence cases (>=90%): {high_conf_ratio:.1%}")
        print(f"  Accuracy on high-confidence: {high_conf_acc:.4f}")
    else:
        print(f"  High-confidence cases (>=90%): 0.0%")
        print(f"  Accuracy on high-confidence: N/A")

    # TNBC Recall（论文重点指标），安全查找
    if 'TNBC' in SUBTYPE_NAMES:
        tnbc_idx = SUBTYPE_NAMES.index('TNBC')
        tnbc_mask = labels == tnbc_idx
        if tnbc_mask.sum() > 0:
            tnbc_recall = (preds[tnbc_mask] == labels[tnbc_mask]).mean()
            print(f"  TNBC Recall: {tnbc_recall:.3f}")
        else:
            print(f"  TNBC Recall: N/A (测试集中无 TNBC 样本)")
    else:
        print(f"  TNBC Recall: N/A (SUBTYPE_NAMES 中不存在 'TNBC')")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get('args', {})

    model = CMALANet(
        embed_dim=ckpt_args.get('embed_dim', 1024),
        lora_rank=ckpt_args.get('lora_rank', 16),
        lora_alpha=ckpt_args.get('lora_alpha', 32),
        sinkhorn_eps=ckpt_args.get('sinkhorn_eps', 0.06),
        sinkhorn_iters=ckpt_args.get('sinkhorn_iters', 50),
        pretrained=False,
        dinov2_path=args.dinov2_path
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # 修复数据加载：直接传CSV路径，替换不存在的 augment 参数
    test_dataset = BreastCancerDataset(
        args.test_csv, img_size=args.img_size, is_train=False,
        us_dir=args.us_dir, path_dir=args.path_dir
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"Test set size: {len(test_dataset)}")
    print(f"SUBTYPE_NAMES: {SUBTYPE_NAMES}")

    # 获取预测
    print("Running inference...")
    results = get_predictions(model, test_loader, device, debug_shapes=True)

    # 打印指标
    print_metrics(results)

    # 生成可视化
    print("\nGenerating visualizations...")
    plot_confusion_matrix(results, os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_roc_curves(results, os.path.join(args.output_dir, 'roc_curves.png'))
    plot_tsne(results, os.path.join(args.output_dir, 'tsne_visualization.png'))
    plot_transport_plan(results, os.path.join(args.output_dir, 'transport_plan.png'))

    # 保存数值结果
    results_summary = {
        'accuracy': float(accuracy_score(results['labels'], results['preds'])),
        'macro_f1': float(f1_score(results['labels'], results['preds'], average='macro')),
        'kappa': float(cohen_kappa_score(results['labels'], results['preds'])),
        'num_samples': int(len(results['labels'])),
    }
    with open(os.path.join(args.output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()