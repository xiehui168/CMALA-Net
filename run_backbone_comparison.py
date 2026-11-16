# -*- coding: utf-8 -*-
"""
CMALA-Net 病理 backbone 对比实验批量运行脚本
==============================================
在同一训练/评估协议下比较 7 种病理基础模型 backbone:
  DINOv2-L, UNI, UNI2-h, Virchow2, Prov-GigaPath, H-optimus-1, CHIEF

每个 backbone 独立训练 + 测试, 结果汇总到:
  results/backbone_comparison.csv
  results/backbone_comparison.json
  results/backbone_comparison.md  

用法示例:
  # 全部 7 个 backbone
  python run_backbone_comparison.py \
      --train_csv data/train.csv --val_csv data/val.csv \
      --test_csvs data/test.csv \
      --epochs 100 --batch_size 8

  # 只跑其中几个
  python run_backbone_comparison.py --backbones uni uni2h virchow2 ...

  # 已完成训练的直接跳过
  python run_backbone_comparison.py --skip_existing ...

  # 只汇总已有结果生成表格
  python run_backbone_comparison.py --summarize_only

注意:
  - UNI/UNI2-h/Virchow2/GigaPath/H-optimus-1 为 gated 模型, 需要先在
    HuggingFace 申请权限并设置 HF_TOKEN; 国内建议 HF_ENDPOINT=https://hf-mirror.com
  - CHIEF 需单独下载 CHIEF.pth 并用 --chief_ckpt 指定
  - 详见 README.md 的 "Backbone 对比实验" 一节
"""
import os
import sys
import json
import argparse
import subprocess
import datetime

from backbones import BACKBONE_NAMES, BACKBONE_REGISTRY

# 对比顺序 (UNI 为主模型, 排第一)
DEFAULT_BACKBONES = ['uni', 'uni2h', 'virchow2', 'gigapath', 'hoptimus1',
                     'chief', 'dinov2']


def parse_args():
    parser = argparse.ArgumentParser(
        description='CMALA-Net backbone 对比实验批量运行')
    parser.add_argument('--backbones', type=str, nargs='*',
                        default=DEFAULT_BACKBONES,
                        choices=BACKBONE_NAMES,
                        help=f'要跑的 backbone 子集, 默认全部: {DEFAULT_BACKBONES}')

    # ---- 数据 ----
    parser.add_argument('--train_csv', type=str, default='data/train.csv')
    parser.add_argument('--val_csv', type=str, default='data/val.csv')
    parser.add_argument('--test_csvs', type=str, nargs='*',
                        default=['data/test.csv'],
                        help='测试集 CSV, 可多个')
    parser.add_argument('--us_dir', type=str, default='')
    parser.add_argument('--path_dir', type=str, default='')

    # ---- 训练超参 (对所有 backbone 一致, 保证公平比较) ----
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--extra_train_args', type=str, default='',
                        help='透传给 train.py 的额外参数, 如 "--grad_clip 5.0"')

    # ---- backbone 权重 ----
    parser.add_argument('--chief_ckpt', type=str, default=None,
                        help='CHIEF.pth 路径 (CHIEF 为 CTransPath 架构, '
                             '不能从 timm hub 加载)')
    parser.add_argument('--ckpt_map', type=str, nargs='*', default=None,
                        help='各 backbone 本地权重, 格式 name=path, 如 '
                             'uni=./weights/UNI virchow2=./weights/Virchow2')

    # ---- 运行控制 ----
    parser.add_argument('--ckpt_root', type=str, default='checkpoints/backbones')
    parser.add_argument('--results_root', type=str, default='results/backbones')
    parser.add_argument('--summary_dir', type=str, default='results')
    parser.add_argument('--skip_existing', action='store_true',
                        help='已有 results.json 的 backbone 跳过训练')
    parser.add_argument('--resume', action='store_true',
                        help='断点续训: last checkpoint 存在时从其继续训练')
    parser.add_argument('--eval_only', action='store_true',
                        help='跳过训练, 只对已有 checkpoint 跑 evaluate.py')
    parser.add_argument('--summarize_only', action='store_true',
                        help='不训练不评估, 仅汇总已有结果生成对比表格')
    return parser.parse_args()


def run_cmd(cmd, log_path):
    """运行子进程并实时输出 + 写日志"""
    print(f"\n$ {' '.join(cmd)}")
    with open(log_path, 'a', encoding='utf-8') as lf:
        lf.write(f"\n{'='*70}\n$ {' '.join(cmd)}\n{'='*70}\n")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            print(line, end='')
            lf.write(line)
        proc.wait()
    return proc.returncode


def train_backbone(bb, args, ckpt_dir, log_path):
    """训练单个 backbone"""
    cmd = [sys.executable, 'train.py',
           '--train_csv', args.train_csv,
           '--path_backbone', bb,
           '--output_dir', ckpt_dir,
           '--epochs', str(args.epochs),
           '--batch_size', str(args.batch_size),
           '--lr', str(args.lr),
           '--embed_dim', str(args.embed_dim),
           '--lora_rank', str(args.lora_rank),
           '--lora_alpha', str(args.lora_alpha),
           '--img_size', str(args.img_size),
           '--num_workers', str(args.num_workers),
           ]
    if args.val_csv:
        cmd += ['--val_csv', args.val_csv]
    # train.py 支持两个测试集槽位
    if args.test_csvs:
        cmd += ['--test_B_csv', args.test_csvs[0]]
        if len(args.test_csvs) > 1:
            cmd += ['--test_C_csv', args.test_csvs[1]]
    if args.us_dir:
        cmd += ['--us_dir', args.us_dir]
    if args.path_dir:
        cmd += ['--path_dir', args.path_dir]
    if args.amp:
        cmd += ['--amp']

    # 断点续训: last checkpoint 存在时从中断处继续
    if args.resume:
        last_ckpt = os.path.join(ckpt_dir, f'cmalanet_{bb}_last.pth')
        if os.path.isfile(last_ckpt):
            cmd += ['--resume', last_ckpt]
            print(f"[resume] {bb}: 从 {last_ckpt} 继续训练")

    # backbone 本地权重
    ckpt_map = dict(kv.split('=', 1) for kv in (args.ckpt_map or []))
    if bb == 'chief' and args.chief_ckpt:
        cmd += ['--path_ckpt', args.chief_ckpt]
    elif bb in ckpt_map:
        cmd += ['--path_ckpt', ckpt_map[bb]]

    if args.extra_train_args:
        cmd += args.extra_train_args.split()

    return run_cmd(cmd, log_path)


def eval_backbone(bb, args, ckpt_dir, res_dir, log_path):
    """对每个测试 CSV 运行 evaluate.py (生成混淆矩阵/ROC 等图)"""
    ckpt = os.path.join(ckpt_dir, f'cmalanet_{bb}_best.pth')
    if not os.path.isfile(ckpt):
        # 兼容自定义/旧命名
        alt = os.path.join(ckpt_dir, 'cmalanet_best.pth')
        if os.path.isfile(alt):
            ckpt = alt
        else:
            print(f"[skip] {bb}: 未找到 checkpoint")
            return 1
    rc = 0
    for i, test_csv in enumerate(args.test_csvs):
        out_dir = res_dir if len(args.test_csvs) == 1 \
            else os.path.join(res_dir, f'test_{i}')
        os.makedirs(out_dir, exist_ok=True)
        cmd = [sys.executable, 'evaluate.py',
               '--checkpoint', ckpt,
               '--test_csv', test_csv,
               '--output_dir', out_dir,
               '--batch_size', str(args.batch_size),
               '--img_size', str(args.img_size),
               '--path_backbone', bb,
               ]
        if args.us_dir:
            cmd += ['--us_dir', args.us_dir]
        if args.path_dir:
            cmd += ['--path_dir', args.path_dir]
        rc |= run_cmd(cmd, log_path)
    return rc


def collect_results(args):
    """汇总各 backbone 的测试结果"""
    rows = {}
    for bb in args.backbones:
        ckpt_dir = os.path.join(args.ckpt_root, bb)
        res_json = os.path.join(ckpt_dir, 'results.json')
        entry = {'display': BACKBONE_REGISTRY[bb]['display'],
                 'params_m': BACKBONE_REGISTRY[bb]['params_m']}
        if os.path.isfile(res_json):
            with open(res_json, 'r', encoding='utf-8') as f:
                r = json.load(f)
            entry['best_val_f1'] = r.get('best_val_f1')
            for test_name, m in (r.get('test_results') or {}).items():
                for key in ('accuracy', 'macro_f1', 'kappa'):
                    entry[f'{test_name}_{key}'] = m.get(key)
        else:
            # 尝试 evaluate.py 的 metrics_summary.json
            ms = os.path.join(args.results_root, bb, 'metrics_summary.json')
            if os.path.isfile(ms):
                with open(ms, 'r', encoding='utf-8') as f:
                    m = json.load(f)
                entry['test_accuracy'] = m.get('accuracy')
                entry['test_macro_f1'] = m.get('macro_f1')
                entry['test_kappa'] = m.get('kappa')
        rows[bb] = entry
    return rows


def write_summary(rows, args):
    """写 CSV / JSON / Markdown 对比表"""
    os.makedirs(args.summary_dir, exist_ok=True)

    # 收集所有指标列
    metric_keys = sorted({k for v in rows.values() for k in v
                          if k not in ('display', 'params_m')})

    # ---- JSON ----
    with open(os.path.join(args.summary_dir, 'backbone_comparison.json'),
              'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # ---- CSV ----
    import csv
    csv_path = os.path.join(args.summary_dir, 'backbone_comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['backbone', 'display_name', 'params_m'] + metric_keys)
        for bb, v in rows.items():
            w.writerow([bb, v['display'], v['params_m']] +
                       [v.get(k, '') for k in metric_keys])

    # ---- Markdown (论文表格) ----
    md_path = os.path.join(args.summary_dir, 'backbone_comparison.md')
    core = ['accuracy', 'macro_f1', 'kappa']

    def cell(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else '-'

    lines = ['## CMALA-Net with Different Pathology Backbones', '',
             '| Pathology Backbone | Params |',
             ]
    # 每个测试集 3 个核心指标
    test_names = sorted({k.rsplit('_', 1)[0] for k in metric_keys
                         if k.rsplit('_', 1)[-1] in core})
    header = '| Pathology Backbone | Params |'
    sep = '|---|---|'
    for tn in test_names:
        for m in core:
            header += f' {tn} {m.capitalize()} |'
            sep += '---|'
    lines = [lines[0], '', header, sep]
    for bb, v in rows.items():
        line = f"| {v['display']} | {v['params_m']}M |"
        for tn in test_names:
            for m in core:
                line += f" {cell(v.get(f'{tn}_{m}'))} |"
        lines.append(line)
    lines += ['',
              f"Best val Macro-F1 详见 backbone_comparison.json; "
              f"生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M}"]
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n[OK] 对比结果已保存:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"  {os.path.join(args.summary_dir, 'backbone_comparison.json')}")

    # 控制台打印简表
    print(f"\n{'Backbone':<16} {'Params':>8}", end='')
    for tn in test_names:
        print(f" | {tn+' Acc':>10} {'MacF1':>7} {'Kappa':>7}", end='')
    print()
    print('-' * 80)
    for bb, v in rows.items():
        print(f"{v['display']:<16} {str(v['params_m'])+'M':>8}", end='')
        for tn in test_names:
            print(f" | {cell(v.get(tn+'_accuracy')):>10} "
                  f"{cell(v.get(tn+'_macro_f1')):>7} "
                  f"{cell(v.get(tn+'_kappa')):>7}", end='')
        print()


def main():
    args = parse_args()

    # Windows GBK 控制台: 避免 tqdm/日志中的非 GBK 字符导致 UnicodeEncodeError
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
        sys.stderr.reconfigure(errors='replace')

    print("=" * 70)
    print("CMALA-Net 病理 Backbone 对比实验")
    print("=" * 70)
    print(f"Backbones: {args.backbones}")
    print(f"Train CSV: {args.train_csv}  Val CSV: {args.val_csv}")
    print(f"Test CSVs: {args.test_csvs}")

    if args.summarize_only:
        rows = collect_results(args)
        write_summary(rows, args)
        return

    log_path = os.path.join(args.summary_dir, 'backbone_comparison.log')
    os.makedirs(args.summary_dir, exist_ok=True)

    for bb in args.backbones:
        ckpt_dir = os.path.join(args.ckpt_root, bb)
        res_dir = os.path.join(args.results_root, bb)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"Backbone: {BACKBONE_REGISTRY[bb]['display']} ({bb})")
        print("=" * 70)

        results_json = os.path.join(ckpt_dir, 'results.json')
        if args.eval_only:
            pass  # 跳过训练
        elif args.skip_existing and os.path.isfile(results_json):
            print(f"[skip] {bb} 已有训练结果 ({results_json})")
        else:
            rc = train_backbone(bb, args, ckpt_dir, log_path)
            if rc != 0:
                print(f"[ERROR] {bb} 训练失败 (rc={rc}), 继续下一个 backbone")
                continue

        # 可视化评估 (失败不阻断汇总)
        try:
            eval_backbone(bb, args, ckpt_dir, res_dir, log_path)
        except Exception as e:
            print(f"[WARNING] {bb} evaluate 失败: {e}")

    # ---- 汇总 ----
    rows = collect_results(args)
    write_summary(rows, args)
    print("\n全部 backbone 对比实验完成!")


if __name__ == '__main__':
    main()
