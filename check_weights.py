# -*- coding: utf-8 -*-
"""
校验 weights/ 下手动下载的各 backbone 权重能否被正确加载。

用法:
    python check_weights.py            # 校验全部
    python check_weights.py uni uni2h  # 只校验指定 backbone
"""
import os
import sys
import glob

import torch

# 权重文件约定 (用户在 HF/GDrive 页面手动下载后放入对应目录)
WEIGHTS_SPEC = {
    'uni':       dict(dir='weights/UNI',         file='pytorch_model.bin'),
    'uni2h':     dict(dir='weights/UNI2-h',      file='pytorch_model.bin'),
    'virchow2':  dict(dir='weights/Virchow2',    file='pytorch_model.bin'),
    'gigapath':  dict(dir='weights/GigaPath',    file='model.safetensors'),
    'hoptimus1': dict(dir='weights/H-optimus-1', file='pytorch_model.bin'),
    'chief':     dict(dir='weights/CHIEF',       file='CHIEF_CTransPath.pth',
                      extra=['weights/CHIEF.pth', 'weights/CHIEF/CHIEF.pth']),
}


def find_ckpt(bb):
    """在 spec 目录中寻找权重文件, 返回路径或 None"""
    spec = WEIGHTS_SPEC[bb]
    d = spec['dir']
    # 首选约定文件名 + 额外候选路径
    candidates = [os.path.join(d, spec['file'])] + list(spec.get('extra', []))
    for p in candidates:
        if os.path.isfile(p):
            return p
    # 兼容: 目录下任意 .safetensors/.bin/.pth
    for pat in ('*.safetensors', '*.bin', '*.pth', '*.pt'):
        hits = [f for f in glob.glob(os.path.join(d, pat))
                if 'optimizer' not in os.path.basename(f)
                and 'scheduler' not in os.path.basename(f)]
        if hits:
            return sorted(hits, key=os.path.getsize)[-1]
    return None


def check_one(bb):
    from backbones import create_path_backbone, BACKBONE_REGISTRY
    ckpt = find_ckpt(bb)
    display = BACKBONE_REGISTRY[bb]['display']
    if ckpt is None:
        spec = WEIGHTS_SPEC[bb]
        print(f"[缺失] {display:14s} 未找到 {spec['dir']}/{spec['file']}")
        return False

    size_gb = os.path.getsize(ckpt) / 1e9
    print(f"\n=== {display} ({bb}) ===")
    print(f"  权重: {ckpt} ({size_gb:.2f} GB)")

    try:
        wrapper = create_path_backbone(bb, pretrained=True, ckpt_path=ckpt)
    except Exception as e:
        print(f"  [FAIL] 加载失败: {type(e).__name__}: {str(e)[:200]}")
        return False

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = wrapper.forward_patches(x)
    n_expected = (224 // wrapper.patch_size) ** 2
    if bb == 'chief':
        n_expected = 49   # Swin-T 经 3 次 PatchMerging, 输出 7x7=49 tokens
    ok = out.shape == (1, n_expected, wrapper.feat_dim)
    print(f"  forward_patches: {tuple(out.shape)} "
          f"(期望 [1, {n_expected}, {wrapper.feat_dim}]) "
          f"| std={out.std().item():.3f}")
    print(f"  [{'OK' if ok else 'FAIL'}]")
    del wrapper
    return ok


def main():
    torch.set_grad_enabled(False)
    names = sys.argv[1:] or list(WEIGHTS_SPEC)
    results = {}
    for bb in names:
        if bb not in WEIGHTS_SPEC:
            print(f"未知 backbone: {bb}, 可选: {list(WEIGHTS_SPEC)}")
            continue
        try:
            results[bb] = check_one(bb)
        except Exception as e:
            print(f"[ERROR] {bb}: {type(e).__name__}: {e}")
            results[bb] = False

    print("\n" + "=" * 40)
    print("权重校验汇总:")
    for bb, ok in results.items():
        print(f"  {'[OK]' if ok else '[--]'} {bb}")
    ready = [b for b, ok in results.items() if ok]
    if ready:
        ckpt_map = ' '.join(f'{b}={find_ckpt(b)}' for b in ready if b != 'chief')
        chief = find_ckpt('chief')
        print("\n可直接用于对比实验的参数:")
        if ckpt_map:
            print(f"  --ckpt_map {ckpt_map}")
        if chief and 'chief' in ready:
            print(f"  --chief_ckpt {chief}")


if __name__ == '__main__':
    # Windows GBK 控制台兼容
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    main()
