#!/usr/bin/env python
"""
DINOv2 权重下载辅助脚本
解决国内网络下载慢/超时/SSL错误问题
"""
import os
import sys

def print_guide():
    print("="*70)
    print("DINOv2 预训练权重下载指南 (facebook/dinov2-large ~1.2GB)")
    print("="*70)
    
    print("""
【方式1: 使用国内镜像自动下载 (推荐)】

在 Windows CMD/PowerShell 中先执行:
  set HF_ENDPOINT=https://hf-mirror.com

然后运行 demo 或训练脚本:
  python demo.py --pretrained
  python train.py --demo

【方式2: 手动下载后本地加载】

1. 浏览器打开镜像站: https://hf-mirror.com/facebook/dinov2-large
2. 下载以下两个文件:
   - config.json
   - model.safetensors (约1.2GB)
3. 在项目目录下创建 models/dinov2-large/ 文件夹
4. 将下载的两个文件放入该文件夹
5. 训练时指定本地路径:
   python train.py --dinov2_path ./models/dinov2-large ...
   评估时同样:
   python evaluate.py --dinov2_path ./models/dinov2-large ...

【方式3: 使用modelscope (国内阿里云)】

pip install modelscope
然后在代码中使用:
  from modelscope import snapshot_download
  model_dir = snapshot_download('AI-ModelScope/dinov2-large')
  model = Dinov2Model.from_pretrained(model_dir)

【方式4: 不使用预训练权重 (快速验证代码)】

直接运行快速演示，完全不需要下载:
  python demo.py
  python train.py --demo --no_pretrained

注意: 不使用预训练权重仅用于验证代码流程能跑通，
      实际训练论文结果必须加载预训练权重!
""")
    print("="*70)


def download_with_mirror():
    """尝试使用镜像自动下载"""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    
    print("正在从 hf-mirror.com 下载 DINOv2-large...")
    print("如果速度慢请改用方式2手动下载\n")
    
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            "facebook/dinov2-large",
            resume_download=True,
        )
        print(f"\n下载成功! 模型保存在: {path}")
        print(f"训练时可直接使用，或指定路径: --dinov2_path {path}")
        return True
    except Exception as e:
        print(f"\n自动下载失败: {e}")
        print("请改用方式2手动下载")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--download":
        success = download_with_mirror()
        sys.exit(0 if success else 1)
    else:
        print_guide()
