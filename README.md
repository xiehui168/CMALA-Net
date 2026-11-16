# CMALA-Net

CMALA-Net 使用超声图像与病理图像进行乳腺癌分子亚型分类，并可预测 ER、PR、HER2、Ki67 等生物标志物。项目支持切换病理图像 backbone，便于复现实验和比较不同预训练模型。

## 项目结构

```text
CMALA-Net/
├── backbones.py                 # 病理 backbone 注册表与加载器
├── dataset.py                   # CSV、超声和病理图像数据集
├── model.py                     # CMALA-Net 模型
├── train.py                     # 训练与测试
├── evaluate.py                  # 使用 checkpoint 独立评估
├── run_backbone_comparison.py   # 批量训练、评估及汇总
├── check_weights.py             # 本地 backbone 权重校验
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── US/                       # 超声图像
│   └── path/                     # 病理图像
├── weights/                      # 手动下载的预训练 backbone 权重
├── checkpoints/                  # 训练 checkpoint
└── results/                      # 指标、图表与对比表
```

## 环境安装

建议使用 Python 3.10 或 3.11，并在有 NVIDIA GPU 的环境安装与 CUDA 版本匹配的 PyTorch。然后在项目根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

处理 NRRD 格式超声图像时还需要：

```powershell
pip install pynrrd
```

先用演示数据检查训练流程：

```powershell
python demo.py
```

## 数据准备

训练、验证和测试 CSV 使用同一列结构：

| 列名 | 含义及格式 |
|---|---|
| `patient_id` | 患者标识 |
| `us_path` | 超声图像路径；支持 NRRD，也可由 OpenCV 解码的图像 |
| `path_path` | 病理图像路径；多张图用分号 `;` 分隔，预处理后取平均 |
| `subtype` | 必须是整数类别编号 0–4 |
| `ER`, `PR`, `HER2` | 数值型标志物标签 |
| `Ki67` | 数值型百分比，数据加载时除以 100 |
| `center` | 中心编号；`A`、`B`、`C` 映射为域标签 |

亚型编号：`0=Luminal A`、`1=Luminal B(HER2-)`、`2=Luminal B(HER2+)`、`3=HER2-enriched`、`4=TNBC`。

```csv
patient_id,us_path,path_path,subtype,ER,PR,HER2,Ki67,center
857,US/A/0857.nrrd,path/A/0857.jpg,4,0,0,0,82,A
550,US/A/0550.nrrd,path/A/0550.jpg;path/A/1550.jpg,3,0,0,1,60,A
```

相对图像路径会尝试从当前目录、传入的图像根目录和 CSV 所在目录解析。若 CSV 位于 `data/`，示例中的 `US/A/...` 和 `path/A/...` 可对应 `data/US/A/...` 和 `data/path/A/...`。训练前请确认每个路径都能找到图像；加载失败会打印警告并使用零图，不会生成随机图像。

## 训练与评估

使用默认病理 backbone（UNI）训练：

```powershell
python train.py `
    --train_csv data/train.csv `
    --val_csv data/val.csv `
    --test_B_csv data/test.csv `
    --path_backbone uni `
    --path_ckpt weights/UNI/pytorch_model.bin `
    --epochs 100 --batch_size 8 --num_workers 0
```

训练参数包括 `--lr`（默认 `3e-4`）、`--patience`（默认 `50`）、`--img_size`（默认 `224`）、`--lora_rank`（默认 `16`）和 `--lora_alpha`（默认 `32`）。显存不足时先降低 `--batch_size`；Windows 上建议从 `--num_workers 0` 开始。

模型权重保存在 `checkpoints/`。设置 `--output_dir` 后，best/last checkpoint 默认命名为 `cmalanet_<backbone>_best.pth` 和 `cmalanet_<backbone>_last.pth`。训练中断后可续训：

```powershell
python train.py --train_csv data/train.csv --val_csv data/val.csv `
    --path_backbone uni --path_ckpt weights/UNI/pytorch_model.bin `
    --output_dir checkpoints/uni --epochs 100 --resume
```

独立评估已有 checkpoint：

```powershell
python evaluate.py `
    --checkpoint checkpoints/uni/cmalanet_uni_best.pth `
    --test_csv data/test.csv `
    --path_backbone uni `
    --batch_size 8 --img_size 224
```

评估图表和指标输出到 `results/` 或指定的 `--output_dir`。

## 病理 Backbone

`--path_backbone` 支持以下选项。权重维度和 patch 数以当前加载实现为准：

| 参数 | 模型 | 架构 | 特征维度 | Patch 数 (224 输入) |
|---|---|---|---:|---:|
| `uni` | UNI | ViT-L/16 | 1024 | 196 |
| `uni2h` | UNI2-h | ViT-g/14 | 1536 | 256 |
| `virchow2` | Virchow2 | ViT-H/14 | 1280 | 256 |
| `gigapath` | Prov-GigaPath | ViT-g/16 | 1536 | 196 |
| `hoptimus1` | H-optimus-1 | ViT-g/14 | 1536 | 256 |
| `chief` | CHIEF / CTransPath | Swin-T | 768 | 49 |
| `dinov2` | DINOv2-L | ViT-L/14 | 1024 | 256 |

### 获取和放置权重

权重仓库的访问条件由各模型提供方管理，部分 Hugging Face 仓库需要先申请权限。可从官方模型页面下载，并按以下路径放置：

| 模型 | 本地权重路径 |
|---|---|
| UNI | `weights/UNI/pytorch_model.bin` |
| UNI2-h | `weights/UNI2-h/pytorch_model.bin` |
| Virchow2 | `weights/Virchow2/model.safetensors` 或 `pytorch_model.bin` |
| Prov-GigaPath | `weights/GigaPath/pytorch_model.bin` 或 `model.safetensors` |
| H-optimus-1 | `weights/H-optimus-1/pytorch_model.bin` |
| CHIEF | `weights/CHIEF/CHIEF_CTransPath.pth` |

CHIEF 权重：[Google Drive 下载页](https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV)。UNI、UNI2-h、Virchow2、GigaPath 和 H-optimus-1 的仓库链接见 [backbones.py](backbones.py) 与相应模型卡。

下载后先校验权重是否能按当前代码加载：

```powershell
python check_weights.py
python check_weights.py uni chief
```

校验通过后，单独训练某模型时用 `--path_backbone` 和 `--path_ckpt` 指定模型与权重。例如：

```powershell
python train.py --train_csv data/train.csv --val_csv data/val.csv `
    --path_backbone virchow2 `
    --path_ckpt weights/Virchow2/model.safetensors `
    --output_dir checkpoints/virchow2 --epochs 100 --batch_size 4
```

H-optimus-1 使用其专用归一化统计量；其他 backbone 使用 ImageNet 统计量。

## 多 Backbone 对比

批量训练、评估并汇总结果：

```powershell
python run_backbone_comparison.py `
    --backbones uni uni2h virchow2 gigapath hoptimus1 chief dinov2 `
    --train_csv data/train.csv --val_csv data/val.csv `
    --test_csvs data/test.csv `
    --chief_ckpt weights/CHIEF/CHIEF_CTransPath.pth `
    --ckpt_map `
        uni=weights/UNI/pytorch_model.bin `
        uni2h=weights/UNI2-h/pytorch_model.bin `
        virchow2=weights/Virchow2/model.safetensors `
        gigapath=weights/GigaPath/pytorch_model.bin `
        hoptimus1=weights/H-optimus-1/pytorch_model.bin `
    --epochs 100 --batch_size 4 --num_workers 0
```

常用运行选项：

- `--backbones uni uni2h`：只运行指定模型。
- `--resume`：检测对应的 `last` checkpoint 并续训。
- `--skip_existing`：已有 `results.json` 的模型跳过训练。
- `--eval_only`：只评估已有 best checkpoint。
- `--summarize_only`：只从现有结果文件重新生成汇总。
- `--amp`：启用混合精度训练。
- `--ckpt_root`、`--results_root`、`--summary_dir`：更改输出目录。

默认输出：

| 路径 | 内容 |
|---|---|
| `checkpoints/backbones/<name>/` | best/last checkpoint 和训练结果 JSON |
| `results/backbones/<name>/` | 单模型评估指标和可视化图表 |
| `results/backbone_comparison.csv` | 指标 CSV |
| `results/backbone_comparison.json` | 完整指标 JSON |
| `results/backbone_comparison.md` | Markdown 对比表 |
| `results/backbone_comparison.log` | 批量运行日志 |

若汇总表出现 `-`，表示对应模型的结果 JSON 中没有该指标；检查日志和 `checkpoints/backbones/<name>/results.json`，必要时使用 `--resume` 续训，或使用 `--eval_only` 重跑评估，再用 `--summarize_only` 更新汇总表。

## 结果解读

核心指标包括 Accuracy、Macro-F1 和 Cohen's Kappa。类别不均衡时建议重点报告 Macro-F1 和各类别指标。多中心划分与独立测试集应在论文中明确说明；不同 backbone 的对比应保持相同的数据划分、训练轮数、优化器和模型超参数。

模型权重和数据需遵循其各自的授权与使用条款。请勿将受限权重、临床图像或患者可识别信息提交到公开代码仓库。
