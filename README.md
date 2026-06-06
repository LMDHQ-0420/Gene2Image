# Gene2Image：可学习结构化通路瓶颈的基因到病理图像生成

从单细胞基因表达生成 H&E 病理图像，将 GeneFlow 的 RNA 编码器替换为**端到端可学习的结构化通路瓶颈编码器**：以固定的通路-基因二值掩码约束稀疏连接、为每个 (通路, 基因) 对赋予可学习权重向量，在 GeneFlow 的「无结构」与 MUPAD 的「固定打分」之间取得平衡。整流流 + UNet 生成主干完全复用 GeneFlow 不变，使性能差异可干净归因到编码器。

- 研究/实验设计：`docs/idea_report.md`
- 详细实现指南：`docs/implementation.md`
- 开发日志：`docs/dev_log.md`

代码基于 [GeneFlow](https://github.com/wangmengbo/GeneFlow)（NeurIPS 2025）改写，位于 `code/`。

---

## 1 环境

使用 conda 环境 `zw@Gene2Image`（PyTorch 2.2.2 + cu121）。

```bash
# 已知坑：环境内 mkl 2025 与 torch 2.2.2 冲突（undefined symbol: iJIT_NotifyEvent）
pip install "mkl==2024.0.0"        # 修复 torch import
pip install gseapy torchmetrics    # 通路库 + 训练依赖（原 requirements 缺）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # 应为 2.2.2 True
```

依赖清单见 `code/requirements.txt`（不含 torch/torchvision/torchaudio，需按 cu121 自行安装）。

可选：`cellpose`（仅 segmentation 版 spatial loss 用，不影响主流程）；`UNI2-h` / `HE2RNA` 权重（病理专用指标，缺失时自动降级跳过）。

---

## 2 数据

三个预处理 Xenium 黑色素瘤样本（已下载解压于 `code/data/processed_data/`）：

| 短名 | 目录 | 基因数 G | 用途 |
|------|------|---------|------|
| c1 | `Xenium_V1_hSkin_Melanoma_Base_FFPE` | 282 | 主实验 |
| c2 | `Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE` | 382 | 主实验 + 跨数据集 |
| p1 | `Xenium_Prime_Human_Skin_FFPE` | 5006 | 主实验 + 通路扩展 |

每个数据集含 `adata.h5ad`（log1p 归一化的基因表达）+ `cell_patch_256_aux/input/`（256×256×4 的 H&E+DAPI cell 图像）。

> 若需重新下载：来自 Zenodo `records/17429142`。本地已就绪，无需下载。

### 2.1 数据准备（训练前一次性）

```bash
cd code
PY=/home/ft/anaconda3/envs/zw@Gene2Image/bin/python

# (1) 修复 cell_image_paths.json 内的失效绝对路径（原作者集群路径 -> 本地）
$PY scripts/fix_image_paths.py \
  --json data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths.json \
  --local_root data/processed_data
# C2、P1 同样各跑一次（实验脚本会在缺失时自动补跑）

# (2) 构造通路掩码（Hallmark∩gene_names，去<3基因通路；含 real/rand/none + ssGSEA 权重）
$PY scripts/build_pathway_mask.py --adata data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad --prefix c1 --db hallmark --out_dir data/pathway_masks
$PY scripts/build_pathway_mask.py --adata data/processed_data/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/adata.h5ad --prefix c2 --db hallmark --out_dir data/pathway_masks
$PY scripts/build_pathway_mask.py --adata data/processed_data/Xenium_Prime_Human_Skin_FFPE/adata.h5ad --prefix p1 --db hallmark --out_dir data/pathway_masks
# 通路扩展消融（P1 + Reactome）
$PY scripts/build_pathway_mask.py --adata data/processed_data/Xenium_Prime_Human_Skin_FFPE/adata.h5ad --prefix p1 --db hallmark_reactome --out_dir data/pathway_masks
```

产物在 `code/data/pathway_masks/{ds}_{db}_{real,rand,none}.npz`（含 `A [P,G]`、`pathway_names`、`gene_names`，real 还含 `W_ssgsea`）。

---

## 3 模型变体（三正交开关）

| 变体 | 编码器 CLI | 翻转的开关 | 作用 |
|------|-----------|-----------|------|
| **gene2image**（主方法）| `--encoder_type pathway --pathway_mask {ds}_hallmark_real.npz` | — | RQ1 满配 |
| **geneflow**（基线）| `--encoder_type rna` | 无通路编码器 | SOTA/下界 |
| **randpath** | `... --pathway_mask {ds}_hallmark_rand.npz` | 真实→随机掩码 | RQ2 机制 |
| **pathprior** | `... --pathway_mask {ds}_hallmark_real.npz --no_learnable_pathway` | 可学习→固定ssGSEA | RQ3 击穿MUPAD |
| **notrans** | `... --pathway_mask ..._real.npz --no_pathway_transformer` | 去 Pathway Transformer | 通路协同 |
| **nomask** | `... --pathway_mask {ds}_hallmark_none.npz` | 稀疏→全连接 | 结构化稀疏 |

维度链：`[B,G] → 通路token[B,P,48] → (+CLS) → 细胞嵌入[B,256] → [B,512]`（硬对齐 UNet）。

---

## 4 运行

> ⚠️ 显存：本机 GPU 为 V100 32GB（idea_report 估算的是 H100 80GB）。建议 `--batch_size` 调小并加 `--use_amp`。

### 4.1 单次训练

```bash
cd code
$PY rectified/rectified_main.py \
  --model_type single --img_size 256 --img_channels 4 \
  --adata data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad \
  --image_paths data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths_local.json \
  --output_dir results/gene2image_c1_seed42 \
  --encoder_type pathway --pathway_mask data/pathway_masks/c1_hallmark_real.npz \
  --batch_size 16 --epochs 100 --gen_steps 100 --seed 42 --use_amp
```

快速烟测：加 `--debug --debug_samples 200 --epochs 1`。

### 4.2 批量主实验 + 消融（6 变体 × 3 数据集 × 3 种子）

```bash
cd code
PY=$PY bash scripts/run_experiments.sh all                 # 全跑（很久）
PY=$PY bash scripts/run_experiments.sh gene2image c1 42     # 单个
# 可调环境变量：BATCH_SIZE / EPOCHS / GEN_STEPS / DB / EXTRA
```

### 4.3 跨数据集泛化（c1→c2 / c2→c1 / c1→p1）

```bash
cd code
PY=$PY bash scripts/run_cross_dataset.sh all
# 自动构造通路名对齐的源/目标掩码（scripts/build_cross_masks.py），在源训练、目标评估
```

### 4.4 评估

```bash
$PY rectified/rectified_evaluate.py \
  --model_path results/gene2image_c1_seed42/checkpoints/best_checkpoint.pt \
  --model_type single --img_size 256 --img_channels 4 \
  --adata <adata> --image_paths <imgpaths_local.json> \
  --output_dir results/gene2image_c1_seed42/eval --gen_steps 100 --use_amp
# 产出 evaluation_summary.json（FID/SSIM/PSNR/UNI2-h），pathway checkpoint 自动重建编码器
```

### 4.5 通路可解释性（RQ4，仅 single）

```bash
$PY analysis/pathway_interpret.py \
  --model_path results/gene2image_c1_seed42/checkpoints/best_checkpoint.pt \
  --adata <adata> --image_paths <imgpaths_local.json> \
  --out_dir results/interpret/c1 --analysis A B C
# A 内生性(CLS通路注意力熵) / B 生物合理性(与GSEA重合) / C 因果(通路干预形态偏移)
```

### 4.6 结果汇总

```bash
$PY scripts/summarize_results.py --results_root results --out_dir results
# 产出 results/summary_main.csv 与 results/ablation/summary.csv（多种子均值±std）
```

---

## 5 代码结构（本研究新增/改动）

```
code/
├── src/
│   ├── pathway_encoder.py     # [新] 通路编码器：掩码嵌入 + Pathway Transformer + CLS
│   ├── single_model.py        # [改] encoder_type 分支 + l1_penalty()
│   ├── multi_model.py         # [改] 同上（multi 附线）
│   └── utils.py               # [改] CLI 新增 Pathway Encoder 参数组
├── rectified/
│   ├── rectified_main.py      # [改] 载掩码 + 列数校验 + model_config 入 checkpoint
│   ├── rectified_train.py     # [改] L1 解耦为 compute_l1_penalty + torch2.2 AMP 兼容
│   └── rectified_evaluate.py  # [改] 修损坏 import + pathway 编码器重建
├── scripts/
│   ├── fix_image_paths.py     # [新] cell_image_paths 路径重映射
│   ├── build_pathway_mask.py  # [新] 通路掩码构造（real/rand/none + ssGSEA）
│   ├── build_cross_masks.py   # [新] 跨数据集通路名对齐掩码
│   ├── run_experiments.sh     # [新] 6变体×3数据集×3种子
│   ├── run_cross_dataset.sh   # [新] 跨数据集泛化
│   └── summarize_results.py   # [新] 多种子结果汇总
├── analysis/
│   └── pathway_interpret.py   # [新] RQ4 三子分析（内生/合理/因果）
├── notebooks/                 # [新] 关键步骤可视化
└── data/
    ├── processed_data/        # 三数据集（已就绪）
    └── pathway_masks/         # 通路掩码 .npz
```

整流流主干（`rectified/rectified_flow.py`）与 UNet（`src/unet.py`）完全复用 GeneFlow，未改动。
