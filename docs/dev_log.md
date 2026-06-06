# 开发日志 — Gene2Image：可学习结构化通路瓶颈编码器
> 创建时间：2026-06-07 | 最后更新：2026-06-07
> 关联实现指南：docs/implementation.md

## 项目概览
| 项目 | 内容 |
|------|------|
| 研究方向 | 从单细胞基因表达生成 H&E 病理图像，将 RNA 编码器替换为可学习结构化通路瓶颈 |
| 实现策略 | 强 baseline 改写：GeneFlow（code/ 已 clone）|
| 框架 | PyTorch 2.2.2 + cu121 |
| 环境 | conda `zw@Gene2Image`（mkl 已降级至 2024.0.0 修复 torch import）|
| 主干 | single 为主、multi 为附 |

## 实现进度

| 模块 | 文件 | 状态 | 完成时间 | 备注 |
|------|------|------|---------|------|
| 环境修复 | mkl/gseapy | ✅ Done | 2026-06-07 | mkl 2025→2024.0.0；gseapy 1.2.1；torch+cuda 可用 |
| 路径修复 | scripts/fix_image_paths.py | ✅ Done | 2026-06-07 | C1 106980/106980 路径命中；C2/P1 待同样处理 |
| eval 修复 | rectified/rectified_evaluate.py | ✅ Done | 2026-06-07 | deprecation import 改 try/except + 守卫；加 pathway 重建分支 |
| 基线烟测 | — | ✅ Done | 2026-06-07 | 原 single 模型 forward+backward 通过，51.5M 参数，输出[B,4,256,256] |
| 通路掩码 | scripts/build_pathway_mask.py | ✅ Done | 2026-06-07 | C1/C2/P1 real/rand/none + W_ssgsea + P1 hallmark_reactome；全部已生成验证 |
| 通路编码器 | src/pathway_encoder.py | ✅ Done | 2026-06-07 | A→B→C，single/multi 入口；6变体 forward+backward 通过 |
| 模型集成 | src/single_model.py, multi_model.py | ✅ Done | 2026-06-07 | encoder_type 分支 + 原编码器补 l1_penalty() |
| CLI/入口 | src/utils.py, rectified_main.py | ✅ Done | 2026-06-07 | 通路参数组 + 载掩码透传 + 列数校验 + model_config 入 checkpoint |
| 训练 L1 | rectified/rectified_train.py | ✅ Done | 2026-06-07 | compute_l1_penalty 封装，4处统一；l1_weight/model_config 形参 |
| 评估修复 | rectified/rectified_evaluate.py | ✅ Done | 2026-06-07 | UNI2-h/sequoia/embeddings 全部优雅降级；pathway 重建；基础指标跑通 |
| 实验脚本 | scripts/run_*.sh + build_cross_masks.py | ✅ Done | 2026-06-07 | 6变体×3数据集×3种子 + 跨数据集(通路名对齐掩码) + summarize_results |
| 可解释性 | analysis/pathway_interpret.py | ✅ Done | 2026-06-07 | RQ4 三子分析 A/B/C 端到端跑通 |
| notebook | notebooks/01_*.ipynb | ✅ Done | 2026-06-07 | 掩码/token/注意力/生成 可视化；JSON 校验通过 |
| 文档 | README.md, .gitignore, requirements | ✅ Done | 2026-06-07 | 根目录运行说明；忽略 results/logs 保留 masks；新依赖记录 |

状态：⬜ TODO / 🔄 WIP / ✅ Done（已运行验证）/ ❌ Blocked

## 开发日志

### 2026-06-07 — 环境修复
- **完成内容**：诊断并修复 `zw@Gene2Image` 环境。根因 = mkl 2025.0.0 与 torch 2.2.2 不兼容（`undefined symbol: iJIT_NotifyEvent`）。`pip install mkl==2024.0.0` 后 `import torch` + `torch.cuda.is_available()=True` 通过。安装 gseapy 1.2.1（通路掩码依赖，原 requirements 缺）。
- **遇到的问题**：conda main channel 无 mkl<2025；anndata/scanpy/numpy 本就可用。
- **解决方案**：改用 pip 安装 mkl 2024.0.0（连带 intel-openmp 2024.2.2、tbb）。
- **数据现状**：C1/C2/P1 adata.h5ad 已在本地（2.4/4.7/5.2 GB），cell_patch_256_aux 图像齐全；但 cell_image_paths.json 内为原作者集群绝对路径，需重映射。

### 2026-06-07 — 路径/eval 修复 + 基线烟测
- **完成内容**：(1) `scripts/fix_image_paths.py` 重映射 C1 cell_image_paths，106980/106980 全命中本地 .tif。(2) 修复 `rectified_evaluate.py`：损坏的 `*_deprecation` import 改 try/except + 两处 append 守卫；新增 pathway 编码器重建分支（从 checkpoint config 还原掩码）。(3) 原始 single 模型 forward+backward 烟测通过（51.5M 参数，输出 [B,4,256,256]，L1 属性路径完好）。
- **遇到的问题**：GPU 实为 4× **V100 32GB**，非 idea_report 估算的 H100 80GB（峰值 78GB）。
- **解决方案**：真实训练需减小 batch_size（如 single 4→8，按显存调）/ 开 --use_amp；完整 100ep 训练不在本地代跑（按用户指示）。
- **未代跑**：原始 GeneFlow 完整训练（~12h），仅做模型级烟测确认代码路径无误。

### 2026-06-07 — 通路编码器 + 全链路集成
- **完成内容**：
  1. `scripts/build_pathway_mask.py`：构造通路掩码 + 三变体 + ssGSEA 权重。已对 C1(G=282,P=33)/C2(G=382,P=40)/P1(G=5006,P=50) 生成 real/rand/none；P1 额外 hallmark_reactome(P=1666)。校验列数=gene_dim、rand 保留每行计数、none 全1、W_ssgsea 行和=1。
  2. `src/pathway_encoder.py`：PathwayMaskEmbedding(edge-list scatter_add)→PathwayTransformer(自定义层可导出CLS注意力)→PathwaySingleEncoder/PathwayMultiEncoder。
  3. 集成到 single_model/multi_model(encoder_type 分支)、utils.py(CLI 参数组)、rectified_main.py(载掩码+列数校验+model_config)、rectified_train.py(compute_l1_penalty 解耦)。
  4. **集成测试通过**：6变体(Gene2Image/randPath/PathPrior/noTrans/noMask/GeneFlow)×(single+multi) forward+backward 全过；输出统一 [B,4,256,256]；PathPrior W 冻结确认；CLS→通路注意力 [B,P] 行和≈1。Gene2Image single 42.3M 可训练参数(< 基线 51.5M)。
- **遇到的问题**：torchmetrics 未安装(原 requirements 注释掉)→训练循环 import 失败。
- **解决方案**：`pip install torchmetrics==1.7.1`。
- **注意**：P1+Reactome 实际 P=1666(idea_report 估 P≈600)，通路编码器会更大；附加消融时留意显存。

### 2026-06-07 — 评估修复 + 实验脚本 + 可解释性 + 文档
- **完成内容**：
  1. **rectified_evaluate.py 全链路修复**：原脚本本地无法运行。逐一修复 import 与可选依赖：`*_deprecation`(try/except)、UNI2-h(load 返回 None 而非 raise + 4 处计算块守卫)、sequoia/HE2RNA(utils_he2rna 惰性 import)、embeddings 保存(空数组守卫)。结果：基础 FID/SSIM/PSNR 完整跑通，UNI2-h/round-trip 缺权重时自动跳过(N/A)。pathway checkpoint 自动重建编码器。
  2. **实验脚本**：run_experiments.sh(6变体×3数据集×3种子)、build_cross_masks.py(通路名对齐的跨数据集掩码)、run_cross_dataset.sh(c1→c2/c2→c1/c1→p1，训练源+评估目标+源参考)、summarize_results.py(多种子均值±std → summary_main/ablation csv)。
  3. **analysis/pathway_interpret.py**：RQ4 三子分析(A 注意力熵+主导通路 / B 与参照重合 / C 通路干预形态偏移)，端到端跑通产出 csv/json。
  4. **文档**：根 README.md(环境/数据/变体/运行全流程)、.gitignore(忽略 results/logs，保留 pathway_masks)、requirements.txt(记录新依赖)、notebooks/01 可视化。
- **端到端验证(真实 C1 数据)**：train(debug,1ep) → checkpoint(weights_only 可载) → 生成+gene importance → eval(FID/SSIM/PSNR,UNI2-h降级) → interpret(A/B/C) → summarize。全链路打通(指标因 toy 训练无意义，仅验证管线)。
  - 修复 torch 2.2.2 AMP 兼容(GradScaler)、checkpoint 存 tensor 而非 ndarray(weights_only 兼容)。
- **额外安装依赖**：torchmetrics, scikit-image, timm, einops, safetensors, opencv-python-headless==4.10.0.84；**numpy 锁回 1.26.4**(scikit-image 曾拉到 2.4 破坏 numba)。
- **遇到的问题**：评估脚本依赖链很长且多为原作者集群环境遗留(失效绝对路径/缺包/硬 raise)；numpy 版本冲突。
- **解决方案**：可选依赖一律惰性化+优雅降级；numpy 显式锁版本 + opencv 降级到 numpy<2 兼容版。

## 已知问题
- [ ] 真实多种子完整训练(100ep)未代跑(按用户指示，~12h/run)；脚本就绪可手动启动
- [ ] UNI2-h FID / RNA round-trip 需 gated 权重，当前降级为 N/A；需要时申请权重并放对路径
- [ ] ssGSEA 权重为通路内等权冻结(低置信度)，如需更贴近 MUPAD 可用表达统计派生
- [ ] P1+Reactome P=1666 远大于估算 600，附加消融显存需留意
- [ ] 真实训练显存：V100 32GB < 估算 78GB，需调小 batch / 开 AMP
- [ ] C2/P1 的 cell_image_paths 仍需各跑一次 fix_image_paths.py（训练前）
- [x] cell_image_paths.json 路径失效（`/depot/natallah/...`）→ 已修复脚本，C1 已处理
- [x] rectified_evaluate.py 导入不存在的 `*_deprecation` → 已 try/except 修复
- [ ] rectified_evaluate.py 导入不存在的 `*_deprecation` 模块，eval 会崩
- [ ] rectified_train.py L1 正则硬编码 `rna_encoder.encoder[0].weight`，通路编码器会 AttributeError
- [ ] ssGSEA 固定权重派生方式为低置信度（先用通路内等权冻结版）
