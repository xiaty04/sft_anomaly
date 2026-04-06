# Notebook 使用指南

本项目包含三个 notebook：

| Notebook | 内容 | 阶段 |
|----------|------|------|
| `tsad_sft_pipeline.ipynb` | 环境初始化 → Baseline 评估 → 检查点 B | 0 – 4 |
| `tsad_sft_pipeline_part2.ipynb` | 环境恢复（检查点 B tar 包）→ 教师蒸馏 / 训练数据准备 → `before_stage8` 打包 | 5 – 7 |
| `tsad_sft_pipeline_part3.ipynb` | 环境恢复（阶段 7 tar 包）→ SFT 训练 / 导出 / 评估 | 8 – 10 |

---

## 阶段总览

| 阶段 | Cell 编号 | 功能 | 产物 |
|------|-----------|------|------|
| 0 | 0.1 – 0.3 | 初始化环境（挂载 Drive、创建目录、安装依赖） | 运行目录结构 |
| 1 | 1.1 – 1.4 | 获取 AnomLLM 代码和数据 | 仓库 + data.pkl |
| **⏸ A** | 验证 cell | **检查点：确认 GPU / 数据就绪** | — |
| 2 | 2.1 – 2.3 | VLM Baseline（本地 vLLM 推理） | `0shot-vision.jsonl` |
| 3 | 3.1 | 统计 Baseline（Isolation Forest） | `0shot.jsonl` |
| 4 | 4.1 – 4.2 | 汇总 Baseline 对比 | `baseline_compare.csv` |
| **⏸ B** | 验证 cell | **检查点：确认 Baseline 指标** | — |
| **📓 Part 2** | — | **切换到 `tsad_sft_pipeline_part2.ipynb`** | — |
| 0-B | 0-B.1 – 0-B.2 | 恢复环境（安装依赖、解压检查点 B tar 包） | 运行目录结构 |
| 5 | 5.1 | 构建 SFT 样本清单 | `sft_manifest.csv` |
| 6 | 6.1 – 6.2b – 6.3 | 教师蒸馏（冒烟→全量）+ 自动清洗 | `sft_final.jsonl` |
| 7 | 7.1 – 7.3 | 转 Unsloth 训练格式 + 打包阶段 7 交接产物 | `train.jsonl` / `val.jsonl` / `eval.jsonl` / `before_stage8_*.tar.gz` |
| **📓 Part 3** | — | **切换到 `tsad_sft_pipeline_part3.ipynb`** | — |
| 0-C | 0-C.1 – 0-C.2 | 恢复环境（安装依赖、解压阶段 7 tar 包） | 运行目录结构 |
| 8 | 8.1 – 8.2b – 8.3 | SFT 训练（冒烟→完整）+ 导出模型 | LoRA adapter + merged model |
| **⏸ C** | 验证 cell | **检查点：确认训练完成** | — |
| 9 | 9.1 – 9.2 | 对比 SFT 与 Baseline | `sft_eval_metrics.csv` |
| **⏸ D** | 验证 cell | **检查点：查看最终对比** | — |
| 10 | — | 可选 GRPO 强化学习 | — |

---

## 各阶段说明

### 阶段 0：初始化环境

| Cell | 功能 |
|------|------|
| **0.1** | 挂载 Google Drive，用于持久化存储数据和模型 |
| **0.2** | 定义运行目录常量（`RUNTIME`、`DRIVE_ROOT` 等），创建所有必要目录 |
| **0.3** | 安装 Python 依赖：unsloth、vllm、openai、accelerate、bitsandbytes、scikit-learn，以及 AnomLLM 依赖的 `loguru` / `google-generativeai` / `affiliation` |

### 阶段 1：获取数据

| Cell | 功能 |
|------|------|
| **1.1** | 克隆 AnomLLM 仓库，设置 PYTHONPATH |
| **1.2** | 优先从 Drive 恢复合成数据；若无缓存则从 S3 下载，首次下载后自动打包回 Drive |
| **1.3** | 遍历 4 个子集（point / range / freq / flat-trend），打印 series 数量、shape、是否有图像 |
| **1.4** | 固定本次实验使用的数据子集列表 |

### 检查点 A

打印 GPU 型号 / 可用 VRAM / BF16 支持，目视一张时间序列 PNG。确认环境正确后继续。

### 阶段 2：VLM Baseline

| Cell | 功能 |
|------|------|
| **2.1** | 用 `nohup` 后台启动 vLLM 服务，加载 `Qwen/Qwen3-VL-8B-Instruct`，端口 8000 |
| **2.2** | 写 `credentials.yml`（本地 API 地址），供 AnomLLM 脚本读取 |
| **2.3** | 用 `online_api.py` 对 4 个子集各 400 条跑 VLM zero-shot 视觉推理 |

### 阶段 3：统计 Baseline

| Cell | 功能 |
|------|------|
| **3.1** | 用 `isoforest.py` 对 4 个子集跑 Isolation Forest baseline |

### 阶段 4：汇总对比

| Cell | 功能 |
|------|------|
| **4.1** | 用 `result_agg.py` 对每个子集计算 precision / recall / F1 / affi-F1 |
| **4.2** | 合并为 `baseline_compare.csv`，打印全表 |

### 检查点 B

用 `wc -l` 确认各 jsonl 均 400 行，打印 `baseline_compare.csv` 全文。当前 notebook 会在检查点 B 前把中途产物打包为 `before_ckptB_*.tar.gz` 并写入 Drive。

---

> **📓 以下阶段在 `tsad_sft_pipeline_part2.ipynb` 中运行**

### 阶段 0-B：恢复环境（Part 2 前导）

| Cell | 功能 |
|------|------|
| **0-B.1** | 挂载 Google Drive，定义运行目录常量，安装 Python 依赖 |
| **0-B.2** | 克隆 AnomLLM 仓库，解压 Drive 中最新的 `before_ckptB_*.tar.gz`，恢复 Part 1 打包产物，并验证 `baseline_compare.csv`、4 个 `data.pkl`、各子集 baseline JSONL / agg pkl 可加载 |

### 阶段 5：构建 SFT 清单

| Cell | 功能 |
|------|------|
| **5.1** | 从 4 个子集的 eval data.pkl 提取 1600 条样本，分层切分为 train(1280) / val(160) / eval(160)，输出 `sft_manifest.csv` |

### 阶段 6：教师蒸馏

| Cell | 功能 |
|------|------|
| **6.1** | 设置 DashScope API Key 和教师模型（qwen3.5-plus） |
| **6.2a** | 冒烟：对前 20 条调用教师模型，输出 `sft_raw_smoke.jsonl`；打印 `is_anomaly` 一致率和 intervals 格式合法率，≥ 70% 才继续 |
| **6.2b** | 全量蒸馏：对全部 train+val 样本调用教师模型，输出 `sft_raw.jsonl` |
| **6.3** | 自动清洗：保留 label 一致、`is_anomaly` 与 `intervals` 一致、且 intervals 结构合法的样本，输出 `sft_final.jsonl` |

### 阶段 7：转训练格式

| Cell | 功能 |
|------|------|
| **7.1** | 将 `sft_final.jsonl` 转为 Unsloth messages 格式（text + image），assistant 输出为 JSON 区间列表，输出 `train.jsonl` / `val.jsonl` |
| **7.2** | eval split 单独转为评估元数据，输出 `eval.jsonl` |
| **7.3** | 将阶段 5-7 的 SFT 产物以及 Part 3 评估仍需用到的 baseline/data 文件打包到 Drive，输出 `before_stage8_*.tar.gz` |

### 切换到 Part 3

确认 Drive 中已生成最新的 `before_stage8_*.tar.gz` 后，切换到 `tsad_sft_pipeline_part3.ipynb`。

---

> **📓 以下阶段在 `tsad_sft_pipeline_part3.ipynb` 中运行**

### 阶段 0-C：恢复环境（Part 3 前导）

| Cell | 功能 |
|------|------|
| **0-C.1** | 挂载 Google Drive，定义运行目录常量，安装 Python 依赖 |
| **0-C.2** | 克隆 AnomLLM 仓库，解压 Drive 中最新的 `before_stage8_*.tar.gz`，恢复 Part 2 打包产物，并校验 `sft_manifest.csv`、`train/val/eval.jsonl`、`sft_final.jsonl` 以及 baseline/data 文件可加载 |

### 阶段 8：SFT 训练

| Cell | 功能 |
|------|------|
| **8.1** | 加载 `Qwen3-VL-8B-Instruct` 4bit 量化模型，添加 LoRA（r=16, all-linear），加载训练/验证数据 |
| **8.2a** | 冒烟训练（`max_steps=5`，`lr=2e-5`，`batch=1`）：验证流程无 OOM、loss 非 nan；检查 loss 曲线在合理范围（1.5~4）后继续 |
| **8.2b** | 完整训练：先释放冒烟模型、清理显存，再重新加载模型并以完整配置（无 `max_steps`，`lr=1e-4`，`batch=2`）训练 3 个 epoch |
| **8.3** | 导出 merged model 和 LoRA adapter 到 sft 目录 |

### 检查点 C

打印蒸馏样本前 5 条（来自 `sft_raw.jsonl`）、训练格式验证（image exists / assistant JSON 格式）。检查 loss 曲线正常收敛后再运行 Cell 8.3 导出。

### 阶段 9：最终对比

| Cell | 功能 |
|------|------|
| **9.1** | 用 `nohup` 后台启动 SFT 模型的 vLLM server（端口 8001），更新 `credentials.yml` 中的 `sft-model`，对 4 个子集跑推理 |
| **9.2** | 在相同的 eval 160 条上计算 isolation-forest / qwen-local-0shot / sft-0shot 三种方法的 F1 对比 |

### 检查点 D

查看 summary 表，判断 SFT 是否有提升（F1 高 ≥ 0.03 为可见提升），决定是否进入 GRPO。

---

## 需要修改的参数

### 必须修改

| 参数 | 位置 | 说明 |
|------|------|------|
| `DASHSCOPE_API_KEY` | Cell 6.1 | 替换 `"sk-xxxx"` 为你的百炼 API Key |

### 根据环境调整

| 参数 | 位置 | 默认值 | 何时修改 |
|------|------|--------|----------|
| `per_device_train_batch_size` | Cell 8.2b | `2` | OOM → 改为 `1` |
| `learning_rate` | Cell 8.2b | `1e-4` | loss=nan → 降至 `2e-5` |
| `num_train_epochs` | Cell 8.2b | `3` | 可根据 loss 曲线增减 |
| `gradient_accumulation_steps` | Cell 8.2b | `8` | 降 batch_size 时相应增大 |
| `TEACHER_MODEL` | Cell 6.1 | `"qwen3.5-plus"` | 可换其他 DashScope VLM |
| `TEACHER_TEMPERATURE` | Cell 6.1 | `0.2` | 一般不需改 |
| `DATASETS` | Cell 1.4 | 4 个子集 | 如只跑部分子集可修改 |

### 可选调整

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `r` (LoRA rank) | Cell 8.1 | `16` | 增大提高表达力但占更多显存 |
| `lora_alpha` | Cell 8.1 | `16` | 通常与 `r` 保持一致 |
| `test_size` | Cell 5.1 | `0.20` | train/eval 切分比例 |
| `random_state` | Cell 5.1 | `3407` | 随机种子 |

---

## 运行方法

### 首次运行 — Part 1（`tsad_sft_pipeline.ipynb`）

1. 在 Colab 中打开 notebook，选择 **A100 GPU** 运行时
2. **按顺序执行** Cell 0.1 → 0.2 → 0.3 → 1.1 → 1.2 → 1.3 → 1.4
3. 在 **检查点 A** 暂停，确认 GPU/数据/BF16 正常
4. 继续执行 Cell 2.1 → 2.2 → 2.3 → 3.1 → 4.1 → 4.2
5. 在 **检查点 B** 暂停，确认 baseline jsonl 各 400 行、F1 非零，并确认 Drive 中已生成最新的 `before_ckptB_*.tar.gz`

### 首次运行 — Part 2（`tsad_sft_pipeline_part2.ipynb`）

1. 打开 `tsad_sft_pipeline_part2.ipynb`，选择 **A100 GPU** 运行时
2. 运行 Cell 0-B.1（挂载 Drive + 路径 + 依赖）和 Cell 0-B.2（克隆仓库 + 解压最新 `before_ckptB_*.tar.gz` + 校验恢复结果）
3. 确认 4 个 `data.pkl`、`baseline_compare.csv`、各子集 baseline JSONL / agg pkl 加载正确
4. 继续执行 Cell 5.1 → 6.1（**填入 API Key**）→ 6.2a（冒烟 20 条，确认质量 ≥ 70%）→ 6.2b（全量蒸馏）→ 6.3 → 7.1 → 7.2
5. 运行 Cell 7.3，把阶段 5-7 产物打包到 Drive
6. 确认 Drive 中已生成最新的 `before_stage8_*.tar.gz`

### 首次运行 — Part 3（`tsad_sft_pipeline_part3.ipynb`）

1. 打开 `tsad_sft_pipeline_part3.ipynb`，选择 **A100 GPU** 运行时
2. 运行 Cell 0-C.1（挂载 Drive + 路径 + 依赖）和 Cell 0-C.2（克隆仓库 + 解压最新 `before_stage8_*.tar.gz` + 校验恢复结果）
3. 确认 `sft_manifest.csv` 的 split 分布为 `train=1280 / val=160 / eval=160`，并且 `train.jsonl` / `val.jsonl` / `eval.jsonl` / `sft_final.jsonl` 均存在且非空
4. 执行 Cell 8.1 → 8.2a（冒烟 5 步，确认 loss 在 1.5~4、无 OOM）
5. 确认冒烟通过后执行 Cell 8.2b 跑完整训练
6. 在 **检查点 C** 暂停，确认训练正常收敛，然后运行 Cell 8.3 导出
7. 执行 Cell 9.1 → 9.2，在 **检查点 D** 查看对比结果

### 断线恢复

- **Part 1**：每次新 session 从 Cell 0.1 开始重跑（重新挂载 Drive、安装依赖）。数据从 S3 重新下载。
- **Part 2**：每次新 session 运行 Cell 0-B.1 和 0-B.2 即可恢复到蒸馏前状态（依赖 + Drive 中最新的检查点 B tar 包）。
- **Part 3**：每次新 session 运行 Cell 0-C.1 和 0-C.2 即可恢复到训练前状态（依赖 + Drive 中最新的 `before_stage8_*.tar.gz`）。

### 关键检查项

| 检查点 | 期望结果 |
|--------|----------|
| A | GPU A100 / BF16 True / 四子集各 400 条 / has_figs True |
| B | 各 jsonl 400 行 / VLM F1 在 0.2~0.7 |
| C | 蒸馏保留率 ≥ 70% / 完整训练 loss 正常收敛 / 格式验证通过 |
| D | SFT F1 > VLM zero-shot F1（差 ≥ 0.03 为有效提升） |

补充说明：`Cell 2.1` / `9.1` 现在使用 `nohup ... > /tmp/*.log 2>&1 &` 后台启动，cell 会很快结束；若后续推理失败，优先查看 `/tmp/vllm.log` 或 `/tmp/sft-vllm.log`。

---

## 运行时目录结构

```
/content/tsad_runtime/
├── code/AnomLLM/              # AnomLLM 仓库
│   ├── data/synthetic/<subset>/eval/
│   │   ├── data.pkl           # 时间序列数据 + 标签
│   │   └── figs/001.png ...   # 预渲染折线图
│   ├── results/synthetic/     # baseline 推理结果
│   └── credentials.yml        # vLLM API 配置
├── sft/
│   ├── sft_manifest.csv       # 样本清单（含 split）
│   ├── sft_raw.jsonl          # 教师蒸馏原始输出
│   ├── sft_final.jsonl        # 清洗后的蒸馏数据
│   ├── train.jsonl            # Unsloth 训练格式（text + image，assistant 为 JSON 区间列表）
│   ├── val.jsonl              # 验证集
│   ├── eval.jsonl             # 评估元数据
│   ├── qwen3vl-tsad-merged/   # 导出的完整模型
│   └── qwen3vl-tsad-adapter/  # LoRA adapter
├── checkpoints/               # 训练 checkpoint
└── results/
    ├── baseline_compare.csv   # Baseline 对比表
    └── sft_eval_metrics.csv   # SFT vs Baseline 最终对比
```
