# TSAD SFT Pipeline — Notebook 使用指南

基于 [AnomLLM](https://github.com/Rose-STL-Lab/AnomLLM) 框架，使用 Qwen3-VL + Unsloth 对时间序列异常检测进行视觉 SFT 微调的完整流水线。

**运行环境**：Google Colab（需要 A100 GPU + 高 RAM 运行时）

---

## 快速总览

| 阶段 | Cell 编号 | 功能 | 产物 |
|------|-----------|------|------|
| 0 | 0.1 – 0.3 | 初始化环境（挂载 Drive、创建目录、安装依赖） | 运行目录结构 |
| 1 | 1.1 – 1.4 | 获取 AnomLLM 代码和数据 | 仓库 + data.pkl |
| **⏸ A** | 验证 cell | **检查点：确认 GPU / 数据就绪** | — |
| 2 | 2.1 – 2.3 | VLM Baseline（本地 vLLM 推理） | `0shot-vision.jsonl` |
| 3 | 3.1 | 统计 Baseline（Isolation Forest） | `0shot.jsonl` |
| 4 | 4.1 – 4.2 | 汇总 Baseline 对比 | `baseline_compare.csv` |
| **⏸ B** | 验证 cell | **检查点：确认 Baseline 指标** | — |
| 5 | 5.1 | 构建 SFT 样本清单 | `sft_manifest.csv` |
| 6 | 6.1 – 6.3 | 教师蒸馏 + 自动清洗 | `sft_final.jsonl` |
| 7 | 7.1 – 7.2 | 转 Unsloth 训练格式 | `train.jsonl` / `val.jsonl` / `eval.jsonl` |
| 8 | 8.1 – 8.3 | SFT 训练 + 导出模型 | LoRA adapter + merged model |
| **⏸ C** | 验证 cell | **检查点：确认训练冒烟通过** | — |
| 9 | 9.1 – 9.2 | 对比 SFT 与 Baseline | `sft_eval_metrics.csv` |
| **⏸ D** | 验证 cell | **检查点：查看最终对比** | — |
| 10 | — | 可选 GRPO 强化学习 | — |

---

## 各 Cell 功能说明

### 阶段 0：初始化环境

| Cell | 功能 |
|------|------|
| **0.1** | 挂载 Google Drive，用于持久化存储数据和模型 |
| **0.2** | 定义运行目录常量（`RUNTIME`、`DRIVE_ROOT` 等），创建所有必要目录 |
| **0.3** | 安装 Python 依赖：unsloth、vllm、openai、accelerate、bitsandbytes、scikit-learn 等 |

### 阶段 1：获取数据

| Cell | 功能 |
|------|------|
| **1.1** | 克隆 AnomLLM 仓库，设置 PYTHONPATH |
| **1.2** | 从 Drive 恢复数据（优先）或从 S3 下载合成数据，首次下载后自动打包备份到 Drive |
| **1.3** | 遍历 4 个子集（point / range / freq / flat-trend），打印 series 数量、shape、是否有图像 |
| **1.4** | 固定本次实验使用的数据子集列表 |

### 检查点 A（验证 cell）

打印 GPU 型号 / 可用 VRAM / BF16 支持，目视一张时间序列 PNG。确认环境正确后继续。

### 阶段 2：VLM Baseline

| Cell | 功能 |
|------|------|
| **2.1** | 后台启动 vLLM 服务，加载 `Qwen/Qwen3-VL-8B-Instruct`，端口 8000 |
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

### 检查点 B（验证 cell）

用 `wc -l` 确认各 jsonl 均 400 行，打印 `baseline_compare.csv` 全文。

### 阶段 5：构建 SFT 清单

| Cell | 功能 |
|------|------|
| **5.1** | 从 4 个子集的 eval data.pkl 提取 1600 条样本，分层切分为 train(1200) / val(150) / eval(150)，输出 `sft_manifest.csv` |

### 阶段 6：教师蒸馏

| Cell | 功能 |
|------|------|
| **6.1** | 设置 DashScope API Key 和教师模型（qwen3.5-plus） |
| **6.2** | 调用教师模型对 train+val 样本做标注，生成 `sft_raw.jsonl`。默认先跑 20 条冒烟 |
| **6.3** | 自动清洗：保留 label 一致 + intervals 格式合法的样本，输出 `sft_final.jsonl` |

### 阶段 7：转训练格式

| Cell | 功能 |
|------|------|
| **7.1** | 将 `sft_final.jsonl` 转为 Unsloth messages 格式（image + text），输出 `train.jsonl` / `val.jsonl` |
| **7.2** | eval split 单独转（只含 user turn，用于推理），输出 `eval.jsonl` |

### 阶段 8：SFT 训练

| Cell | 功能 |
|------|------|
| **8.1** | 加载 `Qwen3-VL-8B-Instruct` 4bit 量化模型，添加 LoRA（r=16, all-linear），加载训练/验证数据 |
| **8.2** | 启动 SFT 训练。默认 `max_steps=5` 冒烟模式，确认无 OOM 后注释掉该行跑完整训练 |
| **8.3** | 导出 merged model 和 LoRA adapter 到 sft 目录 |

### 检查点 C（验证 cell）

打印蒸馏样本前 5 条、清洗保留率、训练格式验证（image exists / label 格式）。

### 阶段 9：最终对比

| Cell | 功能 |
|------|------|
| **9.1** | 启动 SFT 模型的 vLLM server（端口 8001），对 4 个子集跑推理 |
| **9.2** | 在相同的 eval 150 条上计算 isolation-forest / qwen-local-0shot / sft-0shot 三种方法的 F1 对比 |

### 检查点 D

查看 summary 表，判断 SFT 是否有提升（F1 高 ≥ 0.03 为可见提升），决定是否进入 GRPO。

---

## 需要修改的参数

### 必须修改

| 参数 | 位置 | 说明 |
|------|------|------|
| `DASHSCOPE_API_KEY` | Cell 6.1 | 替换 `"sk-xxxx"` 为你的百炼 API Key，用于调用教师模型蒸馏 |

### 根据环境调整

| 参数 | 位置 | 默认值 | 何时修改 |
|------|------|--------|----------|
| `per_device_train_batch_size` | Cell 8.2 | `2` | VRAM 不足 OOM → 改为 `1` |
| `learning_rate` | Cell 8.2 | `1e-4` | loss 出现 nan → 降至 `2e-5` |
| `max_steps` | Cell 8.2 | `5`（冒烟） | 冒烟通过后**注释掉该行**跑完整训练 |
| `num_train_epochs` | Cell 8.2 | `3` | 可根据 loss 曲线增减 |
| `gradient_accumulation_steps` | Cell 8.2 | `8` | 降 batch_size 时可相应增大以保持等效 batch |
| `TEACHER_MODEL` | Cell 6.1 | `"qwen3.5-plus"` | 可换其他 DashScope 支持的 VLM |
| `TEACHER_TEMPERATURE` | Cell 6.1 | `0.2` | 蒸馏标注随机性，一般不需改 |
| `DATASETS` | Cell 1.4 | 4 个子集 | 如只想跑部分子集可修改此列表 |

### 可选调整

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `r` (LoRA rank) | Cell 8.1 | `16` | 增大可提高表达力但占更多显存 |
| `lora_alpha` | Cell 8.1 | `16` | 通常与 `r` 保持一致 |
| `test_size` | Cell 5.1 | `0.20` | train/eval 切分比例 |
| `random_state` | Cell 5.1 | `3407` | 随机种子，影响数据切分 |

---

## 运行方法

### 首次运行

1. 在 Colab 中打开 notebook，选择 **A100 GPU** 运行时
2. **按顺序执行** Cell 0.1 → 0.2 → 0.3 → 1.1 → 1.2 → 1.3 → 1.4
3. 在 **检查点 A** 暂停，确认 GPU/数据/BF16 正常
4. 继续执行 Cell 2.1 → 2.2 → 2.3 → 3.1 → 4.1 → 4.2
5. 在 **检查点 B** 暂停，确认 baseline jsonl 各 400 行、F1 非零
6. 继续执行 Cell 5.1 → 6.1（**填入 API Key**）→ 6.2（先跑 20 条冒烟）→ 6.3 → 7.1 → 7.2
7. 执行 Cell 8.1 → 8.2（`max_steps=5` 冒烟）
8. 在 **检查点 C** 暂停，确认 loss 在 1.5~4、无 OOM
9. **注释掉 `max_steps=5`**，重新运行 Cell 8.2 跑完整训练 → 运行 Cell 8.3 导出
10. 执行 Cell 9.1 → 9.2，在 **检查点 D** 查看对比结果

### 断线恢复

每次新 Colab session 需从 **Cell 0.1 开始重跑**（重新挂载 Drive、安装依赖）。数据会从 Drive 自动恢复（Cell 1.2）。

### 关键检查项

| 检查点 | 期望结果 |
|--------|----------|
| A | GPU A100 / BF16 True / 四子集各 400 条 / has_figs True |
| B | 各 jsonl 400 行 / VLM F1 在 0.2~0.7 |
| C | 蒸馏保留率 ≥ 70% / 冒烟 loss 1.5~4 / 无 OOM |
| D | SFT F1 > VLM zero-shot F1（差 ≥ 0.03 为有效提升） |

---

## 目录结构

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
│   ├── train.jsonl            # Unsloth 训练格式
│   ├── val.jsonl              # 验证集
│   ├── eval.jsonl             # 评估集
│   ├── qwen3vl-tsad-merged/   # 导出的完整模型
│   └── qwen3vl-tsad-adapter/  # LoRA adapter
├── checkpoints/               # 训练 checkpoint
└── results/
    ├── baseline_compare.csv   # Baseline 对比表
    └── sft_eval_metrics.csv   # SFT vs Baseline 最终对比
```
