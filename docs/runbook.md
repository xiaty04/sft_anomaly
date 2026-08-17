# TSAD v2 远程实验 Runbook

本文档面向 AutoDL / 同类单卡 GPU 云主机，描述如何把本仓库从本地代码状态推进到可复现的远程实验。

> 总原则：
> - 本仓库不携带真实 UCR 数据、模型权重和训练产物。
> - UCR 数据需要用户自行放入 `data/raw/ucr/`；模型由 Transformers 在首次运行时自动下载。
> - 所有正式实验在单张 GPU 上完成，通过 SSH 命令行执行。
> - 训练数据只允许使用本地合成数据；UCR 只用于评估，不进入 SFT/RL 训练或 checkpoint 选择。

---

## 1. 环境要求

- 操作系统：Linux（AutoDL 镜像等）
- GPU：单张 NVIDIA 显卡，建议显存 24 GB 以上跑 3B/8B 视觉模型
- Python：推荐 3.10 或 3.11
- 磁盘：除代码外，需要存放 UCR 原始数据、合成数据、checkpoints、输出结果

本地只运行 CPU 级数据/协议/指标验证；不要在本地加载大模型。

---

## 2. 上传仓库与初始化

在本地完成代码验证后，将仓库上传到 GPU 实例（git clone / scp / rsync 均可）。

```bash
cd /path/to/sft_anomaly

# 创建虚拟环境并安装训练依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[train]'
```

检查命令是否可用：

```bash
tsad-v2 --help
# 或
python -m tsad_v2 --help
```

预期输出包含以下子命令：

```text
generate-synthetic prepare-ucr validate-manifest infer evaluate compare train-sft train-rl
```

---

## 3. 数据准备

### 3.1 生成合成训练数据

```bash
# 完整规模（1600 train + 400 val）
tsad-v2 --config configs/base.yaml generate-synthetic

# 小规模 smoke（16 train + 8 val）
tsad-v2 --config configs/base.yaml --config configs/smoke.yaml generate-synthetic
```

生成产物：

```text
data/processed/synthetic/train.jsonl
data/processed/synthetic/val.jsonl
data/processed/synthetic/images/{train,val}/*.png
data/processed/synthetic/series/{train,val}/*.npy
```

校验 manifest：

```bash
tsad-v2 --config configs/base.yaml validate-manifest data/processed/synthetic/train.jsonl
tsad-v2 --config configs/base.yaml validate-manifest data/processed/synthetic/val.jsonl
```

### 3.2 准备 UCR 测试集

将 UCR 原始 `.txt` 文件放入：

```text
data/raw/ucr/
```

文件名需形如：

```text
001_UCR_Anomaly_DISTORTED_demo_name_5_10_12.txt
```

含义：

- `001`：archive id
- `DISTORTED_demo_name`：数据集名称
- `5`：train end（训练边界，通常是计数）
- `10`：anomaly start
- `12`：anomaly end（按配置决定是否 inclusive）

执行转换与窗口渲染：

```bash
tsad-v2 --config configs/base.yaml prepare-ucr
```

生成：

```text
data/processed/ucr/series.jsonl
data/processed/ucr/windows.jsonl
data/processed/ucr/images/*.png
```

`series.jsonl` 是整条序列的评估单元；`windows.jsonl` 是输入给 VLM 推理的窗口单元。

> 如果实际 UCR 文件索引基准与配置不符，请先修改 `configs/base.yaml` 中：
> - `filename_index_base`
> - `anomaly_end_inclusive`
> - `train_end_is_count`
>
> 并在小样本上核对 `series.jsonl` 中的 `intervals` 是否符合真实标签。

---

## 4. 远程 GPU smoke 测试

先跑最小链路，确认模型加载、显存、训练和推理均无 OOM / nan。

### 4.1 小规模合成数据

```bash
# 生成小规模合成数据
tsad-v2 --config configs/base.yaml --config configs/smoke.yaml generate-synthetic
```

### 4.2 SFT smoke

```bash
tsad-v2 --config configs/base.yaml --config configs/smoke.yaml train-sft
```

或使用完整数据但限制步数：

```bash
tsad-v2 --config configs/base.yaml train-sft --limit 16
```

检查点：

- 无 OOM / CUDA error；
- loss 非 nan，且处于合理范围；
- 输出目录出现 `outputs/smoke/sft/final_adapter` 或 `outputs/sft/final_adapter`。

### 4.3 RL smoke

在 SFT smoke 之后：

```bash
tsad-v2 --config configs/base.yaml --config configs/smoke.yaml train-rl \
  --sft-adapter outputs/smoke/sft/final_adapter

# 或完整数据 + limit
tsad-v2 --config configs/base.yaml train-rl \
  --sft-adapter outputs/sft/final_adapter --limit 16
```

检查点：

- reward 曲线可读；
- 无 OOM / nan；
- 输出目录出现 `final_adapter`。

### 4.4 baseline 推理 smoke

```bash
tsad-v2 --config configs/base.yaml --config configs/smoke.yaml prepare-ucr

tsad-v2 --config configs/base.yaml --config configs/smoke.yaml infer \
  data/processed/ucr_smoke/windows.jsonl \
  outputs/smoke/baseline/predictions.jsonl \
  --limit 4

tsad-v2 --config configs/base.yaml --config configs/smoke.yaml evaluate \
  data/processed/ucr_smoke/series.jsonl \
  outputs/smoke/baseline/predictions.jsonl \
  outputs/smoke/baseline
```

---

## 5. 正式实验流程

### 5.1 数据与 baseline

```bash
# 合成数据（正式规模）
tsad-v2 --config configs/base.yaml generate-synthetic

# UCR 数据（用户已放入 data/raw/ucr/）
tsad-v2 --config configs/base.yaml prepare-ucr

# 零样本 baseline
tsad-v2 --config configs/base.yaml infer \
  data/processed/ucr/windows.jsonl outputs/baseline/predictions.jsonl

tsad-v2 --config configs/base.yaml evaluate \
  data/processed/ucr/series.jsonl outputs/baseline/predictions.jsonl outputs/baseline
```

### 5.2 SFT

```bash
# GPU smoke（可选）
tsad-v2 --config configs/base.yaml train-sft --limit 16

# 正式 SFT
tsad-v2 --config configs/base.yaml train-sft

# 推理与评估
tsad-v2 --config configs/base.yaml infer \
  data/processed/ucr/windows.jsonl outputs/sft_ucr/predictions.jsonl \
  --adapter outputs/sft/final_adapter

tsad-v2 --config configs/base.yaml evaluate \
  data/processed/ucr/series.jsonl outputs/sft_ucr/predictions.jsonl outputs/sft_ucr
```

### 5.3 RL

```bash
# GPU smoke（可选）
tsad-v2 --config configs/base.yaml train-rl \
  --sft-adapter outputs/sft/final_adapter --limit 16

# 正式 RL
tsad-v2 --config configs/base.yaml train-rl \
  --sft-adapter outputs/sft/final_adapter

# 推理与评估
tsad-v2 --config configs/base.yaml infer \
  data/processed/ucr/windows.jsonl outputs/rl_ucr/predictions.jsonl \
  --adapter outputs/rl/final_adapter

tsad-v2 --config configs/base.yaml evaluate \
  data/processed/ucr/series.jsonl outputs/rl_ucr/predictions.jsonl outputs/rl_ucr
```

### 5.4 三组结果汇总

```bash
tsad-v2 compare \
  outputs/baseline/summary.json \
  outputs/sft_ucr/summary.json \
  outputs/rl_ucr/summary.json \
  --output outputs/comparison.csv
```

---

## 6. 输出目录约定

```text
outputs/
├── baseline/
│   ├── predictions.jsonl
│   ├── per_sample.jsonl
│   ├── summary.json
│   └── effective_config.yaml
├── sft/
│   ├── checkpoints/
│   └── final_adapter/
├── sft_ucr/
│   ├── predictions.jsonl
│   └── summary.json
├── rl/
│   ├── checkpoints/
│   └── final_adapter/
└── rl_ucr/
    └── summary.json
```

`final_adapter` 同时保存 PEFT adapter 和 processor。

---

## 7. 断点恢复

- `infer` 会读取已有 `predictions.jsonl`，按 `sample_id` 跳过已完成样本，可重复执行继续。
- `train-sft` 支持 `--resume <checkpoint_dir>`。
- `train-rl` 支持 `--resume <checkpoint_dir>`。
- 任何阶段中断后，先检查 `outputs/` 下是否已有可复用产物，再决定是否重跑。

---

## 8. 验收门槛

每个阶段建议至少确认：

| 阶段 | 门槛 |
|---|---|
| 数据 | train/val manifest 非空；UCR windows/series 图像存在；标签为半开区间 |
| Baseline | `parse_rate` 合理；有 `summary.json` |
| SFT smoke | loss 非 nan，5 步内可完成 |
| SFT 正式 | 合成验证集 loss 下降；UCR 上相对 baseline 有提升或可解释变化 |
| RL smoke | reward 非 nan，可观察到 reward 上升 |
| RL 正式 | UCR 上相对 SFT 的指标变化可解释、可复现 |
| 对比 | `comparison.csv` 三组齐全 |

---

## 9. 常见问题

### 9.1 `data/raw/ucr` 为空

```text
FileNotFoundError: no UCR .txt files found in data/raw/ucr
```

把 UCR 原始文件放入该目录后重新执行 `prepare-ucr`。

### 9.2 显存不足

- 降低 `configs/base.yaml` 中 `render.width/height` 或 `model.max_pixels`；
- 使用 `--limit` 做 smoke；
- 确保已启用 4bit / gradient checkpointing。

### 9.3 TRL / Transformers 版本不兼容

- 安装时使用 `pip install -e '.[train]'`；
- 若 GRPO API 报错，先核对 `pyproject.toml` 中 `train` extras 的依赖下界，并在 GPU 实例上用 `pip freeze` 锁定一套可用版本后重试；
- 不要直接修改训练主逻辑，优先调整版本。

### 9.4 UCR 标签转换不符

- 检查 `series.jsonl` 中的 `intervals`；
- 根据实际 UCR 文件名调整 `filename_index_base`、`anomaly_end_inclusive`、`train_end_is_count`。

### 9.5 本地不要跑大模型

本地 CPU 环境只执行 README 中“本地 CPU 验证”列出的数据、协议和指标检查：

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff check src tests
```

不要在本机执行 `infer`、`train-sft`、`train-rl` 的正式命令。
