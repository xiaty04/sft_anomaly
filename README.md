# TSAD v2

TSAD v2 是一个面向真实时序异常区间检测的单卡命令行工程。主线严格对应
[`v2_plan.md`](v2_plan.md)：

```text
AnomLLM 任务协议
→ UCR 本地开源 VLM zero-shot baseline
→ 新生成的精确标签合成数据 SFT
→ 区间级 GRPO
→ 同一套 UCR 协议统一评估
```

当前仓库完成的是本地代码、配置、测试、文档与 AutoDL 操作入口。没有下载真实数据或模型，
也没有伪造 GPU 实验结果。`v1/` 是旧版 Colab/教师蒸馏流程的历史归档，不是 v2 依赖。
真实 UCR 数据由用户在远程环境放入 `data/raw/ucr/`；模型由 Transformers 在首次运行时自动下载。

## 工程结构

```text
configs/                 # 单卡默认配置与小规模 smoke 配置
src/tsad_v2/
  data/                  # UCR 转换、窗口化与新合成数据生成
  training/              # QLoRA SFT 与 TRL GRPO
  intervals.py           # 统一半开区间协议和严格解析
  metrics.py             # point/event/IoU/boundary 指标
  rewards.py             # 区间级 RL reward
  inference.py           # 本地 VLM 推理，支持 adapter
  evaluation.py          # 窗口合并与统一评估
tests/                   # 不需要 GPU 的协议与数据测试
docs/
  runbook.md             # 远程 AutoDL 完整运行、恢复和验收规则
  protocol.md            # 协议、JSONL schema、指标与 reward 口径
```

## 本地 CPU 验证

本地只运行不需要 GPU 的数据、协议和指标验证：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff check src tests
python -m tsad_v2 --config configs/base.yaml --config configs/smoke.yaml generate-synthetic
python -m tsad_v2 --config configs/base.yaml --config configs/smoke.yaml \
  validate-manifest data/processed/synthetic_smoke/train.jsonl
python -m tsad_v2 --config configs/base.yaml --config configs/smoke.yaml \
  validate-manifest data/processed/synthetic_smoke/val.jsonl
```

这会在 gitignored 的 `data/` 下生成少量 smoke 数据，便于本地查看；不要在本机执行
`infer`、`train-sft`、`train-rl` 的正式命令。

## AutoDL 主线

在单卡 GPU 实例中安装训练依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[train]'
```

准备合成训练集和已由用户放入 `data/raw/ucr/` 的 UCR 文件：

```bash
tsad-v2 --config configs/base.yaml generate-synthetic
tsad-v2 --config configs/base.yaml prepare-ucr
```

按小规模到正式实验的顺序执行：

```bash
# 1. 零样本 baseline
tsad-v2 --config configs/base.yaml infer \
  data/processed/ucr/windows.jsonl outputs/baseline/predictions.jsonl
tsad-v2 --config configs/base.yaml evaluate \
  data/processed/ucr/series.jsonl outputs/baseline/predictions.jsonl outputs/baseline

# 2. 合成数据 SFT；先用 --limit 16 做 GPU smoke test
tsad-v2 --config configs/base.yaml train-sft --limit 16
tsad-v2 --config configs/base.yaml train-sft
tsad-v2 --config configs/base.yaml infer \
  data/processed/ucr/windows.jsonl outputs/sft_ucr/predictions.jsonl \
  --adapter outputs/sft/final_adapter
tsad-v2 --config configs/base.yaml evaluate \
  data/processed/ucr/series.jsonl outputs/sft_ucr/predictions.jsonl outputs/sft_ucr

# 3. 从 SFT adapter 继续区间级 GRPO
tsad-v2 --config configs/base.yaml train-rl \
  --sft-adapter outputs/sft/final_adapter --limit 16
tsad-v2 --config configs/base.yaml train-rl \
  --sft-adapter outputs/sft/final_adapter
tsad-v2 --config configs/base.yaml infer \
  data/processed/ucr/windows.jsonl outputs/rl_ucr/predictions.jsonl \
  --adapter outputs/rl/final_adapter
tsad-v2 --config configs/base.yaml evaluate \
  data/processed/ucr/series.jsonl outputs/rl_ucr/predictions.jsonl outputs/rl_ucr

# 4. 三组结果汇总
tsad-v2 compare \
  outputs/baseline/summary.json outputs/sft_ucr/summary.json outputs/rl_ucr/summary.json \
  --output outputs/comparison.csv
```

完整执行、恢复和验收规则见 [`docs/runbook.md`](docs/runbook.md)；
统一区间协议、JSONL 格式、指标与 reward 口径见 [`docs/protocol.md`](docs/protocol.md)。

