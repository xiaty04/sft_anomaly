# TSAD-SFT: 时间序列异常检测的视觉 SFT 微调

基于 [AnomLLM](https://github.com/Rose-STL-Lab/AnomLLM) 框架，通过教师蒸馏 + SFT 微调 Qwen3-VL，使其从时间序列折线图中检测异常区间。

## 项目动机

AnomLLM 提出用 VLM 零样本识别时间序列异常，但零样本推理精度有限。本项目在其基础上加入 SFT 环节：用强教师模型标注合成数据，再微调一个轻量 VLM，验证 SFT 能否提升检测指标。

## 整体流程

```
合成数据（AnomLLM）
    │
    ▼
VLM Baseline ──────────────────────────────┐
    │                                      │
    ▼                                      │
教师蒸馏（qwen3.5-plus 标注）              │
    │                                      │
    ▼                                      │
自动清洗（label 一致性 + intervals 校验）   │
    │                                      │
    ▼                                      │
SFT 训练（Qwen3-VL + LoRA + Unsloth）     │
    │                                      │
    ▼                                      ▼
SFT 推理 ──────────────── 对比评估（F1 / Affi-F1）
```

## 关键组件

| 组件 | 说明 |
|------|------|
| **AnomLLM** | 上游框架，提供合成数据生成、VLM 推理链路、评测指标 |
| **教师模型** | DashScope 上的 qwen3.5-plus，对折线图做异常标注 |
| **学生模型** | Qwen3-VL-8B-Instruct，4bit 量化 + LoRA 微调 |
| **训练框架** | Unsloth + TRL SFTTrainer |
| **推理服务** | vLLM（本地 OpenAI 兼容 API） |

## 数据

使用 AnomLLM 提供的 4 类合成时间序列，每类 400 条评估数据：

- **point** — 正弦波中的点异常（随机噪声替换）
- **range** — 高斯噪声中的幅值偏移
- **freq** — 正弦波中的频率突变
- **flat-trend** — 正弦波上叠加的趋势异常

每条数据包含 1000 个时间步的单变量序列及对应的异常区间标签。

## 评估协议

沿用 AnomLLM 的评估方式：
1. 模型输入折线图，输出 JSON 区间列表 `[{"start": ..., "end": ...}, ...]`
2. 区间转为 0/1 向量后，计算 point-wise F1 和 affiliation F1
3. 在相同的 eval 子集上对比 Isolation Forest / VLM zero-shot / SFT 三种方法

## 运行环境

- Google Colab，A100 GPU + 高 RAM 运行时
- 需要 DashScope API Key（教师蒸馏阶段）

## 文件结构

```
.
├── tsad_sft_pipeline.ipynb      # 完整 pipeline notebook
├── AnomLLM/                     # 上游框架（数据生成、推理、评测）
│   └── src/
│       ├── prompt.py            # VLM prompt 和消息构造
│       ├── utils.py             # 输出解析、指标计算
│       ├── online_api.py        # 在线推理入口
│       └── data/synthetic.py    # 合成数据生成
├── tsad_sft_pipeline_issues.md  # 已知问题与修改方案
├── notebook_guide.md            # Notebook 各阶段详细说明
└── archieve/                    # 历史计划文档
```

## 文档

- [Notebook 使用指南](notebook_guide.md) — 各阶段 Cell 说明、参数配置、检查点、运行步骤
- [修改方案](tsad_sft_pipeline_issues.md) — 静态审查发现的协议不一致问题及修复方案
