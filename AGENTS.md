# AGENTS.md

## 项目概览

TSAD-SFT：基于视觉 SFT 微调的时间序列异常检测项目。该项目基于 AnomLLM 框架，使用教师蒸馏 + SFT 微调 Qwen3-VL，以从时间序列折线图中检测异常区间。

## 流水线

```
合成数据（4 类 x 400）→ VLM 零样本基线 → 教师蒸馏（qwen3.5-plus）
→ 自动清洗 → SFT 训练（Qwen3-VL-8B + LoRA + Unsloth）→ SFT 推理（vLLM）→ 评估（F1 / Affi-F1）
```

## 关键约束

- **不要尝试直接运行 `.ipynb`** —— Notebook 永远只在 Google Colab（A100 GPU）上运行，不要在本地环境执行
- **不要编写过度冗余的代码** —— 保持实现简洁，避免不必要的抽象、重复逻辑和样板代码
- 运行环境：Google Colab；教师蒸馏阶段需要 DashScope API Key

## 文件结构

```
tsad_sft_pipeline.ipynb       # 完整流水线 Notebook（仅在 Colab 上运行）
AnomLLM/src/                  # 上游框架代码
  prompt.py                   # VLM 提示词与消息格式化
  utils.py                    # 输出解析、F1 / Affiliation 指标
  data/synthetic.py           # 4 类合成异常数据
  config.py                   # 模型配置
  online_api.py               # 在线推理与重试逻辑
  baselines/isoforest.py      # Isolation Forest 基线
tsad_sft_pipeline_issues.md   # 已知问题与修复建议
notebook_guide.md             # 分阶段 Notebook 使用指南
```

## 技术栈

- 模型：Qwen3-VL-8B（student），qwen3.5-plus（teacher）
- 训练：Unsloth + TRL SFTTrainer，4-bit quantization，LoRA（rank=16，alpha=16）
- 推理：vLLM（OpenAI-compatible API）
- 指标：point-wise F1，affiliation F1

## 已知问题（见 `tsad_sft_pipeline_issues.md`）

1. **输出格式不匹配** —— 训练输出是 XML 风格，评估阶段期望纯 JSON
2. **输入 / prompt 不匹配** —— 训练时的消息顺序与 AnomLLM 约定不一致
3. 清洗逻辑较弱、重启后数据重复、YAML 键重复

## 约定

- VLM 输出格式：`[{"start": N, "end": M}, ...]`
- 语言：文档使用中文，代码使用英文
- 数据：1000 步单变量时间序列，带异常区间标签
