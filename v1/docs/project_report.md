# TSAD-SFT 项目报告

## 项目定位

TSAD-SFT 基于 AnomLLM 框架，把时间序列异常检测转成视觉理解任务：模型读取折线图，输出异常区间 JSON 列表。当前主线固定为 `Zero-shot -> Teacher Distillation -> SFT -> GRPO`。

运行环境为 Google Colab A100。Notebook 通过 Google Drive 保存中间产物、模型和评估结果，下一阶段从 Drive 归档恢复。Part 1-4 已完成，后续实验使用新的 notebook 从 Drive 保存归档恢复数据。

## 已验证流程

| 阶段 | Notebook | 作用 | 主要产物 |
|------|----------|------|----------|
| Baseline | `notebooks/part1.ipynb` | 跑 Qwen3-VL zero-shot 与 Isolation Forest | `before_ckptB_*.tar.gz`、`baseline_compare.csv` |
| Distillation | `notebooks/part2.ipynb` | 使用 qwen3.5-plus 蒸馏并清洗训练样本 | `sft_manifest.csv`、`sft_final.jsonl`、`before_stage8_*.tar.gz` |
| SFT | `notebooks/part3.ipynb` | 训练、导出并评估 SFT 模型 | `qwen3vl-tsad-merged`、`sft_eval_metrics.csv` |
| GRPO | `notebooks/part4.ipynb` | 在 SFT 模型上继续做 GRPO 并评估 | `qwen3vl-tsad-grpo-merged`、`grpo_eval_metrics.csv` |
| Post-hoc | `notebooks/part5_drive_experiments.ipynb` | 从 Drive 保存结果恢复数据，做诊断、后处理和新实验 | `diagnostics_*.csv` |

核心依赖分工稳定：AnomLLM 负责数据、prompt、推理入口和指标；vLLM 负责 OpenAI-compatible 推理服务；Unsloth 负责 4-bit VLM 加载和 LoRA；TRL 负责 SFTTrainer 与 GRPOTrainer。

## 当前结果

以下结果来自相同 eval split 的四方对比。

| 方法 | Point-wise F1 | Affiliation F1 |
|------|:---:|:---:|
| Isolation Forest | 0.110 | 0.476 |
| Qwen3-VL zero-shot | 0.440 | 0.629 |
| SFT | 0.552 | 0.755 |
| GRPO | 0.538 | 0.744 |

SFT 是当前最强结果。GRPO 流程已经跑通，当前指标低于 SFT。

## 证据边界

已验证事实：

- 四个 Colab notebook 的阶段划分、Drive tar 交接、vLLM 推理、SFT/GRPO 训练和同 split 评估已经形成完整流程。
- 当前 eval 表中，SFT 相比 zero-shot 提升 `0.112` point-wise F1 和 `0.126` affiliation F1。
- 当前 eval 表中，GRPO 相比 SFT 下降 `0.014` point-wise F1 和 `0.011` affiliation F1。

推断分析：

- SFT 的收益包含两类来源：输出 JSON 格式更稳定，以及异常区间检测更准确。两者比例需要通过 parse 成功率和 parse 成功样本上的 F1 对比确认。
- GRPO 当前低于 SFT，可能与候选样本差异小、reward 信号密度有限、视觉层冻结和训练步数有关。这个判断需要 reward 分布和生成样本质量统计支持。
- freq 子集是主要瓶颈。频率变化在当前折线图渲染下可辨识度较低，适合作为下一阶段错误切片重点。

待补证据：

- zero-shot、SFT、GRPO 的 parse 成功率、合法 JSON 比例、空预测比例。
- 按 subset、正常/异常、区间长度、区间数量切分后的 F1。
- GRPO 训练中的 reward 均值、方差、格式合法率和候选间差异。
- 图像渲染、后处理、自一致性推理和 GRPO 参数调整的消融结果。
- Part 5 后续实验产生的诊断 CSV 和消融 CSV。

## 当前结论

当前项目可以稳定表述为：基于 AnomLLM 构建了一个视觉 TSAD Colab 管线，完成了 zero-shot baseline、教师蒸馏、SFT 和 GRPO 的训练评估闭环；在当前 eval split 上，SFT 是最强版本，GRPO 跑通但指标低于 SFT。后续优化应先补诊断证据，再推进训练和推理策略调整。
