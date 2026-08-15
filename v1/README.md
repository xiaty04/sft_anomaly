# TSAD-SFT

TSAD-SFT 是一个基于 AnomLLM 的视觉时间序列异常检测项目。当前主线固定为 `Zero-shot -> Teacher Distillation -> SFT -> GRPO`，运行环境为 Google Colab A100，Notebook 在 Colab 中执行。

Part 1-4 已运行完成。后续诊断、后处理和新实验统一从 Google Drive 已保存的归档恢复数据，并在新的实验 notebook 中进行。

## 当前状态

| 方法 | Point-wise F1 | Affiliation F1 |
|------|:---:|:---:|
| Isolation Forest | 0.110 | 0.476 |
| Qwen3-VL zero-shot | 0.440 | 0.629 |
| SFT | 0.552 | 0.755 |
| GRPO | 0.538 | 0.744 |

SFT 是当前最强结果。GRPO 已完成训练与评估流程，当前 eval 指标低于 SFT。

## 目录

```
.
├── notebooks/part1.ipynb      # Baseline 与检查点 B
├── notebooks/part2.ipynb      # 教师蒸馏与 SFT 数据准备
├── notebooks/part3.ipynb      # SFT 训练、导出与评估
├── notebooks/part4.ipynb      # GRPO 训练、导出与评估
├── notebooks/part5_drive_experiments.ipynb
│                               # 从 Drive 恢复数据做后续诊断和实验
├── docs/
│   ├── notebook_guide.md      # Notebook 使用指南
│   ├── project_report.md      # 当前项目报告与证据边界
│   ├── optimization_plan.md   # 后续诊断与优化方案
│   └── references/            # Unsloth / TRL 离线参考资料
└── archive/
    ├── code/                  # 旧 AnomLLM 本地副本归档
    ├── figures/               # 历史图片输出
    ├── notebook_exports/      # notebook 文本导出
    ├── old_plans/             # 早期计划文档
    └── papers/                # 参考论文
```

## 入口文档

- [Notebook 使用指南](docs/notebook_guide.md)
- [项目报告](docs/project_report.md)
- [优化方案](docs/optimization_plan.md)

## 运行约束

Notebook 只在 Google Colab 中运行。教师蒸馏阶段需要 DashScope API Key。后续实验优先使用 `notebooks/part5_drive_experiments.ipynb`，从 `/content/drive/MyDrive/tsad_anomaly/packs/` 中的 `part3_results_only_*.tar.gz` 和 `part4_results_only_*.tar.gz` 恢复结果。
