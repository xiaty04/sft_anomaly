# TSAD-SFT 优化方案

## 目标与边界

当前优化只聚焦 SFT。Part 1-4 已运行完成，后续实验入口为 `notebooks/part5_drive_experiments.ipynb`，默认从 Google Drive 的 `packs/` 恢复 `part3_results_only_*.tar.gz` 和 `part4_results_only_*.tar.gz`。

优化顺序固定为三步：先判断 SFT 提升来源，再做 SFT 训练方法调优，最后验证在 SFT 输出中加入理由是否带来增益。

## 1. 判断 SFT 提升来源

目的：拆清当前 SFT 的 `0.552/0.755` 提升来自输出格式稳定、异常检出能力、区间边界精度，还是特定子集表现变化。

在 Part 5 中先生成以下诊断表：

- `diagnostics_parse.csv`：统计 zero-shot、SFT 的 parse 成功率、合法 JSON 比例、空预测比例。
- `diagnostics_by_subset.csv`：按 `flat-trend/range/point/freq` 统计 F1、parse 成功率和空预测比例。
- `diagnostics_by_interval.csv`：按正常样本、短异常、中等异常、长异常统计 F1。
- `diagnostics_valid_pair_compare.csv`：只在 zero-shot 和 SFT 都能成功解析的样本上比较 F1。

判断标准：

- 若 SFT 的主要增益来自 parse 成功率提升，后续重点放在输出约束和格式稳定性。
- 若双方都 parse 成功的样本上 SFT 仍明显领先，后续重点放在 SFT 学到的检测能力。
- 若 freq 或短异常区间明显拖低均值，后续调优要单独报告这些切片。

## 2. 围绕 SFT 方法做调优

目标：保持当前图像输入、数据划分和评估协议，优先调 SFT 训练过程本身。

候选实验：

- 学习率与调度：把完整训练学习率从当前配置下调，加入 warmup、cosine scheduler，并记录 eval loss 与最终 F1 的关系。
- 正则化：加入 weight decay、max grad norm，观察过拟合和格式退化情况。
- checkpoint 选择：保存多个 checkpoint，用 val loss、parse 成功率和 val F1 共同选模型，再在 eval split 上做最终对比。
- LoRA 容量：比较 `r=16` 与更高 rank，在显存允许范围内观察检测 F1 和格式稳定性。
- 训练轮数：比较 1/2/3/5 epoch，重点看 eval F1 是否随 train loss 继续下降而退化。

每个实验必须保存：

- `sft_eval_metrics.csv`
- `diagnostics_parse.csv`
- `diagnostics_by_subset.csv`
- 训练 log history
- 当前训练配置摘要

通过标准：

- 同一 eval split 上超过当前 SFT `0.552/0.755`。
- parse 成功率保持稳定。
- freq 和短异常切片至少保持不下降。

## 3. SFT 加入理由

目标：验证 teacher rationale 能否帮助模型学习异常判断依据，尤其是频率变化、局部模式改变和边界定位。

数据来源：

- 使用 Part 2 蒸馏结果中的 teacher `rationale` 字段。
- 先在 Part 5 抽样检查 rationale 质量，过滤空泛理由、异常类型不一致、坐标与区间明显冲突的样本。

推荐输出格式：

```text
Reason: The signal pattern changes from regular oscillation to dense high-frequency fluctuation around x=240-310.
Answer: [{"start": 240, "end": 310}]
```

训练分支：

- `sft_plain`：当前 SFT baseline。
- `sft_rationale_short`：一到两句理由，加最后一行 `Answer: [...]`。
- `sft_rationale_filtered`：只使用通过 rationale 质量过滤的样本。

评估方式：

- F1 只解析 `Answer:` 后的 JSON 区间列表。
- 额外统计 `Answer:` 是否存在、JSON 是否合法、理由长度、理由中是否包含异常类型或坐标。
- 与 plain SFT 在同一 eval split 上比较总体 F1、per-subset F1、parse 成功率。

风险控制：

- 理由长度控制在一到两句。
- 最后一行固定为 `Answer: [...]`。
- 若 rationale 分支 parse 成功率下降，优先缩短理由和强化输出模板。
- 若 rationale 对 freq 有提升但整体下降，保留为专项分支继续分析。

## 推荐执行顺序

1. 在 Part 5 先跑 SFT 来源诊断。
2. 根据诊断结果选择 1-2 个 SFT 训练调优实验。
3. 单独建立 `sft_rationale_short` 分支实验。
4. 用相同 eval split 对比 plain SFT、调优 SFT、rationale SFT。

当前优先级最高的是先证明 SFT 的提升来源，再决定 rationale 是否值得进入主线。
