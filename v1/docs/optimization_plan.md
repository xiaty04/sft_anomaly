# TSAD-SFT 优化方案

## 目标与边界

当前优化只聚焦 SFT。Part 1-4 已运行完成，后续实验入口为 `notebooks/part5_drive_experiments.ipynb`，默认从 Google Drive 的 `packs/` 恢复 `part3_results_only_*.tar.gz` 和 `part4_results_only_*.tar.gz`。

优化顺序固定为三步：先判断 SFT 提升来源，再做 SFT 训练方法调优，最后验证在 SFT 输出中加入理由是否带来增益。

## 1. 判断 SFT 提升来源

目的：拆清当前 SFT 的 `0.552/0.755` 提升来自输出格式稳定、异常检出能力、区间边界精度，还是特定子集表现变化。

当前第一步评审：

- 已覆盖 parse、subset、interval bucket 和 valid-pair 四个入口，方向正确。
- 需要补强 affiliation F1。当前项目同时报告 point-wise F1 和 Affi-F1，来源诊断也要同时解释两项指标。
- 需要补强样本级错误表。聚合表能说明平均差异，样本级表能定位 SFT 具体修复了哪些 zero-shot 错误，以及新增了哪些错误。
- 需要补强边界误差分解。TSAD 的核心差异常见于 start/end 偏移、短异常漏检、正常样本误报和异常类型混淆，单一 F1 表很难定位。
- 需要把诊断结果直接映射到第二步 SFT 调优动作，避免诊断完成后仍然只能凭经验选实验。

诊断必须固定的口径：

- 只使用 `eval` split 的 `160` 条样本，保持和当前 `0.552/0.755` 结果一致。
- 三类模型输出统一进入同一套解析逻辑：`qwen-local-0shot`、`sft-0shot`、必要时附带 `grpo-0shot` 作为参考。
- 预测解析失败时按当前 notebook 口径记为全零向量，同时单独记录 `parse_success=False`。
- 每张表同时保留 `f1` 和 `affi_f1`，并保留 `parse_success_rate`、`empty_pred_rate`、样本数。
- 所有对比都保留 `sample_id/subset/pkl_idx/label/anomaly_type/gt_intervals/pred_intervals`，便于回看图片。

第一组表：格式与输出稳定性。

- `diagnostics_parse.csv`：按 method 汇总 parse 成功率、合法 JSON 比例、空预测比例、平均预测区间数、异常样本空预测率、正常样本非空预测率。
- `diagnostics_format_errors.csv`：记录 parse 失败样本，包含原始 completion、错误类型、是否能通过简单截取修复。
- 判断重点：
  - SFT 的 parse 成功率大幅高于 zero-shot，说明收益主要来自输出格式稳定。
  - SFT 的正常样本非空预测率下降，说明误报减少。
  - SFT 的异常样本空预测率下降，说明漏检减少。
- 对应第二步动作：优先检查 loss mask、checkpoint 选择、decode 参数和输出模板；训练调优先关注 parse 稳定性和空预测比例。

第二组表：检测能力与 valid-pair 对比。

- `diagnostics_valid_pair_compare.csv`：只保留 zero-shot 和 SFT 都能成功解析的样本，比较 point-wise F1、Affi-F1、空预测率和区间数量。
- `diagnostics_delta_by_sample.csv`：样本级 delta 表，记录 `sft_f1 - zeroshot_f1`、`sft_affi_f1 - zeroshot_affi_f1`，并标记 `fixed_by_sft/regressed_by_sft/same`。
- 判断重点：
  - valid-pair 上 SFT 仍然领先，说明收益包含真实检测能力和边界定位能力。
  - valid-pair 上差异很小，说明当前 SFT 主要学到格式和输出约束。
  - SFT regression 样本集中于某个 subset 或区间长度，说明后续调优要围绕该切片设计。
- 对应第二步动作：valid-pair 仍领先时优先推进学习率、checkpoint、rank 和 vision layer 消融；valid-pair 差异较小时优先做输出稳定性和早停。

第三组表：子集、异常类型和正常/异常切片。

- `diagnostics_by_subset.csv`：按 `flat-trend/range/point/freq` 统计 F1、Affi-F1、parse 成功率、空预测比例、误报率、漏检率。
- `diagnostics_by_anomaly_type.csv`：按 `trend/range/point/freq/none` 统计同样指标。
- `diagnostics_label_confusion.csv`：把 GT 是否异常和预测是否异常组成 `TP/FP/FN/TN`，按 method 和 subset 汇总。
- 判断重点：
  - freq 切片低，优先关注 vision layer、LoRA 容量和 rationale 分支。
  - point 切片低，优先关注短区间和 start/end 边界偏移。
  - none 切片误报高，优先关注空输出学习、早停和正则化。
- 对应第二步动作：subset 差异明显时，每个训练实验都要保留 by-subset 表；整体平均提升要和瓶颈切片同步报告。

第四组表：区间长度、边界偏移和区间数量。

- `diagnostics_by_interval.csv`：按 `none/short/medium/long` 统计 F1、Affi-F1、parse 成功率、空预测比例。
- `diagnostics_boundary_error.csv`：对可匹配的 GT/pred 区间统计 start 偏移、end 偏移、中心点偏移、长度比例误差、IoU。
- `diagnostics_count_error.csv`：统计预测区间数与 GT 区间数的差值，区分少报、多报和数量正确但边界偏移。
- 判断重点：
  - point-wise F1 低而 Affi-F1 尚可，说明大致区域正确但边界不准。
  - Affi-F1 低，说明预测区域和真实异常段的覆盖关系较弱。
  - 短区间明显低，说明模型对局部尖峰或短窗口变化敏感性不足。
- 对应第二步动作：边界偏移主导时优先尝试学习率下降、checkpoint 选择和更长训练观察；短区间主导时优先观察 vision layer 与 rank。

第五组表：人工复核样本清单。

- `diagnostics_error_review_list.csv`：每个 method 取 top regression、top improvement、freq 错误、短区间错误、正常样本误报，各保留 5-10 个样本。
- 每条记录保留图片路径、GT、zero-shot 输出、SFT 输出、F1/Affi-F1 delta 和错误标签。
- 复核目的：确认低分来自真实预测错误、坐标解析问题、图片可读性问题，还是 GT/teacher 噪声。

最终判读规则：

| 诊断现象 | 结论 | 第二步优先动作 |
|---|---|---|
| parse 成功率贡献最大 | SFT 主要稳定了输出协议 | loss mask、checkpoint 选择、decode 模板、早停 |
| valid-pair 上 SFT 仍领先 | SFT 学到检测和定位能力 | lr/warmup/cosine、rank sweep、vision 消融 |
| freq 明显拖低均值 | 频率变化视觉表征是瓶颈 | vision layer 消融、`r=32`、rationale 分支专项观察 |
| 短区间明显拖低均值 | 局部异常敏感性是瓶颈 | 降学习率、checkpoint 选择、vision layer、rank |
| 正常样本误报高 | 空输出和负样本边界需要强化 | 早停、weight decay、LoRA dropout、输出稳定性检查 |
| start/end 偏移大 | 边界回归式定位能力需要增强 | checkpoint 选择、较低 lr、Affi-F1 与 point-wise F1 联合选模 |

## 2. 围绕 SFT 方法做调优

目标：保持当前图像输入、数据划分和评估协议，优先调 SFT 训练过程本身。

当前适配前提：

- 训练集只有 `1280` 条，验证集和 eval split 各 `160` 条；这是小规模、教师蒸馏、图像输入、区间 JSON 输出的 SFT。
- 当前完整训练配置为 `3 epoch`、有效 batch size `16`、`learning_rate=1e-4`、`r=16`、`lora_alpha=16`、`lora_dropout=0.0`、`finetune_vision_layers=True`。
- 当前要超过的基线是 SFT `0.552/0.755`，同时保持 parse 成功率、freq 子集和短异常区间表现稳定。
- eval split 只用于最终确认；模型选择优先使用 val split，避免把 eval 变成调参依据。

优先级 0：先核对训练口径。

- 核对 SFT loss 是否主要落在 assistant JSON 区间答案上。这个任务的有效监督信号是 `[{start, end}]`，若 prompt 与图片占据过多 loss 权重，训练会把容量浪费在复述输入模板上。
- 固定 seed、训练配置摘要、checkpoint 路径、训练日志保存方式。后续每个实验只改一个主变量，便于判断收益来源。
- 记录训练样本中的区间长度、异常类型和正常样本比例。freq 和短异常是当前疑似瓶颈，后续指标必须单独报告这些切片。

优先级 1：checkpoint 选择。

- 原理：小数据 SFT 中，训练 loss 继续下降时，生成格式和边界定位可能已经开始过拟合教师输出。用最后一个 checkpoint 直接评估会把训练轮数和 checkpoint 选择混在一起。
- 做法：保留 `save_steps=50` 或 `save_steps=100`，对每个 checkpoint 在 val split 上推理，统计 val point-wise F1、val affiliation F1、parse 成功率、空预测比例、by-subset F1。
- 适配当前数据：`160` 条 val 足够做轻量 checkpoint 排序；eval split 保留给最终一次对比。选择规则建议用 `val F1 + val Affi-F1 + parse 成功率` 共同排序，freq 和短异常作为 tie-breaker。
- 推荐实验：先复跑当前配置，只增加 checkpoint 评估，建立 `sft_ckpt_select_base`。这是成本最低、解释最清楚的一步。

优先级 2：学习率、warmup 和 scheduler。

- 原理：当前 `1e-4` 对多模态 LoRA SFT 偏激进，尤其在 `finetune_vision_layers=True` 时，较大学习率容易扰动视觉特征和输出格式。warmup 可以缓和早期大梯度，cosine scheduler 可以让后期更新幅度收敛。
- 做法：以当前配置为基线，比较 `learning_rate=5e-5` 和 `2e-5`，加入 `warmup_ratio=0.05` 或 `0.10`、`lr_scheduler_type="cosine"`、`max_grad_norm=0.3` 或 `1.0`。
- 适配当前数据：训练集 `1280` 条、有效 batch `16`，3 epoch 大约只有数百个更新步；warmup 比例过大时有效学习阶段会缩短。优先试 `5e-5 + warmup_ratio=0.05 + cosine`，再试更保守的 `2e-5`。
- 推荐实验：`sft_lr5e5_cosine` 作为第一组主实验；若 parse 或 by-subset F1 波动明显，再跑 `sft_lr2e5_cosine`。

优先级 3：训练轮数、早停和轻正则化。

- 原理：当前目标是短 JSON 区间，输出空间小，训练过久会增强对教师边界风格的记忆。weight decay、梯度裁剪和少量 LoRA dropout 可以降低这种记忆化趋势。
- 做法：在优先级 2 的较优学习率下比较 `1/2/3/5 epoch`，保留 checkpoint 选择；同时测试 `weight_decay=0.01`、`lora_dropout=0.05`。
- 适配当前数据：正常样本输出 `[]`，异常样本输出短列表，格式学习很快；检测能力和边界定位需要更多视觉对齐。若 1 epoch parse 已稳定但 F1 偏低，继续到 2/3 epoch；若 5 epoch train loss 更低但 val F1 下降，主线采用早停 checkpoint。
- 推荐实验：`sft_epoch_sweep_lr_best`，先用同一组 checkpoint 得到 1/2/3 epoch 对比，5 epoch 作为过拟合探针。

优先级 4：LoRA 容量调整。

- 原理：LoRA rank 控制可训练低秩更新的容量。rank 太小会限制视觉模式到区间 JSON 的映射；rank 太大更容易在 `1280` 条蒸馏样本上记忆教师噪声。
- 做法：以最优学习率和 checkpoint 选择规则为前提，比较 `r=8/16/32`，保持 `lora_alpha` 与 rank 同步或使用稳定缩放策略。
- 适配当前数据：当前 `r=16` 是合理起点。`r=32` 主要用于观察 freq 子集和边界定位是否获益；`r=8` 用作低容量对照，帮助判断当前收益来自格式学习还是视觉检测能力。
- 推荐实验：先跑 `r=32`。若整体 F1 提升但 parse 波动，加入更低学习率或 `lora_dropout=0.05` 复验。

优先级 5：fine-tune vision layer 消融。

- 原理：`finetune_vision_layers=True` 会让图像编码端参与适配，更可能改善频率变化、局部形态和边界定位；冻结 vision layer 会把训练集中到语言侧输出格式和视觉特征到 JSON 的映射。
- 做法：保留当前数据和输出格式，比较两组：
  - `vision_on`：当前设置，`finetune_vision_layers=True`。
  - `vision_frozen`：`finetune_vision_layers=False`，语言层、attention 和 MLP LoRA 保持开启。
- 适配当前数据：若 `vision_frozen` 接近当前 F1，SFT 收益主要来自输出约束和语言侧映射；若 `vision_on` 在 freq、短异常和边界 Affi-F1 上明显领先，视觉层适配有保留价值。
- 推荐实验：放在学习率和 checkpoint 选择稳定之后执行。vision 消融会改变训练动力学，直接和当前最后 checkpoint 对比会混入调度差异。

优先级 6：PEFT 方法替换评估。

- 主线继续使用 LoRA/QLoRA。当前项目已经通过 Unsloth + 4-bit LoRA 跑通训练、导出和 vLLM 评估，工程风险最低，结果最容易和 Part 3 基线对齐。
- `rsLoRA`：适合在提高 rank 后稳定缩放，优先级高于完整替换 PEFT 方法。若 `r=32` 或更高 rank 出现训练不稳，优先尝试 rsLoRA 风格缩放。
- `DoRA`：把权重更新拆成方向和幅度，容量通常强于普通 LoRA，可能改善边界细节和 freq 子集；代价是显存、训练时间和框架兼容性压力更高。适合作为第二轮实验，前提是当前 notebook 环境直接支持。
- `AdaLoRA`：动态分配 rank，理论上适合不同层贡献差异大的任务；当前多模态 VLM、Unsloth 导出和 Colab 复现实验链路会增加调试成本。优先级低于学习率、checkpoint、vision 消融和 rank sweep。
- `IA3`、prefix tuning、prompt tuning：参数更少，适合作为低成本下界实验；这个任务需要从图像细节定位连续区间，容量通常偏紧，主线收益预期较低。

建议的第一轮实验矩阵：

| 实验名 | 改动 | 目的 | 通过信号 |
|---|---|---|---|
| `sft_ckpt_select_base` | 当前配置 + val checkpoint 选择 | 分离训练过程和 checkpoint 选择 | val 最优 checkpoint 在 eval 超过或接近当前最后 checkpoint |
| `sft_lr5e5_cosine` | `lr=5e-5`、`warmup_ratio=0.05`、`cosine`、grad clipping | 稳定多模态 LoRA 更新 | parse 稳定，F1/Affi-F1 提升 |
| `sft_epoch_sweep_lr_best` | 1/2/3/5 epoch + 同一选择规则 | 找到训练时长和过拟合拐点 | 早停 checkpoint 优于固定 3 epoch |
| `sft_lora_r32` | `r=32`，其余沿用最优训练设置 | 测试容量是否限制检测能力 | freq 或短异常提升，整体指标保持提升 |
| `sft_vision_frozen` | `finetune_vision_layers=False` | 判断视觉层适配贡献 | 对比 `vision_on` 的 by-subset 和 Affi-F1 |

每个实验必须保存：

- `sft_eval_metrics.csv`
- val split 上的同格式评估 CSV
- `diagnostics_parse.csv`
- `diagnostics_by_subset.csv`
- `diagnostics_by_interval.csv`
- 训练 log history
- 当前训练配置摘要

通过标准：

- 同一 eval split 上超过当前 SFT `0.552/0.755`。
- parse 成功率保持稳定。
- freq 和短异常切片保持或提升。
- 结果优先按 point-wise F1、affiliation F1、parse 成功率、freq 子集 F1 的顺序判断。

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
2. 建立 `sft_ckpt_select_base`，先确认当前配置下的最佳 checkpoint。
3. 执行 `sft_lr5e5_cosine`，把学习率、warmup、scheduler 和 grad clipping 作为第一组训练调优。
4. 根据 val 和 eval 结果，再推进 epoch sweep、`r=32`、`vision_frozen`。
5. PEFT 替换作为第二轮实验；优先考虑 rsLoRA 或 DoRA，保留 LoRA/QLoRA 主线作为对照。
6. 单独建立 `sft_rationale_short` 分支实验。
7. 用相同 eval split 对比 plain SFT、调优 SFT、rationale SFT。

当前主线先证明 SFT 的提升来源，再用 checkpoint 选择和学习率调度建立可复验的 SFT 调优基线。
