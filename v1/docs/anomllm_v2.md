# AnomLLM v2 重做方案

> 状态：规划中（方向级）| 定位：v1（TSAD-SFT）的问题驱动重做版
> 任务大框架：**异常检测 + LLM** —— 用大语言模型（含多模态）做时序异常检测
> v2 不沿用 v1 的流水线（Zero-shot → 蒸馏 → SFT → GRPO），只保留大框架，重新选择实现路径

---

## 0. v2 既定决策（已确认，后续方向以此为准）

1. **运行平台：AutoDL（替代 Google Colab）**
   - 目标实例：A100 40G（约 ¥6~8/小时，按小时计费，关机只收数据盘存储费）
   - 模型下载走 ModelScope（魔搭）国内源，不再依赖 HuggingFace 代理
   - 国内访问 DashScope / 数据源延迟远优于 Colab
2. **工作流：SSH 终端（脱离 notebook）**
   - 使用 SSH 命令行 / VS Code Remote 直接开发，不再沿用 v1 的 notebook + Drive tar 包机制
   - 数据持久化改用 AutoDL 独立数据盘（关机保留、续租恢复），废弃 Drive 归档体系
   - 环境用 AutoDL 自定义镜像固化，避免每次重装依赖
3. **放弃教师蒸馏（不做 qwen3.5-plus API 蒸馏）**
   - 理由：v1 蒸馏清洗只过滤 label 不过滤区间位置（§2.1 有毒监督），且依赖闭源 API、成本与延迟不可控
   - 训练数据的生成方式另行决定（候选：直接 GT 监督、合成数据、其他数据源——具体方案待定，见 §4 方向 C）
4. **延续 v1 的教训约束**：评估必须带 bootstrap 置信区间；渲染必须验证刻度方案；GRPO 不设为主线（§2 各 P0/P1 问题仍为设计约束）

---

## 1. v1 现状回顾（做了什么 + 结果）

### 1.1 v1 流水线（已跑通）

```
合成数据（4 类 x 400 = 1600 条）→ VLM 零样本基线（Qwen3-VL-8B 读图）
→ Isolation Forest 基线 → 教师蒸馏（qwen3.5-plus）→ 自动清洗
→ SFT（LoRA r=16, 3 epoch）→ vLLM 评估（F1 / Affi-F1）→ GRPO → 评估
```

- 数据：AnomLLM 合成数据 4 子集（flat-trend/range/point/freq），train=1280 / val=160 / eval=160
- 模态：时间序列渲染成折线图 → VLM 输出异常区间 JSON `[{"start","end"}]`
- 蒸馏：qwen3.5-plus（temperature 0.2），清洗规则 = label 一致 + 结构合法
- 训练：Qwen3-VL-8B 4bit + LoRA（r=16），SFT lr=1e-4 / 3 epoch；GRPO reward = format(1.0)+GT-F1(2.0)

### 1.2 最终结果（eval 160 条）

| 方法 | Point-wise F1 | Affiliation F1 |
|------|:---:|:---:|
| Isolation Forest | 0.110 | 0.476 |
| Qwen3-VL zero-shot | 0.440 | 0.629 |
| **SFT** | **0.552** | **0.755** |
| GRPO | 0.538 | 0.744 |

v1 结论：SFT 最强；GRPO 跑通但低于 SFT。

---

## 2. v1 深度问题诊断（重做依据）

按严重程度排序，每项给出机制与影响。这些是 v1 的教训，v2 方向必须正面回应。

### 2.1 【P0】蒸馏清洗只过滤 label，不过滤区间位置 → 有毒监督

- 清洗条件只看 `teacher_anom == ground_truth` + 结构合法，从不检查 teacher 区间与 GT 的**位置关系**。"label 对、位置错"（教师标 [500,800]，GT 是 [100,200]）的样本照样进训练集，等于教模型把区间标到错误位置
- 影响：SFT 收益混入"格式稳定 + label 模式记忆"，真实定位能力收益未知；此类样本占比从未统计

### 2.2 【P0】渲染去掉坐标刻度 → 边界精度的结构性天花板

- v1 渲染 `set_xticks([])` / `set_yticks([])`，模型只能靠像素比例猜坐标
- 影响：直接解释 point-wise F1 (0.552) 远低于 affi-F1 (0.755)——区域大概对、边界不准；比任何训练超参都更本质，v1 从未讨论

### 2.3 【P0】GRPO 方法论错位 + 结论无统计支持

- GRPO 用 **GT-F1（oracle reward）**，标签已知时 RL 无信息增益；正常样本输出 `[]` 即满分，梯度稀薄
- 160 条 eval 单次运行、无 seed 平均，0.014 差异大概率在噪声范围内——"GRPO 低于 SFT"缺乏显著性

### 2.4 【P1】IF baseline 0.110 存疑

- IF 吃原始数值、VLM 吃图（含信息损失），跨模态比较；0.110 疑似低于 AnomLLM 原论文同类表现，需官方配置复现核对

### 2.5 【P1】泛化受限 + 报告缺切片 + checkpoint 未用

- eval 与 train 同源合成数据（同类 hold-out）；freq 瓶颈口头断言无 by-subset 数字；训练存了 checkpoint 但只评估最终模型

---

## 3. 外部调研（v2 可选路线的依据）

### 3.1 文本路线（非读图）

- **SigLLM（MIT, arXiv 2405.14755）**：时序转文本（缩放/量化/滚动窗口），两条管线——PROMPTER（直接指认异常）vs DETECTOR（预测误差引导）。DETECTOR 平均 F1 0.525，比 PROMPTER 高 135%，但 SOTA DL 仍高约 30%，延迟高
- **Delving into LLMs for Effective TSAD（NeurIPS 2025, KAIST）**：在 AnomLLM benchmark 上发现两大失效模式——复杂正常模式掩盖异常（漏检）、LLM 靠 token 计数定位而计数能力弱（定位差）。解法：**de-seasonalization + index-aware prompting**（输入改为 `[(0,v0),(1,v1),...]` 显式索引）。超越 21 种既有 prompt，F1 平均 +39.9%、最高 +66.6%；纯文本可反超视觉。**直接挑战 v1 所选路线的依据（"视觉优于文本"）**

### 3.2 蒸馏/表征路线

- **AnomalyLLM（IJCAI 2024, 浙大, arXiv 2401.15123）**：LLM teacher 蒸馏 student 表征，推理时 teacher/student 表征差异判异常，15 数据集 SOTA。与 v1 基于的 AnomLLM（读图）是不同工作，勿混淆

### 3.3 其他

- **LLM-AP（2026）**：LLM 生成合成异常做数据增强
- **MDPI 2026 多模态对比**：text-only vs vision-only vs text+vision，MLLM 无法有效融合文本与视觉表征

---

## 4. v2 方向（方向级，暂不展开）

大框架：**异常检测 + LLM**。不沿用 v1 流程，先回答四个方向性问题，再选定组合路径。

### 方向 A：输入模态怎么选（读图 / 读文本 / 读数值）
- v1 默认视觉，但 NeurIPS 2025 证明文本 + 索引感知可反超；AnomLLM 原论文结论是视觉更好——两个结论冲突，需要在自己的数据上重新验证
- 候选：纯文本（index-aware）、纯视觉（刻度渲染）、文本+视觉融合、数值直接输入（时序基础模型式）

### 方向 B：LLM 在检测中的角色（直接检测 / 预测误差 / 表征蒸馏 / 特征提取器）
- 直接区间输出（v1 方式，格式与定位难） vs 预测误差引导（SigLLM DETECTOR，稳但精度受限） vs 表征蒸馏（AnomalyLLM，无监督友好） vs 微调特征提取器（GPT4TS 式）

### 方向 C：训练范式（监督微调 / RL / 零样本+后处理 / 其他）

- **教师蒸馏已放弃（见 §0）**，v1 的"教师蒸馏 + SFT + GRPO"组合不再考虑
- 剩余候选：直接 GT 监督 SFT（跳过教师，数据用 GT 区间直接构造）、纯零样本 + 后处理（不训练，靠 prompt + 启发式修正）、RL（可选，且需规避 v1 的 oracle reward 错位）、无监督/自监督（表征蒸馏式，如 AnomalyLLM——注意该"蒸馏"指 teacher-student 表征差异，与已放弃的教师 API 蒸馏是两回事，仅作方法参考）

### 方向 D：数据与任务设定
- 合成 vs 真实（SWaT/SMAP/MSL）与跨域泛化；有监督 vs 无监督/自监督；区间定位 vs 点级检测 vs 事件级；加入上下文过滤（利用 LLM 语义推理优势）

### 建议的组合方向（待验证）
1. **首选探针**：模态对比实验（同模型、同数据：视觉 vs index-aware 文本 vs 融合）——一步确定模态选择，成本最低
2. **基于结果再选**：若文本/数值胜出 → 走"时序转文本 + 预测误差/直接输出"轻量路线（无需蒸馏，甚至可能无需训练）；若视觉仍胜 → 走"刻度渲染 + GT 直接监督"路线
3. **差异化叙事**：无论哪条路线，都保留"LLM 语义上下文过滤"与"少样本跨域泛化"作为区别于传统方法的卖点

---

## 5. 下一步

1. **环境搭建（AutoDL + SSH）**：开 A100 实例 → SSH/VS Code Remote 接入 → 用 ModelScope 拉取 Qwen3-VL 与训练依赖 → 把 v1 归档数据从 Drive 导出迁移到 AutoDL 数据盘（数据盘即持久化，不再用 tar 包）
2. 确认 v2 方向 A/B/C/D 的取舍（或先跑模态探针实验再定）；训练数据来源决策（替代蒸馏的方案）在选定路径后细化
3. 选定路径后升级本文件为详细版（阶段划分 + 实验矩阵）
