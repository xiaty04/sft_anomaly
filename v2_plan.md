# 真实数据上的 LLM / VLM 时序异常检测计划

> **状态：草案（重构 v2，2026-07）**  
> 目标：在真实开源数据集上验证轻量开源大模型（视觉 VLM / 文本 LLM）的时序异常检测能力；零样本 → SFT →（可选）RL 探索。  
> 环境约束：单卡（AutoDL 或同类）+ SSH；数据集、模型、框架全部开源、轻量。

---

## 1. 目标与验收标准

**任务**：给定真实时序窗口，检测其中的异常。标签允许**单点异常**或**区间异常**，**二分类或多分类**均可；实际使用时统一转换为区间输出协议（见 D1）。

**最低验收标准**

| 项 | 标准 |
|----|------|
| 数据 | ≥2 个真实开源数据集跑通（推荐 TSB-AD 子集 + 另一个） |
| 基线 | 同一 eval 集对比：传统基线 ≥1 + 文本 LLM + 视觉 VLM |
| SFT | 至少完成 1 次 SFT（合成为主，可加真实数据），F1 相对零样本可复现提升或明确失败分析 |
| 指标 | point-wise F1 + Affiliation F1（VUS-PR 可选），附 bootstrap 区间 |
| 可复现 | 一键脚本；单卡可跑通推理 + LoRA SFT |

---

## 2. 关键决策点（待拍板）

| # | 决策 | 候选 | 当前倾向 |
|---|------|------|----------|
| **D1** | **标签处理方案** | ① 保留点级 ② 单点异常聚合为区间 ③ 分段分类（每段正常/异常） | 视觉路线必须 ②/③；统一输出 `[{start,end}]` |
| **D2** | **模态选择** | 文本 LLM vs 视觉 VLM | 阶段 1 头对头探针后定，不先验站队 |
| **D3** | **训练数据** | 合成 / 真实 / 混合 | 合成优先（精确可控），真实数据可选加入 |
| **D4** | **后训练路线** | SFT（LoRA）→ 可选 RL | 先 SFT；RL 无具体计划，仅作探索 |

**D1 展开（最重要决策）**：VLM 从折线图上直接指认**单点**异常难度过高、标注噪声大。建议标签统一为**区间级**：单点异常用邻近窗口膨胀成小区间（或转分段分类），模型输出"哪些区间异常"。文本路线采用同一区间协议，保证双模态可比。二分类优先；多分类（异常类型）作为可选扩展。

**D2 展开**：文献对"视觉 vs 文本"无一致结论。用同一批窗口做头对头对比（指标 + 吞吐 + 实现复杂度）后锁定主模态。

**D3 展开（精确性要求）**
- 清洗规则：结构合法 `[{start,end}]`、`start < end`、坐标在窗内；正常样本必须输出 `[]`；合成标签由生成器保证精确，真标与 GT 严格一致
- 合成异常类型：point / range / level shift / trend / season break / freq change

---

## 3. 资源清单（已核查，2026-07）

### 3.1 真实标注数据集

| 数据集 | 性质 | 标签 | 下载入口 | 核查结论 / 适配度 |
|--------|------|------|----------|-------------------|
| **TSB-AD-U / M** | 精筛真实+半真实 | 区间 | [U.zip](https://www.thedatum.org/datasets/TSB-AD-U.zip) / [M.zip](https://www.thedatum.org/datasets/TSB-AD-M.zip) / [repo](https://github.com/TheDatumOrg/TSB-AD) | ✓ 直链 HTTP 200 可用；NeurIPS 2024 精筛，质量最高。**主评估集**，先 U 后 M |
| SKAB v0.9 | 工业水泵实验台，8 通道 | 点 + 集体 | [waico/SKAB](https://github.com/waico/SKAB) / [Kaggle 镜像](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab) | ✓ 可下；体量小，适合冒烟与多类异常测试 |
| SMAP / MSL | NASA 遥测，通道级 | 区间 | [telemanom](https://github.com/khundman/telemanom) / [HF 镜像](https://huggingface.co/datasets/appleparan/telemanom) | ✓ 可下；标签有已知争议，需抽检或只用精筛子集 |
| NAB-real | 真实流量 / 云监控 / 广告 / Twitter | 区间（人工标） | [numenta/NAB](https://github.com/numenta/NAB)（real 类子集） | ✓ 数据在仓库内；领域多样，经典对照 |
| HAI 23.05 | 工控安全 HIL，多变量 | 攻击区间 | [icsdataset/hai](https://github.com/icsdataset/hai) | ✓ 可下（README 内网盘链接）；需预处理，适合工业安全叙事 |
| SMD | 服务器多变量 | 区间（有噪声） | [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)（多镜像） | ✓ 可下（网盘/镜像）；标签噪声需知情 |
| AIOps 2018 KPI | 真实运维 KPI，单变量 | 区间 | [NetManAIOps/KPI-Anomaly-Detection](https://github.com/NetManAIOps/KPI-Anomaly-Detection) | ✓ 官方仓库；单变量区间标签，与任务**高度匹配** |

**推荐组合**：主评估 = **TSB-AD-U 子集 + AIOps 2018 KPI（或 NAB-real）**；SKAB 用于快速冒烟。

### 3.2 相关开源工作

| 工作 | 对本项目的用途 | 入口 | 核查状态 |
|------|----------------|------|----------|
| **AnomLLM** (ICLR 2025) | 合成异常生成、prompt/输出格式、affiliation 评估 | [Rose-STL-Lab/AnomLLM](https://github.com/Rose-STL-Lab/AnomLLM) | ✓ 可用；2024-10 后停更，仅作参考 |
| **MOMENT** | TS 基础模型（掩码预训练），做重建/表征异常分 | [repo](https://github.com/moment-timeseries-foundation-model/moment) | ✓ 活跃；权重 [AutonLab/MOMENT-1-large](https://huggingface.co/AutonLab/MOMENT-1-large) |
| **OmniAnomaly** (KDD 2019) | 多变量无监督 DL 基线 | [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) | ✓ 经典基线 |
| **Anomaly Transformer** (ICLR 2022) | 多变量 SOTA 对照 | [thuml/Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer) | ✓ |
| **VLM-R1** | 视觉 RL（GRPO）参考，3B 权重已开源 | [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) | ✓ 活跃（6k+ stars）；权重 `omlab/VLM-R1-Qwen2.5VL-3B-*` |
| **TSB-AD** (NeurIPS 2024 D&B) | 主评估协议（VUS-PR、切分方式） | [TheDatumOrg/TSB-AD](https://github.com/TheDatumOrg/TSB-AD) | ✓ 活跃 |

文本路线补充参考（可选读）：[SigLLM](https://github.com/sintel-dev/sigllm)（Prompter/Detector 管线）、[LLM-TSAD](https://github.com/junwoopark92/LLM-TSAD)（index-aware prompting）。

### 3.3 轻量开源模型

| 模型 | 规模 | 入口 | 定位 |
|------|------|------|------|
| Qwen3-1.7B | 1.7B | [HF](https://huggingface.co/Qwen/Qwen3-1.7B) / ModelScope | **文本主模型** |
| Qwen3-4B | 4B | [HF](https://huggingface.co/Qwen/Qwen3-4B) / ModelScope | 文本上限档（QLoRA 单卡可行） |
| Qwen3-VL-2B-Instruct | 2B | [HF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) / ModelScope | **视觉主模型** |
| Qwen3-VL-4B-Instruct | 4B | [HF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) / ModelScope | 视觉上限档 |
| MOMENT-1-large | 385M | [HF](https://huggingface.co/AutonLab/MOMENT-1-large) | 重建/预测残差对照 |
| Chronos-Bolt-small | 9M | [HF](https://huggingface.co/amazon/chronos-bolt-small) | 轻量预测残差对照 |

Qwen3 系均为 Apache-2.0；国内优先 ModelScope 镜像下载。单卡显存参考：24G 可跑 4B QLoRA SFT；OOM 先降模型规模。

### 3.4 开源框架

| 框架 | 用途 | 核查状态 |
|------|------|----------|
| Transformers / PEFT / TRL | 模型底座 + LoRA 适配 + SFT | ✓ 主流活跃 |
| Unsloth | 单卡 SFT 加速 | ✓ 活跃 |
| vLLM | 批量推理（OpenAI 兼容） | ✓ 活跃 |
| TimeEval | 传统基线 + 统一评估（VLDB 2022） | ✓ 活跃，pip 可装 |
| ~~Merlion~~ | — | ✗ **仓库已归档**（archived），从清单移除 |

---

## 4. 方法设计

### 4.1 任务与输出协议

给定窗口 \(x_{1:T}\)，输出异常区间列表；无异常输出 `[]`：

```json
[{"start": 120, "end": 145}]
```

- 标签统一为区间级（D1）；点级标签先膨胀为小区间
- 二分类优先；多分类（异常类型）可选扩展
- 解析失败记 format_error，按全零向量计分并单独报错误率

### 4.2 数据协议

```
原始序列 → 切窗(512–1024, stride 可配) → 标准化(z-score / robust)
→ 双视图:  text: index-aware 序列 或 降采样+量化
           image: 带刻度折线图(dpi/轴标签/样式固定)
→ 标签: intervals 列表；元数据: sample_id / split / domain
```

### 4.3 文本路线

1. 可选去趋势 / 去季节
2. 编码：index-aware `[(0,0.12),(1,0.15),…]`（过长则 stride 采样并声明步长），或 SigLLM 式缩放量化字符串
3. Prompt：检测异常 → 只输出 JSON 区间
4. 零样本 → 可选同域 few-shot（1–3 例）→ SFT

### 4.4 视觉路线

1. 渲染**必须保留 x/y 刻度与数值范围**；固定画布、线宽、颜色，避免视觉捷径
2. 使用区间级标签（D1），不做单点指认
3. 长序列：滑窗；可选粗筛 + 精修两阶段（VLM4TS 思路）

### 4.5 传统基线（每张结果表必有）

- Isolation Forest（子序列特征）
- Spectral Residual 或 Matrix Profile（至少一个）
- 可选：Chronos-Bolt-small / MOMENT-1-large 预测-重建残差 + 阈值

### 4.6 指标

主：**point-wise F1**、**Affiliation F1**；辅：VUS-PR、precision/recall、正常样本空输出率、格式错误率；主结果附 bootstrap 95% CI。

---

## 5. 分阶段计划

### 阶段 0：环境打通（0.5–1 天）

租卡（4090 24G 起步）→ SSH → 装依赖（torch / transformers / peft / trl / unsloth / vllm，按最终选型裁剪）→ ModelScope 拉模型 → 冒烟推理。

**退出标准**：CUDA 可用；文本与视觉各完成 1 次合法 JSON 输出。

### 阶段 1：真实数据零样本探针（2–3 天）— 决策关卡

**数据**：TSB-AD-U 抽 30–50 条（多领域）+ AIOps KPI / NAB-real 子集，统一切窗。

| ID | 方法 | 模型 |
|----|------|------|
| B0 | 传统基线 | IsolationForest / SR |
| T0 | 文本 plain | Qwen3-1.7B |
| T1 | 文本 index-aware（±去季节） | Qwen3-1.7B |
| V0 | 视觉零样本（有刻度，区间标签） | Qwen3-VL-2B |
| V1 | 视觉消融（无刻度） | Qwen3-VL-2B |

**退出标准**：F1 / Affi-F1 总表 → **拍板 D1 标签方案 + D2 主模态**（差距明显选胜者；接近选实现简单、吞吐高者）；记录失败模式。

### 阶段 2：后训练 SFT（3–5 天）

- **数据**：默认合成为主（AnomLLM 风格多类异常，每类数百窗，标签由生成器保证精确）；可选加入真实数据（与 eval 按序列 ID 严格隔离）
- **训练**：LoRA SFT（Unsloth / TRL），1–3 epoch；先 20–50 step 冒烟（无 nan、loss 合理、能出 JSON）；导出 adapter，可选 merge
- **评估**：同一 eval 对比 基线 / 零样本 / SFT；按 domain 切片 + bootstrap
- **退出标准**：F1 提升 ≥ 0.03，或完整失败分析（分布偏移、过拟合合成等）

### 阶段 3：扩展与 RL 探索（按需，无硬计划）

1. 扩到完整 TSB-AD-U eval 列表
2. Qwen3-4B / Qwen3-VL-4B 上限对照
3. 同域 few-shot
4. 多变量：通道拼图或逐通道汇总
5. 长序列：滑窗聚合 / 粗到细
6. **RL 探索**：参考 VLM-R1（GRPO，3B 权重已开源）；仅在 SFT 基线稳定后考虑，不作为默认训练路线

---

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 真实标签噪声（SMD / SMAP-MSL） | 主用 TSB-AD 精筛；人工抽检窗口 |
| 点级标注对 VLM 太难 | D1：统一转区间级标签 |
| 上下文超长 | stride 采样 + 显式索引；滑窗 |
| 输出格式不稳定 | 约束解码 / 正则后处理；SFT 强化格式 |
| 合成 → 真实负迁移 | 降低合成比例；仅真标 LoRA 对照 |
| 视觉捷径（靠布局而非形状） | 固定样式；刻度消融实验 |
| 单卡 OOM | 降模型 / batch=1 / grad accum / 冻结视觉塔 |
| 下载慢 | ModelScope + 数据盘缓存 |

---

## 7. 待办

- [ ] 定 **D1 标签方案**：区间化策略（膨胀半径 / 分段粒度）
- [ ] 定 **D2 模态**（阶段 1 探针后）
- [ ] 定主数据集与 eval 子集（推荐 TSB-AD-U + AIOps KPI）
- [ ] 租卡、环境冒烟
- [ ] 零样本探针 → SFT → 评估
- [ ] （探索）RL 方案调研（VLM-R1）
