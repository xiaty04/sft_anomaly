# v2 数据与开源资料

本文档只记录资源入口和资源本身的信息。数据规模、维度和下载方式可能随版本变化，使用前需要核对具体版本。

---

## 1. 真实数据集

磁盘大小会随压缩格式和数据版本变化；标记为“约”的数字需要以实际下载文件为准。

| 数据集 | 数量与大小 | 单条数据形状 | 标签 | 官方切分与训练转换 |
|---|---|---|---|---|
| [TSB-AD-U](https://github.com/TheDatumOrg/TSB-AD) | 870 条单变量 CSV；zip 约 69.5 MB，解压约 329.6 MB；来自 24 个源数据集 | 每条约为 `(T, 2)`：`Data, Label`，不同序列的 `T` 不同 | 逐点 0/1 | 连续的 1 合并成区间；需保留原始来源和原始切分，防止与 NAB、UCR、Yahoo 等重复 |
| TSB-AD-M | 多变量聚合集合；当前下载包约 540 MB，具体序列数和维度随版本核验 | 每条为 `(T, D)`，`D` 由来源数据决定 | 逐点或区间标签，统一版本通常转成逐点 0/1 | 保留 entity、source dataset 和通道信息；不能把通道直接当成独立实体 |
| [NAB](https://github.com/numenta/NAB) | 58 个 CSV，约 9.6 MB；其中约 47 个属于真实数据类别 | 每条为 `(T, 1)` 数值序列，另有 timestamp | `combined_windows.json` 提供异常时间窗口 | 时间戳窗口映射成局部索引区间；窗口是容忍范围，不是精确逐点 GT |
| [UCR Anomaly Archive](https://www.cs.ucr.edu/~eamonn/Time_Series_Data_Mining_UCR.html) | 250 条单变量序列；长度差异较大 | 每条为 `(T, 1)` | 文件名通常编码 train end、anomaly start、anomaly end | 保留文件给出的训练边界；异常区间可直接转换成 `[start,end)` |
| [SKAB](https://github.com/waico/SKAB) | 35 个实验 CSV，约 4.4 MB | 每个实验约为 `(T, 8)` 传感器特征，另含标签列；`T` 随实验变化 | 逐点 `anomaly` 和 `changepoint` | 连续 anomaly 点合并成区间；changepoint 不直接当作异常区间；按实验文件切分 |
| [AIOps 2018 KPI](https://github.com/NetManAIOps/KPI-Anomaly-Detection) | 约 29 条 KPI；公开 train 约 99 MB、test 约 90 MB | 长表格式，每行含 KPI ID、timestamp、value、label；按 ID 还原为多条 `(T,1)` 序列 | 逐点 0/1 | 按 KPI ID 组织和切分；连续 1 合并成区间；处理缺失时间戳和不等长序列 |
| [SMAP](https://github.com/khundman/telemanom) | 常用预处理版本约 55 条目标序列；合并矩阵 train 约 `135183×25`，test 约 `427617×25` | 多变量 `(T,25)`，通常指定一个目标通道 | 测试异常区间及逐点展开标签 | 保留官方 train/test；需要明确是目标通道任务还是完整多变量输入 |
| [MSL](https://github.com/khundman/telemanom) | 常用预处理版本约 27 条目标序列；合并矩阵 train 约 `58317×55`，test 约 `73729×55` | 多变量 `(T,55)`，通常指定一个目标通道 | 测试异常区间及逐点展开标签 | 保留官方 train/test；核对所用仓库的区间索引版本 |
| [SMD](https://github.com/NetManAIOps/OmniAnomaly) | 28 台机器，每台分别有 train/test 文件 | 每台机器约为 `(T,38)` | 测试集逐点 0/1 | 以 machine ID 为实体；连续 1 合并成区间；不要把同一事件的 38 个通道当成 38 个独立标签样本 |
| [PSM](https://github.com/eBay/RANSynCoders) | 常用版本 train 约 132481 点、test 约 87841 点 | 多变量，约 `train=(132481,25)`、`test=(87841,25)` | 测试逐点 0/1 | 保留官方时间切分；连续 1 合并成区间；标签对应整体多变量状态 |
| [Exathlon](https://github.com/exathlonbenchmark/exathlon) | 约 93 条 Spark 应用运行 trace；磁盘规模和处理后维度随版本变化 | 每条为不等长多变量 `(T,D)`，`D` 取决于保留的系统指标 | 异常区间、异常类型及部分解释信息 | 按 application/execution 切分；可生成“区间”或“区间+类型”训练目标 |
| [SWaT](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) | 约 11 天运行数据，7 天正常、4 天攻击；约 36 个攻击事件 | 约 `(T,51)` 传感器和执行器变量 | 攻击起止时间和逐点 Normal/Attack 标签 | 官方正常段用于 train、攻击段用于 test；属于公开可申请数据 |
| [WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) | 约 16 天数据，14 天正常、2 天攻击；约 15 个攻击事件 | 常用清洗版本约 `(T,123)`，原始字段数随版本变化 | 攻击起止时间或逐点标签 | 保留官方正常/攻击划分；需要清理时间戳、常量列和缺失值；公开可申请 |
| [HAI](https://github.com/icsdataset/hai) | 多个 release；文件通常为数百 MB 到数 GB，数量和维度随版本变化 | 多变量长表 `(T,D)`，包含工业传感器和控制变量 | 逐点攻击标签、攻击区间或攻击描述，取决于 release | 必须固定具体版本；按官方 train/test 文件组织，不混用不同 release |
| [Yahoo S5](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70) | 367 条单变量序列：A1 约 67 条真实序列，A2–A4 各约 100 条合成序列 | 每条为不等长 `(T,1)`，含 timestamp/value | 逐点异常标签 | 真实 A1 与合成 A2–A4 分开使用；连续 1 合并为区间；下载通常需要 Webscope 条款 |
| [BATADAL](https://www.batadal.net/) | 多个供水系统 train/test 文件；规模为数千到数万时间点 | 多变量 `(T,D)`，常见版本约 40 余个传感器和控制变量 | 攻击起止区间或攻击标签 | 保留竞赛给定的数据文件和攻击区间；版本及标签可见性需下载后核验 |

### 1.1 聚合与评测框架

[TimeEval](https://github.com/TimeEval/TimeEval) 是数据和算法评测框架，不是独立训练数据集。它收录并标准化多个 TSAD 数据源，提供数据元信息、算法执行和评测支持。

---

## 2. 合成数据资源

### 2.1 AnomLLM Synthetic Dataset

- **入口**：[Rose-STL-Lab/AnomLLM](https://github.com/Rose-STL-Lab/AnomLLM)
- **现有规模**：4 类 × 400 条，共 1600 条。
- **形状**：每条为长度 1000 的单变量序列 `(1000,1)`，并有对应 PNG 折线图。
- **标签**：生成器直接给出异常区间；正常样本对应 `[]`。
- **类型**：point、range、frequency、flat/trend。
- **输出形式**：时间序列、对应折线图和精确异常区间。

### 2.2 GutenTAG

GutenTAG 是 TimeEval 生态中的可控合成数据生成工具。数据数量和长度由配置决定，输出通常是 `(T,1)` 或 `(T,D)` 序列以及精确逐点标签。它支持配置正常背景、异常类型、位置、长度、幅度和污染率。

---

## 3. LLM/VLM 时序异常检测

### 3.1 AnomLLM

- **入口**：[Rose-STL-Lab/AnomLLM](https://github.com/Rose-STL-Lab/AnomLLM)
- **输入**：时间序列折线图及文本提示。
- **输出**：结构化异常区间。
- **代码内容**：数据加载、折线图渲染、prompt、推理、区间解析和评估。

AnomLLM 把时间序列异常检测转成视觉理解和区间生成问题，使用视觉语言模型读取折线图并生成结构化异常区间。

### 3.2 SigLLM

- **论文**：[SigLLM](https://arxiv.org/abs/2405.14755)
- **代码**：[sintel-dev/sigllm](https://github.com/sintel-dev/sigllm)
- **输入**：缩放、量化和分段后的数值文本。
- **输出**：直接异常判断，或通过预测误差生成异常分数。

SigLLM 使用文本化时间序列，提供“直接生成异常”和“先预测、再根据残差检测”两种路线。

### 3.3 Delving into LLMs for Effective TSAD / LLM-TSAD

- **代码**：[junwoopark92/LLM-TSAD](https://github.com/junwoopark92/LLM-TSAD)
- **输入**：去季节化并带显式索引的数值文本。
- **输出**：异常位置或区间。

该工作研究去季节化和 index-aware 数值表示，指出复杂正常周期和 token 位置计数会影响 LLM 的异常定位。

### 3.4 AnomalyLLM

- **论文**：[AnomalyLLM](https://arxiv.org/abs/2401.15123)
- **输入**：数值时间序列。
- **输出**：基于 teacher/student 表征差异得到的异常分数。

AnomalyLLM 不读取折线图或直接生成区间。它通过 teacher/student 表征差异产生异常分数。

### 3.5 时间序列基础模型

MOMENT、Chronos、TimesFM 等模型可通过预测残差、重构误差或通用 embedding 产生异常分数。它们通常不直接输出异常区间，需要额外进行阈值判断和区间合并。

---

## 4. 传统与深度 TSAD

| 方法 | 输入与输出 | 与区间检测的关系 |
|---|---|---|
| Isolation Forest | 点或窗口特征 → anomaly score | 选择阈值后把连续异常点合并成区间 |
| Spectral Residual | 单变量序列 → 逐点显著性分数 | 适合局部突变，阈值决定区间 |
| Matrix Profile | 子序列 → discord distance | subsequence length 直接影响预测区间长度 |
| OmniAnomaly | 多变量序列 → 概率/重构异常分数 | 常用于 SMD、SMAP/MSL；需要阈值和后处理 |
| USAD | 多变量序列 → 重构误差 | 不直接生成区间 |
| TranAD | 多变量窗口 → 逐点异常分数 | 常用于 SWaT、WADI、SMD、SMAP/MSL、PSM |
| Anomaly Transformer | 时间关联 → association discrepancy | 输出逐点分数，常见结果需核对 point adjustment |
| DCdetector | 多尺度时序表示 → 逐点异常分数 | 用于研究多尺度表示对短异常和持续异常的作用 |

## 5. 区间异常评估

| 指标 | 评价内容 | 主要限制 |
|---|---|---|
| Point-wise F1 | 把区间展开为逐点 0/1 后计算 | 长区间会主导结果；边界偏移会产生大量错误 |
| Range-based precision/recall | 覆盖比例、位置和事件分裂情况 | 需要指定位置和重叠权重 |
| Affiliation F1 | 预测事件与真实事件的时间归属关系 | 需要明确空 GT 和空预测的处理 |
| Event-level F1 | 异常事件是否被检测到 | 匹配条件过宽时不反映边界精度 |
| IoU | 预测区间和 GT 区间的交并比 | 多区间时需要定义匹配算法 |
| VUS-ROC/VUS-PR | 对阈值和容忍窗口积分 | 更适合具有连续 anomaly score 的方法 |
| NAB score | 检测时间相对告警窗口的位置 | 关注在线告警，不等同于精确离线边界 |

一些工作使用 point adjustment：只要命中 GT 区间内任意一点，就把整个区间算作正确。这种处理会明显提高逐点指标，因此需要与严格区间评估区分。
