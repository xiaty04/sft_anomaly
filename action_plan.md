# 行动执行计划

每个检查点的节奏：执行对应的 cell → 停下来看输出 → 把观察结果告诉 Claude Code → 继续下一块。

---

## 检查点 0：环境确认

**这一步做了什么**
挂载 Google Drive 作为持久化存储，在本地 `/content/tsad_runtime/` 创建所有运行目录，安装 Unsloth/vLLM/TRL 等训练推理依赖。这些操作在 Colab 每次新 session 都要重跑。

**执行**：阶段 0 全部 cell。

**停下来确认**：
- Drive 挂载成功，`/content/drive/MyDrive/tsad_anomaly/` 目录存在
- `import unsloth` 无报错
- `torch.cuda.get_device_name(0)` 显示 A100
- `torch.cuda.is_bf16_supported()` 返回 True

**告诉 Claude Code**：GPU 型号、可用 VRAM（GB）、BF16 是否支持。

---

## 检查点 1：数据结构确认

**这一步做了什么**
克隆 AnomLLM 仓库，从 S3 下载（或本地生成）合成时序数据，备份到 Drive。数据核心是每个子集一个 `data.pkl`，包含 400 条时序（shape `(1000,1)`）和对应的异常区间标注。`figs/` 目录存放已预渲染的折线图，后续 SFT 直接用这些图，不需要重新渲染。

**执行**：阶段 1 全部 cell，重点看 Cell 1.3 输出。

**停下来看**：

```
flat-trend series: 400 (1000, 1) | has_figs: True
range      series: 400 (1000, 1) | has_figs: True
point      series: 400 (1000, 1) | has_figs: True
freq       series: 400 (1000, 1) | has_figs: True
```

还要手动打开一张 PNG（如 `data/synthetic/point/eval/figs/001.png`）目视确认图像正常。

**如果异常**：
- `has_figs=False`：S3 未包含预渲染图，后续阶段 5 需要加渲染步骤
- series 数量不是 400 或 shape 不对：告诉 Claude Code 实际值

**告诉 Claude Code**：四个子集的 series 数量和 shape，has_figs 是否全为 True。

---

## 检查点 2：VLM Baseline 单次冒烟

**这一步做了什么**
用 vLLM 在本地起一个 OpenAI 兼容的推理服务，把 Qwen3-VL-8B 注册为 `qwen-local`（名字必须与 `credentials.yml` 里的 key 一致，`online_api.py` 会直接把这个名字发给 API）。然后只对 `point` 子集的 400 条时序跑一次 zero-shot 视觉推理，每条输出异常区间列表 `[{"start":...,"end":...},...]`，存入 jsonl。

先只跑 `point` 一个子集，确认链路通了再跑全量。

**执行**：Cell 2.1 启动 server → Cell 2.2 写 credentials → 只跑 point：

```bash
cd /content/tsad_runtime/code/AnomLLM
python src/online_api.py --data point --model qwen-local --variant 0shot-vision
```

**停下来看**：

```bash
head -3 results/synthetic/point/qwen-local/0shot-vision.jsonl
```

预期 response 字段是 `[{"start":...,"end":...},...]` 格式，非空字符串。

**如果出错**：
- `model not found`：Cell 2.1 的 `--served-model-name qwen-local` 是否加了
- response 全是 `[]`（空列表）：可能是图像未正确传入，把前 3 条 response 原文告诉 Claude Code

**告诉 Claude Code**：前 3 条 response 内容、单条推理耗时（秒）。

---

## 检查点 3：VLM Baseline 全量 + 统计 Baseline

**这一步做了什么**
用 VLM 对全部 4 个子集各 400 条跑完 zero-shot 推理。再用 isolation forest 跑传统 baseline——它直接在时序数值上跑异常检测，不需要图像，速度很快。两种方法的结果都存成 jsonl，格式相同，方便后续统一评估。

**执行**：Cell 2.3 全量 VLM + Cell 3.1 isolation forest。

**停下来看**：

```bash
wc -l results/synthetic/*/qwen-local/0shot-vision.jsonl
wc -l results/synthetic/*/isolation-forest/0shot.jsonl
```

预期每个文件都是 400 行。

**告诉 Claude Code**：各文件实际行数（不足 400 说明推理中途中断，需要续跑）。

---

## 检查点 4：Baseline 指标汇总

**这一步做了什么**
`result_agg.py` 读取 `results/synthetic/<subset>/` 下所有 jsonl，把预测的区间列表转成 0/1 向量，与 ground truth 对比，输出 precision/recall/F1 和 affiliation F1（对异常边界位置更宽容的指标）。结果按 model×variant 组织成 DataFrame，保存为 pkl，再手动合并成一张对比 CSV。

这张表是后续 SFT 评估的**基准线**，必须先有它才能知道 SFT 是否有提升。

**执行**：阶段 4 全部 cell。

**停下来看**：打印 `baseline_compare.csv`，关注：
- `qwen-local (0shot-vision)` 的 `f1` 在各子集是否非零（预期 0.2~0.7）
- `isolation-forest (0shot)` 的 `f1` 是否也有数字

**如果 qwen-local 的 f1 全是 0**：response 解析失败，把 `baseline_compare.csv` 头几行告诉 Claude Code。

**告诉 Claude Code**：`baseline_compare.csv` 全文。

---

## 检查点 5：SFT 样本清单

**这一步做了什么**
从 4 个子集的 eval `data.pkl` 中读出全部 1600 条样本（400×4），建立索引清单，记录每条的图像路径、label、异常类型和真实区间。然后做分层抽样，切分为：
- train 1200 条（用于蒸馏 + SFT 训练）
- val 150 条（用于 SFT 训练时的验证集）
- eval 150 条（固定不参与训练，用于最终对比）

eval split 按 subset×label 分层，保证每种异常类型都有代表。

**执行**：Cell 5.1。

**停下来看** print 输出，预期：

```
split   count
eval    150
train   1200
val     150

subset×label 分布各自均匀
```

**如果 has_figs=False**（检查点 1 发现的）：告诉 Claude Code，需要在这步之后加一个渲染 cell。

**告诉 Claude Code**：split/label 分布表，以及各子集 label=1 的数量。

---

## 检查点 6：教师蒸馏小批冒烟

**这一步做了什么**
用 DashScope 的 Qwen3.5-plus 作为教师模型，把时序图像 base64 编码后发给它，让它输出结构化的标注 JSON：是否异常、异常类型、异常区间列表（多段）、简短 rationale。这些标注将成为 SFT 的训练目标。

**先只跑 20 条验证质量，再跑全量。**

**执行**：Cell 6.2 加 `.head(20)` 限制，跑完后看输出质量，确认没问题再去掉限制跑全量，最后跑 Cell 6.3 清洗。

**停下来看** `sft_raw.jsonl` 前 5 条：

```python
import json
with open("/content/tsad_runtime/sft/sft_raw.jsonl") as f:
    for _ in range(5):
        print(json.dumps(json.loads(f.readline()), indent=2, ensure_ascii=False))
```

检查：
- `intervals` 字段是否是列表（哪怕是空列表 `[]`）
- 教师判断的 `is_anomaly` 与 `ground_truth` 一致率是否 ≥ 70%
- `rationale` 是否有实质内容

全量跑完后看清洗结果：

```
clean=NNN, reject=NNN, keep_rate=XX%
```

**如果一致率 < 50%**：prompt 问题，把前 5 条原始 JSON 告诉 Claude Code。

**告诉 Claude Code**：前 5 条 JSON、clean/reject 数量、保留率。

---

## 检查点 7：训练格式验证

**这一步做了什么**
把蒸馏数据转成 Unsloth 的 messages 格式：user turn 包含图像路径和问题文本，assistant turn 包含结构化的标签字符串（`<label>`, `<type>`, `<region>`, `<rationale>` 标签）。eval split 单独转成推理用格式（只有图像路径和 ground truth，不含 assistant 答案）。

**执行**：阶段 7 全部 cell。

**停下来看** `train.jsonl` 第一条：

```python
import json, os
with open("/content/tsad_runtime/sft/train.jsonl") as f:
    r = json.loads(f.readline())
print(json.dumps(r, indent=2, ensure_ascii=False))
print("image exists:", os.path.exists(r["messages"][0]["content"][0]["image"]))
```

确认：
- messages 结构包含 user（image+text）和 assistant
- image 路径文件实际存在
- assistant content 包含 `<label>ANOMALY</label>` 或 `<label>NORMAL</label>`

**告诉 Claude Code**：第一条 JSON 内容，以及 image exists 结果。

---

## 检查点 8：SFT 训练冒烟

**这一步做了什么**
加载 4bit 量化的 Qwen3-VL-8B，挂上 LoRA 适配器（r=16，覆盖视觉层和语言层），用 Unsloth 的 `UnslothVisionDataCollator` 处理图文数据，SFTTrainer 执行监督微调。`max_steps=5` 冒烟是为了在花 1~2 小时跑完整训练之前，先确认不 OOM、loss 格式正常。

**执行**：Cell 8.1 加载模型 → Cell 8.2 临时加 `max_steps=5` 跑冒烟。

**停下来看**：
- 是否有 CUDA OOM 报错
- loss 第 1 步在 1.5~4 之间（不应是 nan 或 0）

**如果 OOM**：告诉 Claude Code GPU VRAM 大小，调整 `per_device_train_batch_size=1`。

**确认没问题后**：去掉 `max_steps=5`，跑完整训练（Cell 8.2 正式版 → Cell 8.3 导出）。

**告诉 Claude Code**：冒烟的前 5 步 loss 输出，是否有报错。

---

## 检查点 9：SFT 评估与公平对比

**这一步做了什么**
用与 baseline 完全相同的推理方式（vLLM server + online_api.py），对全部 4 个子集跑 SFT 模型的 zero-shot 推理，结果存入 jsonl。然后从 baseline 和 SFT 的 jsonl 中**各自过滤出 eval 150 条对应的样本**（按 pkl_idx 映射），在相同样本上计算指标，确保对比公平。

**执行**：Cell 9.1 全部（启动 SFT server + 推理）→ Cell 9.2 计算指标。

**停下来看** 打印的 summary 表，对比三种方法在 eval 150 条上的平均 F1：

| 方法 | avg F1 | avg affi F1 |
|------|--------|-------------|
| isolation-forest | ? | ? |
| qwen-local 0shot-vision | ? | ? |
| sft-0shot | ? | ? |

**关键问题**：SFT 是否比 VLM zero-shot 有提升（F1 高 ≥ 0.03）？

**告诉 Claude Code**：这张对比表，以及是否考虑进入 GRPO 阶段。

---

## 检查点 10（可选）：GRPO

**这一步做了什么**
GRPO 是强化学习微调：不再用教师标注作为监督信号，而是让模型自己输出多个答案，按奖励函数（比如 label 是否正确、region 是否与 ground truth 重叠）给分，用分数差异做梯度更新。适合在 SFT 基础上进一步优化模型推理质量。

**进入条件**（全部满足才做）：
1. SFT 在 eval 上比 VLM zero-shot 有明确提升（F1 高 ≥ 0.05）
2. 蒸馏保留率 ≥ 70%
3. 还有剩余 Colab session 时间

**告诉 Claude Code**：满足几个条件，以及希望用什么奖励函数（label 正确/region 精确/两者结合）。
