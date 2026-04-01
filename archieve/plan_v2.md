# LLM 时序异常检测项目：工作计划书

## 项目定位

在公开 benchmark 上系统评估并微调多模态 LLM 的时序异常检测能力，侧重 **LLM 推理工程 + VLM 参数高效微调** 的完整实操链路。核心路径：Zero-shot 基线评估 → 构建 SFT 数据（路径 A：Qwen3.5-Plus API 蒸馏；路径 B：GRPO 强化学习）→ QLoRA 微调 → 效果对比。

---

## 数据集

### AnomLLM Synthetic Dataset

- **来源：** UCSD Rose Lab，ICLR 2025 论文《Can LLMs Understand Time Series Anomalies?》附属数据集
- **规模与形状：** 包含 4 类异常模式（trend / frequency / pattern / contextual），每类数千条单变量时间序列，序列长度 512–2048 点，附带逐点二值标签
- **获取方式：** `git clone https://github.com/Rose-STL-Lab/AnomLLM.git`，数据通过 README 中 S3 链接下载
- **用途：** Phase 1 zero-shot 评估的主要 benchmark；Phase 2 SFT 数据来源之一

### NAB（Numenta Anomaly Benchmark）

- **来源：** Numenta，业界广泛使用的真实 + 合成混合数据集
- **规模与形状：** 58 条单变量时间序列，序列长度从数百至数万点不等，覆盖服务器指标、交通流量、IoT 传感器等领域，附带逐点标签与 3 套标注方案（standard / reward / null）
- **获取方式：** `git clone https://github.com/numenta/NAB.git`，数据直接位于 `data/` 目录，无需额外下载
- **用途：** 作为真实世界数据集对 AnomLLM 合成数据进行补充评估；Phase 2 SFT 数据的第二来源

### TSB-AD（可选扩展）

- **来源：** NeurIPS 2024，The Datum Org
- **规模与形状：** 1070 条时间序列，涵盖多个公开来源，推荐评估指标为 VUS-PR（Volume Under the Surface for Precision-Recall）
- **获取方式：** `pip install TSB-AD`，内置评估脚本
- **用途：** 如需更大规模验证，可在 Phase 1 消融实验中追加；不作为 SFT 数据来源

---

## 模型选型

### 主模型：Qwen3-VL-8B-Instruct（SFT 路径）/ Qwen3-VL-8B-Thinking（GRPO 路径）

**选用理由：**

Qwen3-VL-8B 是目前 Unsloth 完整支持的最新一代开源 VLM，替代上上代的 Qwen2.5-VL-7B：

1. **T4 可行性不变：** Unsloth 官方已发布 Qwen3-VL-8B 的 4-bit 量化权重及免费 Colab notebook，QLoRA 显存占用与 Qwen2.5-VL-7B 相当，T4（16GB）可完整运行推理与微调。
2. **视觉理解能力更强：** 相比 Qwen2.5-VL-7B，Qwen3-VL-8B 在视觉推理、长上下文（256K）、空间感知和视频理解上全面升级，GUI 交互能力达到新一代水平。
3. **GRPO 路径额外优势：** Thinking 变体内置混合思考模式，冷启动阶段具备基础推理链生成能力，可降低 GRPO warm-up 所需的 SFT 数据量。
4. **Unsloth 原生支持 GRPO：** Unsloth 已发布 Qwen3-VL Vision GSPO/GRPO notebook，SFT 和 RL 两条路径均有官方 notebook 可直接复用。

**备选：** Qwen3-VL-4B（4-bit，T4 可跑）用于资源极度受限时的消融对比实验。

### 教师模型：Qwen3.5-Plus（百炼 API）

用于路径 A 的 SFT 数据生成。Qwen3.5-Plus 是百炼当前推荐的多模态旗舰，多模态能力显著优于 Qwen3-VL 系列，且与微调目标同属 Qwen 生态，生成的推理风格天然对齐。API 调用方式与 OpenAI 兼容，模型字符串为 `qwen3.5-plus`。

---

## 方案成熟度与相似案例

本项目所采用的核心技术路径在学术界和工程实践中均有直接先例：

**1. 时序 → 图像 → VLM 推理**

- **AnomLLM（ICLR 2025）** 系统性评估了 GPT-4V、Gemini 等 VLM 在时序折线图上的零样本异常检测能力，本项目直接在其 benchmark 上复现并延伸。
- **SigLLM（MIT，DSAA 2024）** 构建了完整的时序→图像→LLM 推理 pipeline，可作为工程实现参考。

**2. 教师模型蒸馏 → 小模型 SFT（路径 A）**

用大模型生成含 CoT 推理链的标注数据，再 QLoRA 微调轻量模型，是目前最成熟的数据高效微调范式之一。本项目选用 Qwen3.5-Plus（百炼 API）作为教师模型：多模态能力显著优于 Qwen3-VL 系列，与微调目标 Qwen3-VL-8B 同属 Qwen 生态，生成推理风格天然对齐，分布迁移问题最小。Unsloth 官方已发布 Qwen3-VL-8B SFT Colab notebook，可直接复用。

**2-B. GRPO 强化学习直接优化推理能力（路径 B）**

TSAD 任务存在天然可验证的 reward 信号（逐点二值标签），无需任何外部教师模型即可构建 verifiable reward 函数。模型通过 trial-and-error 直接学习推理策略，完全规避标注数据成本。TRL `GRPOTrainer` 和 `2U1/Qwen-VL-Series-Finetune`（方案已列出）均原生支持 Qwen2.5-VL 的 GRPO 全流程。路径 B 在训练稳定性上比路径 A 要求更高，建议以少量路径 A 的蒸馏数据做 warm-up SFT 再切换 GRPO。

**3. QLoRA + Unsloth 在 Qwen3-VL 上的可行性**

Unsloth 官方已发布 Qwen3-VL-8B 的 SFT 和 Vision GRPO Colab notebook，社区已有多个在 T4/L4 上成功完成微调的公开案例，方案可行性经过验证。注意：Qwen3.5 系列（原生多模态）不推荐 QLoRA 4-bit 训练，本项目选用 Qwen3-VL-8B 规避此限制。

**4. LLM-TSAD（NeurIPS 2025）**

提出了 index-aware prompting 和统计分解等改进 prompt 策略，本项目在消融实验中复现这些策略，可直接与其报告数值对比。

**已知局限：**

- 时序异常检测对图像分辨率敏感，窗口大小与 `max_pixels` 设置需经实验校准，无通用最优值。
- 路径 A：蒸馏数据质量依赖 prompt 设计，需人工抽检约 50 条以确保 72B 标注方向与 ground truth 一致。
- 路径 B：VLM 上的 GRPO 冷启动阶段可能不收敛，需先以 200 条蒸馏数据做 warm-up SFT 再切换强化学习。

---

## Phase 1：Zero-shot 基线评估

### 目标

跑通完整 pipeline，产出可与 AnomLLM / LLM-TSAD 论文直接对比的 baseline 数据表。

### 时序 → 图像渲染

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def ts_to_image(values, window_size=128):
    """将时序片段渲染为标准化折线图"""
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.plot(values, color='steelblue', linewidth=1)
    ax.set_xlim(0, len(values))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.1)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)
```

### 模型加载（T4 16GB，4-bit 量化）

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,  # 需要 bitsandbytes
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    min_pixels=256*28*28,
    max_pixels=512*28*28,
)
```

### 评估执行

直接调用 AnomLLM 评估脚本，不自行实现指标计算：

```bash
# Vision 模态（折线图输入）
python src/online_api.py --data trend --model qwen3-vl --variant 0shot-vision
python src/online_api.py --data frequency --model qwen3-vl --variant 0shot-vision

# Text 模态（原始数值字符串输入）
python src/online_api.py --data trend --model qwen3-vl --variant 0shot-text
```

### 消融实验矩阵

| 维度 | 变量 |
|------|------|
| 输入模态 | vision（折线图）vs. text（原始数值） |
| Prompt 策略 | 0-shot / 2-shot / CoT / 统计分解（参考 LLM-TSAD index-aware prompting） |
| 窗口大小 | 64 / 128 / 256 |
| 模型规模 | Qwen3-VL-8B vs. 4B |

**输出：** F1、Precision、Recall 对比表，直接对齐论文 Table。

---

## Phase 2：SFT 微调

### 目标

通过两条并行路径提升 Qwen3-VL-8B 的时序异常检测能力，量化微调前后 F1 的提升幅度，并对比两条路径的效果差异。

- **路径 A（Qwen3.5-Plus API 蒸馏）：** 用 Qwen3.5-Plus 生成含 CoT 推理链的图文标注对，QLoRA 微调 Qwen3-VL-8B-Instruct。
- **路径 B（GRPO）：** 以 AnomLLM ground truth 标签为 reward 信号，直接强化学习优化推理策略，基座使用 Qwen3-VL-8B-Thinking，无需外部标注。

---

### 路径 A：Qwen3.5-Plus API 蒸馏

#### 2A.1 生成 SFT 数据（百炼 API）

数据来源：AnomLLM 合成数据 + NAB 子集，目标 1500 条。使用阿里云百炼 OpenAI-compatible 端点，模型指定 `qwen3.5-plus`。

```python
from openai import OpenAI
import base64

client = OpenAI(
    api_key="YOUR_DASHSCOPE_API_KEY",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def generate_sft_sample(image_path, ground_truth_label):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="qwen3.5-plus",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": (
                    "Analyze this time series plot for anomalies. "
                    "Step 1: Describe the overall pattern (trend, seasonality, noise level). "
                    "Step 2: Identify any anomalous regions and classify them "
                    "(spike/dip/level_shift/trend_change). "
                    "Step 3: Give your final judgment: ANOMALY or NORMAL. "
                    "Format: {reasoning: '...', anomaly_type: '...', label: 'ANOMALY'|'NORMAL'}"
                )}
            ]
        }],
    )
    return {
        "image": image_path,
        "conversations": [
            {"role": "user", "content": "Analyze this time series plot for anomalies."},
            {"role": "assistant", "content": response.choices[0].message.content}
        ],
        "ground_truth": ground_truth_label
    }
```

生成后需人工抽检约 50 条，确认标注方向与 ground truth 一致，过滤 label 错误样本。

#### 2A.2 训练数据格式（LLaVA 格式）

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "images/trend_001.png"},
          {"type": "text", "text": "Analyze this time series for anomalies."}
        ]
      },
      {
        "role": "assistant",
        "content": "Step 1: The series shows a stable upward trend with low noise...\nStep 2: Around index 80–95, there is a sharp level shift...\nStep 3: ANOMALY (type: level_shift)"
      }
    ]
  }
]
```

#### 2A.3 QLoRA 微调（Unsloth）

```python
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    finetune_vision_layers=False,   # 冻结视觉编码器，降低显存压力
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)
```

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.trainer import UnslothVisionDataCollator

training_args = TrainingArguments(
    output_dir="./qwen3vl-tsad-sft",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
)

trainer.train()
```

Unsloth 官方 notebook：https://unsloth.ai/docs/get-started/unsloth-notebooks（找 "Qwen3-VL (8B)" 行）

---

### 路径 B：GRPO 强化学习

#### 2B.1 原理与适用性

TSAD 任务每条样本均有明确的二值 ground truth（ANOMALY / NORMAL），天然符合 GRPO 对 verifiable reward 的要求，不需要任何外部标注。模型通过对同一输入采样多个候选输出，按 reward 排序后更新策略，逐步学会在推理链中定位并描述异常。

#### 2B.2 Warm-up SFT（约 200 条）

GRPO 冷启动阶段 VLM 极易不收敛，需先用路径 A 生成的少量数据（200 条）完成一轮 SFT 热身，使模型具备基本的 Step 1/2/3 输出结构，再切换到 GRPO。Qwen3-VL-8B-Thinking 自带混合思考模式，warm-up 数据量需求比 Instruct 变体更少。

#### 2B.3 Reward 函数设计

```python
def reward_fn(response: str, ground_truth: str) -> float:
    # 格式奖励：输出包含完整 Step 1/2/3 结构
    format_score = 0.2 if all(f"Step {i}" in response for i in [1, 2, 3]) else 0.0

    # 准确性奖励：从 Step 3 后提取最终 label
    step3_text = response.split("Step 3")[-1].upper() if "Step 3" in response else ""
    if "ANOMALY" in step3_text and "NORMAL" not in step3_text:
        predicted = "ANOMALY"
    elif "NORMAL" in step3_text:
        predicted = "NORMAL"
    else:
        predicted = "UNKNOWN"

    accuracy_score = 1.0 if predicted == ground_truth else 0.0
    return format_score + accuracy_score
```

#### 2B.4 GRPO 训练（TRL GRPOTrainer）

```python
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    output_dir="./qwen3vl-tsad-grpo",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    bf16=True,
    num_generations=4,          # 每条样本采样 4 个候选输出
    max_new_tokens=512,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=0.05,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=reward_fn,
    train_dataset=grpo_train_dataset,
    processing_class=tokenizer,
)

trainer.train()
```

> 备选实现：Unsloth 官方已发布 Qwen3-VL Vision GSPO/GRPO notebook，可直接在免费 Colab T4 上运行，无需手动配置 TRL GRPOTrainer。

---

### 2C 权重保存与统一评估

```python
# 保存合并权重（路径 A 或路径 B 均适用）
model.save_pretrained_merged("qwen3vl-tsad-merged", tokenizer)

# 或仅保存 LoRA adapter
model.save_pretrained("qwen3vl-tsad-adapter")
```

```bash
# 用 Phase 1 相同评估脚本对比三组结果：zero-shot / 路径A SFT / 路径B GRPO
python src/online_api.py --data trend --model qwen3vl-tsad-merged --variant sft-distill
python src/online_api.py --data trend --model qwen3vl-tsad-grpo   --variant sft-grpo
```

---

## 评估工具（直接复用，零开发成本）

| 工具 | 用途 | 来源 |
|------|------|------|
| AnomLLM evaluation scripts | F1 / Precision / Recall on synthetic data | `Rose-STL-Lab/AnomLLM/evaluation/` |
| LLM-TSAD evaluation | 支持 AnomLLM + TSB-AD benchmark | `junwoopark92/LLM-TSAD/src/result_agg_by_model.py` |
| NAB scoring | NAB 专用评分（含 early detection reward） | `numenta/NAB/run.py` |
| TSB-AD metrics | VUS-PR（NeurIPS 2024 推荐指标） | `pip install TSB-AD` 内置 |

---

## 开源资源速查

### 数据集

| 名称 | 链接 | 备注 |
|------|------|------|
| AnomLLM Synthetic | https://github.com/Rose-STL-Lab/AnomLLM | ICLR 2025，4 类异常 |
| NAB | https://github.com/numenta/NAB | 58 条时序，data/ 目录直接可用 |
| TSB-AD | https://github.com/TheDatumOrg/TSB-AD | 1070 条，pip install 可用 |

### 模型

| 名称 | 来源 | 显存 | 用途 |
|------|------|------|------|
| Qwen3-VL-8B-Instruct (4-bit) | `unsloth/Qwen3-VL-8B-Instruct-bnb-4bit` | ~6GB，T4 可跑 | 微调目标（路径 A） |
| Qwen3-VL-8B-Thinking (4-bit) | `unsloth/Qwen3-VL-8B-Thinking-bnb-4bit` | ~6GB，T4 可跑 | 微调目标（路径 B GRPO） |
| Qwen3-VL-4B (4-bit) | `unsloth/Qwen3-VL-4B-Instruct-bnb-4bit` | ~3GB，T4 可跑 | 消融对比备选 |
| Qwen3.5-Plus（教师模型） | 阿里云百炼 API：`qwen3.5-plus` | 仅 API 调用，无需本地显存 | SFT 数据生成 |

### 微调工具

| 工具 | 链接 | 适用场景 |
|------|------|----------|
| **Unsloth**（首选） | https://github.com/unslothai/unsloth | Colab 友好，官方 Qwen3-VL (8B) SFT + GRPO notebook |
| 2U1/Qwen-VL-Series-Finetune | https://github.com/2U1/Qwen-VL-Series-Finetune | SFT + DPO + GRPO 全支持 |
| HuggingFace TRL | https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl | 官方教程 |

### 关键论文代码

| 论文 | 代码 | 用途 |
|------|------|------|
| Can LLMs Understand TS Anomalies? (ICLR 2025) | https://github.com/Rose-STL-Lab/AnomLLM | 数据 + 评估 + baseline |
| Delving into LLMs for TSAD (NeurIPS 2025) | https://github.com/junwoopark92/LLM-TSAD | 改进 prompt 方法 |
| SigLLM (MIT, DSAA 2024) | https://github.com/sintel-dev/sigllm | Pipeline 工程参考 |

---

## 简历 Bullet Points 模板

> - 在 AnomLLM（ICLR 2025）和 NAB benchmark 上系统评估 Qwen3-VL-8B 的零样本时序异常检测能力，覆盖视觉/文本双模态输入与 4 种 prompting 策略（0-shot / 2-shot / CoT / 统计分解），复现并对齐 ICLR 2025 / NeurIPS 2025 论文数值。
>
> - 基于 Qwen3.5-Plus API（百炼）构建含逐步 CoT 推理链的 SFT 训练集（1500+ 条图文对），通过 QLoRA（Unsloth）对 Qwen3-VL-8B-Instruct 实施参数高效微调，在 AnomLLM benchmark 上 F1 由 X 提升至 Y，完整实现同代旗舰模型蒸馏 → VLM 微调全链路。
>
> - 以 AnomLLM ground truth 标签设计 verifiable reward 函数，对 Qwen3-VL-8B-Thinking 实施 GRPO 强化学习微调（Unsloth Vision GRPO），在无外部标注成本条件下将 F1 由 X 提升至 Z，对比蒸馏 SFT 与 RL 两条路径在分布内/分布外异常类型上的性能差异。
