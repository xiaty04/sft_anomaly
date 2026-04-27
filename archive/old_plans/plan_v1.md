# LLM 时序异常检测项目：工作计划书

## 项目定位

在公开 benchmark 上系统评估并微调多模态 LLM 的时序异常检测能力，侧重 **LLM 推理工程 + VLM SFT 微调** 的完整实操链路。核心路径：Zero-shot 基线评估 → GPT-4o 蒸馏构建 SFT 数据 → QLoRA 微调 → 效果对比。

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

### 主模型：Qwen2.5-VL-7B-Instruct

**选用理由：**

Qwen2.5-VL-7B 是目前开源 VLM（视觉语言模型）中综合性价比最高的选项之一，具体体现在：

1. **个人项目可操作性强：** 4-bit 量化版本（Unsloth BNB）显存占用约 6GB，Colab T4（16GB）即可完整运行推理与 LoRA 微调，无需 A100。这是大多数个人项目的核心约束。
2. **视觉理解能力充分：** 7B 参数规模在时序折线图的模式识别任务上效果显著优于纯文本模型，官方支持多分辨率图像输入（`min_pixels` / `max_pixels` 可控）。
3. **微调生态成熟：** Unsloth 官方提供 Qwen2.5-VL-7B 的 SFT Colab notebook，可直接复用；HuggingFace TRL 和 2U1/Qwen-VL-Series-Finetune 均支持该模型的 SFT / DPO / GRPO 全流程。
4. **与目标 benchmark 的已有对齐：** AnomLLM 和 LLM-TSAD 的评估脚本兼容 OpenAI-compatible API，本地 vLLM 部署 Qwen2.5-VL 可无缝对接。

**备选：** Qwen2.5-VL-3B（4-bit，约 3GB）用于资源极度受限时的对比实验。

---

## 方案成熟度与相似案例

本项目所采用的核心技术路径在学术界和工程实践中均有直接先例：

**1. 时序 → 图像 → VLM 推理**

- **AnomLLM（ICLR 2025）** 系统性评估了 GPT-4V、Gemini 等 VLM 在时序折线图上的零样本异常检测能力，本项目直接在其 benchmark 上复现并延伸。
- **SigLLM（MIT，DSAA 2024）** 构建了完整的时序→图像→LLM 推理 pipeline，可作为工程实现参考。

**2. GPT-4o 蒸馏 → 小模型 SFT**

用大模型生成含 CoT 推理链的标注数据，再 QLoRA 微调轻量模型，是目前最成熟的数据高效微调范式之一。HuggingFace 官方 VLM 微调教程（TRL）、Unsloth 的多个公开案例均采用此路径，在医学图像、文档理解等视觉任务中已有大量复现记录。

**3. QLoRA + Unsloth 在 Qwen2.5-VL 上的可行性**

Unsloth 官方已发布 Qwen2.5-VL-7B Vision SFT notebook，社区已有多个在 T4/L4 上成功完成微调的公开案例，方案可行性经过验证。

**4. LLM-TSAD（NeurIPS 2025）**

提出了 index-aware prompting 和统计分解等改进 prompt 策略，本项目在消融实验中复现这些策略，可直接与其报告数值对比。

**已知局限：**

- 时序异常检测对图像分辨率敏感，窗口大小与 `max_pixels` 设置需经实验校准，无通用最优值。
- GPT-4o 蒸馏数据的质量依赖 prompt 设计，需人工抽检以确保标注一致性。

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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,  # 需要 bitsandbytes
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels=256*28*28,
    max_pixels=512*28*28,
)
```

### 评估执行

直接调用 AnomLLM 评估脚本，不自行实现指标计算：

```bash
# Vision 模态（折线图输入）
python src/online_api.py --data trend --model qwen2.5-vl --variant 0shot-vision
python src/online_api.py --data frequency --model qwen2.5-vl --variant 0shot-vision

# Text 模态（原始数值字符串输入）
python src/online_api.py --data trend --model qwen2.5-vl --variant 0shot-text
```

### 消融实验矩阵

| 维度 | 变量 |
|------|------|
| 输入模态 | vision（折线图）vs. text（原始数值） |
| Prompt 策略 | 0-shot / 2-shot / CoT / 统计分解（参考 LLM-TSAD index-aware prompting） |
| 窗口大小 | 64 / 128 / 256 |
| 模型规模 | Qwen2.5-VL-7B vs. 3B |

**输出：** F1、Precision、Recall 对比表，直接对齐论文 Table。

---

## Phase 2：SFT 微调

### 目标

用 GPT-4o 生成含 CoT 推理链的标注数据，QLoRA 微调 Qwen2.5-VL-7B，量化 SFT 前后 F1 的提升幅度。

### 2.1 GPT-4o Batch API 生成 SFT 数据

数据来源：AnomLLM 合成数据 + NAB 子集，目标 1500 条。

```python
import openai, base64

def generate_sft_sample(image_path, ground_truth_label):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
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

### 2.2 训练数据格式（LLaVA 格式）

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

### 2.3 QLoRA 微调（Unsloth）

```python
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
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
    output_dir="./qwen25vl-tsad-lora",
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

Unsloth 官方 notebook：https://unsloth.ai/docs/get-started/unsloth-notebooks（找 "Qwen2.5-VL (7B)" 行）

### 2.4 权重保存与评估

```python
# 保存合并权重
model.save_pretrained_merged("qwen25vl-tsad-merged", tokenizer)

# 或仅保存 LoRA adapter
model.save_pretrained("qwen25vl-tsad-adapter")
```

```bash
# 用 Phase 1 相同评估脚本对比 SFT 前后
python src/online_api.py --data trend --model qwen25vl-tsad-merged --variant sft-vision
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

| 名称 | HuggingFace | 显存 |
|------|-------------|------|
| Qwen2.5-VL-7B (4-bit, Unsloth) | `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit` | ~6GB，T4 可跑 |
| Qwen2.5-VL-7B (FP16) | `Qwen/Qwen2.5-VL-7B-Instruct` | ~17GB，L4/A100 |
| Qwen2.5-VL-3B (4-bit) | `unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit` | ~3GB，T4 可跑 |

### 微调工具

| 工具 | 链接 | 适用场景 |
|------|------|----------|
| **Unsloth**（首选） | https://github.com/unslothai/unsloth | Colab 友好，官方 Qwen2.5-VL notebook |
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

> - 在 AnomLLM（ICLR 2025）和 NAB benchmark 上系统评估 Qwen2.5-VL 的零样本时序异常检测能力，覆盖视觉/文本双模态输入与 4 种 prompting 策略（0-shot / 2-shot / CoT / 统计分解），复现并对齐 ICLR 2025 / NeurIPS 2025 论文数值。
>
> - 基于 GPT-4o Batch API 构建含逐步 CoT 推理链的 SFT 训练集（1500+ 条图文对），通过 QLoRA（Unsloth）对 Qwen2.5-VL-7B 实施参数高效微调，在 AnomLLM benchmark 上 F1 由 X 提升至 Y，完整实现 GPT-4o 蒸馏 → VLM 微调全链路。
