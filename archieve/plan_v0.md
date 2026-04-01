# LLM 时序异常检测项目：执行手册

## 项目定位

在公开 benchmark 上评估并微调多模态 LLM 的时序异常检测能力。侧重 **LLM 推理工程 + VLM SFT 微调** 实操经验积累。

---

## 资源约束

| 项目 | 方案 |
|------|------|
| GPU | Colab Pro ($9.99/月)：T4 16GB (免费)，L4 24GB，A100 40GB |
| API | GPT-4o：$2.50/M input, $10/M output，Batch API 半价 |
| 预算上限 | Colab Pro 1个月 + API ~$20 = **总计 ~$30** |

Colab 计费参考：T4 ~1.76 CU/hr，A100 ~15 CU/hr。Pro 套餐含 100 CU（约 6.7h A100 或 57h T4）。

---

## Phase 1：Zero-shot 评估（T4 免费即可，1周）

**目标：** 跑通 pipeline，产出与已有论文可对比的 baseline 数据。

### 1.1 获取数据和代码

```bash
# AnomLLM benchmark（UCSD Rose Lab，ICLR 2025）
git clone https://github.com/Rose-STL-Lab/AnomLLM.git
cd AnomLLM
conda env create --file environment.yml
# 数据下载见 README 中的 S3 链接

# NeurIPS 2025 改进方法
git clone https://github.com/junwoopark92/LLM-TSAD.git

# NAB 数据集（58条真实+合成时序，带标注）
git clone https://github.com/numenta/NAB.git
# 数据直接在 repo 的 data/ 目录下，无需额外下载

# TSB-AD benchmark（NeurIPS 2024，1070条时序，可选）
pip install TSB-AD
```

### 1.2 模型加载（Colab T4 16GB）

```python
# INT4 量化推理，T4 可跑
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
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
    max_pixels=512*28*28,  # 控制显存
)
```

### 1.3 时序→图片 pipeline

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def ts_to_image(values, window_size=128, save_path=None):
    """将时序片段渲染为标准化折线图"""
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.plot(values, color='steelblue', linewidth=1)
    ax.set_xlim(0, len(values))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.1)
    if save_path:
        plt.savefig(save_path)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)
```

### 1.4 推理与评估

```python
# AnomLLM 已有评估脚本，直接对接
# 见 AnomLLM/src/online_api.py 的接口格式
# 支持 OpenAI-compatible API，本地 vLLM 也兼容

# 用 AnomLLM 的评估脚本跑指标
python src/online_api.py --data trend --model qwen2.5-vl --variant 0shot-vision
python src/online_api.py --data frequency --model qwen2.5-vl --variant 0shot-text
```

### 1.5 消融实验矩阵

| 维度 | 变量 | 值 |
|------|------|----|
| 输入模态 | vision vs text | 折线图 vs 原始数值字符串 |
| Prompt 策略 | 0-shot / 2-shot / CoT / 统计分解 | 参考 LLM-TSAD 的 index-aware prompting |
| 窗口大小 | 64 / 128 / 256 | — |
| 模型 | Qwen2.5-VL-7B / 3B | — |

**评估指标直接复用：** AnomLLM 的 `evaluation/` 脚本输出 F1、Precision、Recall，无需自己实现。

---

## Phase 2：SFT 蒸馏（核心，A100 ~3-5h，2周）

**目标：** 用 GPT-4o 生成训练数据，QLoRA 微调 Qwen2.5-VL-7B，获得完整的 VLM SFT 经验。

### 2.1 用 GPT-4o Batch API 生成训练数据

```python
# 成本估算：1000张图 × ~500 tokens/图 ≈ 0.5M tokens
# input: $2.50/M × 0.5M = $1.25
# output: $10/M × 0.2M = $2.00
# Batch API 半价 → 总计约 $1.6

import openai, base64, json

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
                    "Step 2: Identify any anomalous regions and classify them (spike/dip/level_shift/trend_change). "
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

**数据量目标：** 1000-2000 条（AnomLLM 合成数据 + NAB 子集）。

### 2.2 QLoRA 微调（Unsloth，Colab T4/A100）

Unsloth 官方提供 Qwen2.5-VL-7B Vision 的 Colab notebook：
- **SFT notebook**: https://unsloth.ai/docs/get-started/unsloth-notebooks （找 "Qwen2.5-VL (7B)" 行）
- **GRPO notebook**: 同页面，找 "Qwen2.5-VL - Vision GSPO"

```python
# === Colab 中执行 ===
!pip install unsloth
!pip install --force-reinstall --no-deps trl

from unsloth import FastVisionModel
import torch

# 加载 4-bit 量化模型（T4 16GB 可用）
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# 添加 LoRA adapter
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                     "gate_proj","up_proj","down_proj"],
    finetune_vision_layers=False,   # 冻结视觉层，省显存
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)
```

### 2.3 训练数据格式（LLaVA 格式）

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
        "content": "Step 1: The series shows a stable upward trend with low noise...\nStep 2: Around index 80-95, there is a sharp level shift...\nStep 3: ANOMALY (type: level_shift)"
      }
    ]
  }
]
```

### 2.4 训练执行

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.trainer import UnslothVisionDataCollator

training_args = TrainingArguments(
    output_dir="./qwen25vl-tsad-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # 有效 batch=8
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
# 预计 T4: ~4-6h / 1500条; A100: ~1.5-2h / 1500条
```

### 2.5 合并权重 & 评估

```python
# 合并 LoRA
model.save_pretrained_merged("qwen25vl-tsad-merged", tokenizer)

# 或保存 adapter 单独加载
model.save_pretrained("qwen25vl-tsad-adapter")

# 用 Phase 1 相同的评估脚本对比 SFT 前后效果
```

---

## Phase 3：DPO 对齐（可选加分项，A100 ~2-3h，1周）

**目标：** 体验偏好对齐全流程，展示 Prompting→SFT→Alignment 三阶段能力。

### 3.1 构造偏好数据（无需人工标注）

```python
# 用 SFT 模型对每条测试数据采样 N=5 个回答
# 按 F1 与 ground truth 的一致性自动排序
# chosen = F1最高的回答, rejected = F1最低的回答

def build_dpo_pair(image_path, ground_truth, model, n_samples=5):
    responses = []
    for _ in range(n_samples):
        resp = model.generate(image_path, temperature=0.7)
        f1 = compute_f1(parse_prediction(resp), ground_truth)
        responses.append({"response": resp, "f1": f1})
    
    responses.sort(key=lambda x: x["f1"], reverse=True)
    return {
        "image": image_path,
        "prompt": "Analyze this time series for anomalies.",
        "chosen": responses[0]["response"],
        "rejected": responses[-1]["response"],
    }
```

### 3.2 DPO 训练

**方案 A：Unsloth GSPO（推荐，Colab 友好）**

Unsloth 官方有 Qwen2.5-VL Vision GSPO notebook，直接改数据即可。

**方案 B：2U1/Qwen-VL-Series-Finetune**

```bash
git clone https://github.com/2U1/Qwen-VL-Series-Finetune.git
# 支持 SFT / DPO / GRPO，文档完整
# DPO 数据格式见 README 的 DPO 章节
```

---

## 评估体系（直接复用，零开发成本）

| 工具 | 用途 | 来源 |
|------|------|------|
| AnomLLM evaluation scripts | F1/Precision/Recall on synthetic data | `Rose-STL-Lab/AnomLLM/evaluation/` |
| LLM-TSAD evaluation | 支持 AnomLLM + TSB-AD benchmark | `junwoopark92/LLM-TSAD/src/result_agg_by_model.py` |
| NAB scoring | NAB 专用评分（含 early detection reward） | `numenta/NAB/run.py` |
| TSB-AD metrics | VUS-PR（NeurIPS 2024 推荐指标） | `pip install TSB-AD` 内置 |

**关键：** 不要自己写评估代码。上述四个工具覆盖所有场景，直接调用。

---

## 开源资源速查

### 数据集

| 名称 | 链接 | 备注 |
|------|------|------|
| AnomLLM Synthetic | https://github.com/Rose-STL-Lab/AnomLLM | ICLR 2025，4类异常 |
| NAB | https://github.com/numenta/NAB | 58条时序，data/ 目录直接可用 |
| TSB-AD | https://github.com/TheDatumOrg/TSB-AD | 1070条，pip install 可用 |

### 模型

| 名称 | HuggingFace | 显存 |
|------|-------------|------|
| Qwen2.5-VL-7B (4-bit, Unsloth) | `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit` | ~6GB，T4 可跑 |
| Qwen2.5-VL-7B (FP16) | `Qwen/Qwen2.5-VL-7B-Instruct` | ~17GB，L4/A100 |
| Qwen2.5-VL-3B (4-bit) | `unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit` | ~3GB，T4 可跑 |

### 微调工具

| 工具 | 链接 | 适用场景 |
|------|------|----------|
| **Unsloth** (首选) | https://github.com/unslothai/unsloth | Colab 友好，官方 Qwen2.5-VL notebook |
| 2U1/Qwen-VL-Series-Finetune | https://github.com/2U1/Qwen-VL-Series-Finetune | SFT+DPO+GRPO 全支持 |
| HuggingFace TRL | https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl | 官方教程 |

### 关键论文代码

| 论文 | 代码 | 用途 |
|------|------|------|
| Can LLMs Understand TS Anomalies? (ICLR 2025) | https://github.com/Rose-STL-Lab/AnomLLM | 数据+评估+baseline |
| Delving into LLMs for TSAD (NeurIPS 2025) | https://github.com/junwoopark92/LLM-TSAD | 改进 prompt 方法 |
| SigLLM (MIT, DSAA 2024) | https://github.com/sintel-dev/sigllm | Pipeline 参考 |

---

## 时间线

| 周 | 任务 | 资源 | 产出 |
|----|------|------|------|
| 1 | Phase 1: 跑通 AnomLLM benchmark，完成 zero-shot 评估 | T4 免费 | baseline 数据表 |
| 2 | Phase 1: 消融实验（prompt/模态/窗口） | T4 免费 | 消融对比表 |
| 3 | Phase 2: GPT-4o 生成 SFT 数据 + 数据格式化 | API ~$2-3 | 1500条训练数据 |
| 4 | Phase 2: QLoRA 微调 + 评估 | A100 ~3h | SFT 模型 + 对比表 |
| 5 | Phase 3: 构造 DPO 数据 + 训练 + 评估 | A100 ~2h | DPO 模型（可选） |

**总成本：** Colab Pro $9.99 + API ~$5 = **~$15**

---

## 简历 Bullet Points 模板

> - 复现 ICLR 2025 / NeurIPS 2025 方法，在 AnomLLM 和 NAB benchmark 上系统评估 Qwen2.5-VL 的零样本时序异常检测能力，覆盖视觉/文本双模态输入与多种 prompting 策略。
> - 基于 GPT-4o 蒸馏构建含 CoT 推理链的 SFT 训练集（1500+ 条），通过 QLoRA 对 Qwen2.5-VL-7B 实施参数高效微调，F1 由 X 提升至 Y。
> - 采用 DPO 对齐优化模型检测偏好，实现 precision/recall 权衡的可控调节，展示偏好学习在异常检测场景的实用价值。
