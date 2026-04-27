# TRL SFTTrainer / SFTConfig 官方文档

> 来源: https://huggingface.co/docs/trl/sft_trainer

## Overview

TRL 的 Supervised Fine-Tuning (SFT) Trainer 用于训练语言模型。`SFTTrainer` 是 HuggingFace `Trainer` 的封装，继承其所有属性和方法。

## Quick Start

```python
from trl import SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

## 数据格式

SFT 支持 **language modeling** 和 **prompt-completion** 两种数据集格式，每种又分 standard 和 conversational：

```python
# Standard language modeling
{"text": "The sky is blue."}

# Conversational language modeling
{"messages": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is blue."}]}

# Standard prompt-completion
{"prompt": "The sky is",
 "completion": " blue."}

# Conversational prompt-completion
{"prompt": [{"role": "user", "content": "What color is the sky?"}],
 "completion": [{"role": "assistant", "content": "It is blue."}]}
```

## SFTTrainer 主要参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `str` / `PreTrainedModel` / `PeftModel` | 模型 ID 或已加载的模型对象 |
| `args` | `SFTConfig` | 训练配置 |
| `data_collator` | `DataCollator` | 数据整理函数。VLM 默认用 `DataCollatorForVisionLanguageModeling` |
| `train_dataset` | `Dataset` / `IterableDataset` | 训练数据集 |
| `eval_dataset` | `Dataset` / `dict` | 验证数据集 |
| `processing_class` | `PreTrainedTokenizerBase` / `ProcessorMixin` | tokenizer 或 processor |
| `peft_config` | `PeftConfig` | PEFT/LoRA 配置 |
| `formatting_func` | `Callable` | 格式化函数，将数据集转为 language modeling 格式 |

## SFTConfig 关键参数

SFTConfig 继承自 `TrainingArguments`，以下是 SFT 特有或默认值不同的参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `logging_steps` | `10` | 日志记录间隔（TrainingArguments 默认 500） |
| `gradient_checkpointing` | `True` | 梯度检查点（TrainingArguments 默认 False） |
| `bf16` | `True` | BF16 精度（TrainingArguments 默认 False） |
| `learning_rate` | `2e-5` | 学习率（TrainingArguments 默认 5e-5） |
| `max_length` | - | 最大序列长度。**VLM 训练建议设为 None** |
| `packing` | `False` | 是否启用 example packing |
| `assistant_only_loss` | `False` | 是否只在 assistant 回复上计算 loss |
| `completion_only_loss` | `True` | prompt-completion 格式下只在 completion 上计算 loss |
| `output_dir` | - | 模型保存目录 |
| `per_device_train_batch_size` | - | 每设备 batch size |
| `gradient_accumulation_steps` | - | 梯度累积步数 |
| `num_train_epochs` | - | 训练轮数 |
| `max_steps` | - | 最大训练步数（设置后覆盖 num_train_epochs） |
| `save_strategy` | - | 保存策略: "steps" / "epoch" / "no" |
| `eval_strategy` | - | 评估策略: "steps" / "epoch" / "no" |
| `optim` | - | 优化器: "adamw_hf" / "adamw_8bit" 等 |
| `report_to` | - | 日志报告: "wandb" / "tensorboard" / "none" |
| `remove_unused_columns` | - | **VLM 训练需设为 False** |

## VLM (Vision Language Model) 训练

```python
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=SFTConfig(max_length=None),  # VLM 必须设 max_length=None 避免截断图像 token
    train_dataset=load_dataset("trl-lib/llava-instruct-mix", split="train"),
)
trainer.train()
```

> **重要**: VLM 训练中截断可能移除 image tokens 导致错误。设 `max_length=None` 让模型处理完整序列。

## PEFT/LoRA 集成

```python
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    "Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    peft_config=LoraConfig(),
)
trainer.train()
```

> **Tip**: 使用 LoRA adapter 训练时通常使用更高的学习率（约 1e-4）。

## Packing

启用 packing 将多个样本打包到同一序列中，提高训练效率：

```python
training_args = SFTConfig(packing=True)
```

## Train on Assistant Messages Only

```python
training_args = SFTConfig(assistant_only_loss=True)
```

仅在 assistant 回复上计算 loss，忽略 user/system 消息。

## Logged Metrics

| 指标 | 说明 |
|------|------|
| `loss` | 非 mask token 上的平均交叉熵 loss |
| `entropy` | 模型预测 token 分布的平均熵 |
| `mean_token_accuracy` | 模型 top-1 预测与 ground truth 匹配的 token 比例 |
| `learning_rate` | 当前学习率 |
| `grad_norm` | 梯度 L2 范数（裁剪前） |
| `num_tokens` | 已处理的总 token 数 |
