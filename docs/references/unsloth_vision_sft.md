# Unsloth Vision Fine-tuning 官方文档

> 来源: https://docs.unsloth.ai/basics/vision-fine-tuning

## Overview

Vision fine-tuning enables models to excel at specialized tasks like object detection and movement analysis. The platform offers free notebooks for several vision models including Qwen3-VL, Ministral 3, Gemma 3, Llama 3.2, and Qwen2.5 VL.

**Image Dimensions:** Use dimensions of 300-1000px to ensure your training does not take too long or use too many resources.

## Model Configuration

### 加载模型

```python
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
```

### LoRA 配置: `FastVisionModel.get_peft_model()`

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,   # 是否微调 vision encoder
    finetune_language_layers   = True,   # 是否微调 language model
    finetune_attention_modules = True,   # 是否微调 attention layers
    finetune_mlp_modules       = True,   # 是否微调 MLP layers
    r = 16,                              # LoRA rank
    lora_alpha = 16,                     # LoRA alpha
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
    target_modules = "all-linear",
    modules_to_save = ["lm_head", "embed_tokens"],
)
```

## Dataset Format

Vision fine-tuning datasets follow a structured conversation format with image inputs:

```python
[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "image": image}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": answer}
        ]
    },
]
```

### Dataset Conversion Example

```python
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["caption"]}
            ]
        },
    ]
    return {"messages": conversation}

converted_dataset = [convert_to_conversation(sample) for sample in dataset]
```

## Multi-Image Training

For multi-image datasets, use list comprehension instead of the `.map()` function:

```python
# Instead of:
# ds_converted = ds.map(convert_to_conversation)

# Use:
ds_converted = [convert_to_conversation(sample) for sample in dataset]
```

This approach avoids strict dataset standardization rules that complicate multi-image processing.

## Vision Data Collator

### UnslothVisionDataCollator 参数

```python
from unsloth.trainer import UnslothVisionDataCollator

class UnslothVisionDataCollator:
    def __init__(
        self,
        model,
        processor,
        max_seq_length = None,
        formatting_func = None,
        resize = "min",              # Options: (10, 10), "min", "max"
        ignore_index = -100,
        train_on_responses_only = False,
        instruction_part = None,
        response_part = None,
        force_match = True,
        num_proc = None,
        completion_only_loss = True,
        pad_to_multiple_of = None,
        resize_dimension = 0,
        snap_to_patch_size = False,
    )
```

### 使用示例

```python
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = dataset,
    args = SFTConfig(...),
)
```

## Inference

```python
FastVisionModel.for_inference(model)

image = dataset[0]["image"]
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                   use_cache=True, temperature=1.5, min_p=0.1)
```

## Training on Assistant Responses Only

For vision models, configure response-only training through the data collator:

```python
UnslothVisionDataCollator(
    model, tokenizer,
    train_on_responses_only=True,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

## 官方 Notebooks

- **Qwen3-VL (8B) Vision SFT**: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision.ipynb
- **Qwen3-VL (8B) Vision GRPO**: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb
- **Qwen2.5-VL (7B) Vision SFT**: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_VL_(7B)-Vision.ipynb
- **Qwen2.5-VL (7B) Vision GRPO**: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_5_7B_VL_GRPO.ipynb
