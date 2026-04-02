# Unsloth Qwen3-VL (8B) Vision GRPO 官方 Notebook

> 来源: https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb
> Colab: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb

以下是官方 notebook 的完整代码，展示了如何用 GRPO 对 Qwen3-VL 视觉语言模型做强化学习训练。

---

## 1. 安装依赖

```python
%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    import torch; v = re.match(r"[0-9]{1,}\.[0-9]{1,}", str(torch.__version__)).group(0)
    xformers = "xformers=="+("0.0.33.post1" if v=="2.9" else "0.0.32.post2" if v=="2.8" else "0.0.29.post3")
    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth
!pip install transformers==4.57.0
!pip install --no-deps trl==0.26.2
```

## 2. 加载模型

```python
from unsloth import FastVisionModel
import torch

max_seq_length = 16384
lora_rank = 16

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,
    gpu_memory_utilization=0.8,
)
```

## 3. 配置 LoRA

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,     # GRPO 中通常不微调 vision layers
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    use_gradient_checkpointing="unsloth",
)
```

## 4. 加载数据集

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("AI4Math/MathVista", split="testmini")
```

## 5. 数据预处理

### 过滤为数值型答案

```python
def is_numeric_answer(example):
    try:
        float(example["answer"])
        return True
    except:
        return False

dataset = dataset.filter(is_numeric_answer)
```

### 图片预处理

```python
def resize_images(example):
    image = example["decoded_image"]
    image = image.resize((512, 512))
    example["decoded_image"] = image
    return example
dataset = dataset.map(resize_images)

def convert_to_rgb(example):
    image = example["decoded_image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["decoded_image"] = image
    return example
dataset = dataset.map(convert_to_rgb)
```

## 6. 构造 Prompt

```python
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

def make_conversation(example):
    text_content = (
        f"{example['question']}. Also first provide your reasoning or working out"
        f" on how you would go about solving the question between {REASONING_START} and {REASONING_END}"
        f" and then your final answer between {SOLUTION_START} and (put a single float here) {SOLUTION_END}"
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_content},
            ],
        },
    ]
    return {"prompt": prompt, "image": example["decoded_image"], "answer": example["answer"]}

train_dataset = dataset.map(make_conversation)
train_dataset = train_dataset.remove_columns("image")
train_dataset = train_dataset.rename_column("decoded_image", "image")
```

## 7. 定义 Reward 函数

### 格式奖励

```python
import re

def formatting_reward_func(completions, **kwargs):
    thinking_pattern = f'{REASONING_START}(.*?){REASONING_END}'
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'
    scores = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion else ""
        score = 0
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
        if len(thinking_matches) == 1:
            score += 1.0
        if len(answer_matches) == 1:
            score += 1.0
        if len(completion) != 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                score -= 2.0
        scores.append(score)
    return scores
```

### 正确性奖励

```python
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'
    completions = [(c[0]["content"] if c else "") if isinstance(c, list) else c for c in completions]
    responses = [re.findall(answer_pattern, completion, re.DOTALL) for completion in completions]
    q = prompts[0]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:{completions[0]}")
    return [
        2.0 if len(r) == 1 and a == r[0].replace('\n', '') else 0.0
        for r, a in zip(responses, answer)
    ]
```

## 8. 推理测试（训练前）

```python
image = train_dataset[100]["image"]
prompt = train_dataset[100]["prompt"]

inputs = tokenizer(
    image,
    prompt,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1024,
                   use_cache=True, temperature=1.0, min_p=0.1)
```

## 9. 训练配置

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    log_completions=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=2,             # 每个 prompt 生成的 completion 数量 (G)
    max_prompt_length=1024,
    max_completion_length=1024,
    num_train_epochs=0.5,
    save_steps=60,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
    importance_sampling_level="sequence",
    mask_truncated_completions=False,
    loss_type="dr_grpo",           # DR-GRPO loss
)
```

## 10. 启动训练

```python
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        formatting_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

## 11. 保存模型

```python
model.save_pretrained("qwen_grpo_lora")
tokenizer.save_pretrained("qwen_grpo_lora")
```

---

## 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `finetune_vision_layers` | `False` | GRPO 中通常冻结 vision encoder，只训练 language model |
| `num_generations` | `2` | 每个 prompt 生成 G 个 completion，用于计算 group relative advantage |
| `loss_type` | `"dr_grpo"` | Dynamic Reward GRPO，改进的 loss 计算方式 |
| `max_prompt_length` | `1024` | prompt 最大 token 长度 |
| `max_completion_length` | `1024` | 生成 completion 的最大长度 |
| `importance_sampling_level` | `"sequence"` | 重要性采样在序列级别而非 token 级别 |
