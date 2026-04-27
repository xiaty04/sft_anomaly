# TRL GRPOTrainer / GRPOConfig 官方文档

> 来源: https://huggingface.co/docs/trl/grpo_trainer

## Overview

GRPO (Group Relative Policy Optimization) 是 PPO 的变体，来自论文 [DeepSeekMath](https://huggingface.co/papers/2402.03300)。核心思想：对每个 prompt 生成一组 completions，用组内相对奖励作为 advantage，无需额外的 value model。

## Quick Start

```python
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

## GRPO 算法四步骤

### 1. Generating completions
每个训练步，采样一批 prompts，每个 prompt 生成 G 个 completions。

### 2. Computing the advantage
对每组 G 个 completion 计算 reward，然后归一化：

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

> `scale_rewards=False` 可禁用 std 缩放（避免 question-level difficulty bias）。
> `scale_rewards="batch"` 可在 batch 级别计算 std（更鲁棒）。

### 3. Estimating the KL divergence
默认 `beta=0.0`（不使用 KL 项）。设 `beta > 0` 启用 KL 约束。

### 4. Computing the loss
使用 clipped surrogate objective，类似 PPO。支持多种 loss_type。

## GRPOConfig 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_generations` | `8` | 每个 prompt 生成的 completion 数 G |
| `max_prompt_length` | `512` | prompt 最大 token 长度 |
| `max_completion_length` | `256` | completion 最大 token 长度 |
| `beta` | `0.0` | KL 散度系数（0 = 不使用 KL） |
| `scale_rewards` | `True` | 是否用 std 缩放 reward |
| `loss_type` | `"grpo"` | Loss 类型（见下方） |
| `num_iterations` | `1` | 每次生成后的更新次数 (mu) |
| `epsilon_low` | `0.2` | clip 下界 |
| `epsilon_high` | `0.2` | clip 上界 |
| `mask_truncated_completions` | `True` | 是否 mask 被截断的 completions |
| `importance_sampling_level` | `"token"` | 重要性采样级别: "token" / "sequence" |
| `learning_rate` | - | 学习率 |
| `per_device_train_batch_size` | - | 每设备 batch size |
| `gradient_accumulation_steps` | - | 梯度累积 |
| `max_grad_norm` | - | 梯度裁剪阈值 |
| `optim` | - | 优化器 |
| `use_vllm` | `False` | 是否使用 vLLM 加速推理 |
| `reward_weights` | `None` | 多个 reward 函数的权重 |

## Loss Types

| loss_type | 说明 |
|-----------|------|
| `"grpo"` | 标准 GRPO |
| `"dr_grpo"` | Dynamic Reward GRPO |
| `"sapo"` | 非对称温度，惩罚 bad actions 更严格 |
| `"bnpo"` | Batch Normalized Policy Optimization |

## 自定义 Reward 函数

### 接口规范

Reward 函数接收以下 keyword arguments：
- `prompts`: prompt 列表
- `completions`: 生成的 completion 列表
- `completion_ids`: tokenized completion 列表
- `trainer_state`: 当前训练状态
- 数据集中的其他列名（如 `answer`, `ground_truth` 等）

返回值：`list[float]`，每个 float 对应一个 completion 的 reward。

> **建议**: 函数签名中使用 `**kwargs` 接收未用到的参数。

### 示例 1: 奖励更长的 completion

```python
def reward_func(completion_ids, **kwargs):
    """Reward function that assigns higher scores to longer completions."""
    return [float(len(ids)) for ids in completion_ids]
```

### 示例 2: 格式奖励

```python
import re

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
```

### 示例 3: 基于参考答案的正确性奖励

```python
import re

def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
```

### 使用多个 Reward 函数

```python
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        formatting_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
)
```

可通过 `reward_weights` 设置各函数权重。

## Logged Metrics

| 指标 | 说明 |
|------|------|
| `completions/mean_length` | 生成 completion 的平均长度 |
| `completions/clipped_ratio` | 被截断 completion 的比例 |
| `reward/{func_name}/mean` | 各 reward 函数的平均奖励 |
| `reward/{func_name}/std` | 各 reward 函数奖励的标准差 |
| `reward` | 加权求和后的总平均 reward |
| `frac_reward_zero_std` | reward std 为零的样本比例（多样性指标） |
| `entropy` | token 预测的平均熵 |
| `kl` | 模型与参考模型的 KL 散度（仅 beta > 0） |
| `clip_ratio/region_mean` | 被 clip 的概率比例 |
| `step_time` | 每步平均耗时（含生成） |

## vLLM 加速

### Colocate 模式（默认）

vLLM 在 trainer 进程内运行，共享 GPU 显存：

```python
training_args = GRPOConfig(
    use_vllm=True,  # vllm_mode="colocate" by default
)
```

### Server 模式

vLLM 在独立进程运行（独立 GPU），通过 HTTP 通信：

```bash
trl vllm-serve --model <model_name>
```

```python
training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="server",
)
```
