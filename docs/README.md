# 技术文档索引

本目录保存了 `tsad_sft_pipeline.ipynb` 所用核心技术的官方文档（离线查阅用）。

## 文档列表

| 文件 | 说明 | 来源 |
|------|------|------|
| [unsloth_vision_sft.md](unsloth_vision_sft.md) | Unsloth Vision Fine-tuning: FastVisionModel API, LoRA 配置, 数据格式, DataCollator | [docs.unsloth.ai](https://docs.unsloth.ai/basics/vision-fine-tuning) |
| [unsloth_vision_grpo_notebook.md](unsloth_vision_grpo_notebook.md) | Unsloth Qwen3-VL Vision GRPO 官方 notebook 完整代码 | [GitHub unslothai/notebooks](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb) |
| [trl_sft_trainer.md](trl_sft_trainer.md) | TRL SFTTrainer / SFTConfig: 参数说明, 数据格式, VLM 训练, PEFT 集成 | [HuggingFace TRL](https://huggingface.co/docs/trl/sft_trainer) |
| [trl_grpo_trainer.md](trl_grpo_trainer.md) | TRL GRPOTrainer / GRPOConfig: GRPO 算法, 自定义 reward 函数, loss types | [HuggingFace TRL](https://huggingface.co/docs/trl/grpo_trainer) |

## 技术栈概览

```
Unsloth (FastVisionModel)     -- 模型加载 + LoRA + 推理加速
  ├── SFT: trl.SFTTrainer     -- 监督微调
  │   └── UnslothVisionDataCollator -- 视觉数据整理
  └── GRPO: trl.GRPOTrainer   -- 强化学习 (Group Relative Policy Optimization)
      └── Custom Reward Funcs  -- 格式奖励 + 正确性奖励

HuggingFace datasets           -- 数据加载
HuggingFace transformers       -- tokenizer / generation
```

## 官方 Colab Notebooks

- **Vision SFT (Qwen3-VL 8B)**: [Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision.ipynb)
- **Vision GRPO (Qwen3-VL 8B)**: [Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb)
