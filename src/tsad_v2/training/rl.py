from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..config import save_effective_config, save_run_metadata, set_seed
from ..data.common import load_training_manifest
from ..prompts import interval_prompt
from ..rewards import format_reward_func, quality_reward_func


def _load_rl_stack():
    try:
        import torch
        from datasets import Dataset, Image
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("Install the train extra first: pip install -e '.[train]'") from exc
    return locals()


def train_rl(
    config: Dict[str, Any],
    sft_adapter: Path,
    resume_from_checkpoint: Optional[str] = None,
    limit: Optional[int] = None,
) -> Path:
    stack = _load_rl_stack()
    torch = stack["torch"]
    seed = int(config["project"].get("seed", 3407))
    set_seed(seed)
    rl_config, model_config = config["rl"], config["model"]
    records = load_training_manifest(Path(rl_config["train_manifest"]))
    if any(record["split"] != "train" for record in records):
        raise ValueError("RL manifest contains a non-train split")
    if limit is not None:
        records = records[:limit]
    output_dir = Path(rl_config["output_dir"])
    save_effective_config(config, output_dir)
    save_run_metadata(
        {
            "stage": "rl",
            "base_model": model_config["name"],
            "sft_adapter": str(sft_adapter),
            "seed": seed,
        },
        output_dir,
    )
    peft_config = stack["PeftConfig"].from_pretrained(str(sft_adapter))
    base_name = peft_config.base_model_name_or_path or model_config["name"]
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[model_config.get("dtype", "bfloat16")]
    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": dtype,
        "attn_implementation": model_config.get("attn_implementation", "sdpa"),
    }
    if bool(model_config.get("load_in_4bit", True)):
        model_kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    base_model = stack["AutoModelForImageTextToText"].from_pretrained(base_name, **model_kwargs)
    model = stack["PeftModel"].from_pretrained(base_model, str(sft_adapter), is_trainable=True)
    processor = stack["AutoProcessor"].from_pretrained(str(sft_adapter))
    dataset_rows = []
    for record in records:
        start = int(record.get("window_start", 0))
        end = int(record.get("window_end", record["length"]))
        dataset_rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": interval_prompt(start, end)},
                        ],
                    }
                ],
                "image": record["image_path"],
                "ground_truth": record.get("intervals", []),
                "window_start": start,
                "window_end": end,
            }
        )
    dataset = stack["Dataset"].from_list(dataset_rows).cast_column("image", stack["Image"]())
    arguments: Dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": float(rl_config.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(rl_config.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(rl_config.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(rl_config.get("learning_rate", 5e-6)),
        "warmup_ratio": float(rl_config.get("warmup_ratio", 0.05)),
        "beta": float(rl_config.get("beta", 0.04)),
        "num_generations": int(rl_config.get("num_generations", 4)),
        "max_completion_length": int(rl_config.get("max_completion_length", 192)),
        "logging_steps": int(rl_config.get("logging_steps", 5)),
        "save_steps": int(rl_config.get("save_steps", 100)),
        "save_total_limit": 2,
        "bf16": model_config.get("dtype") == "bfloat16" and torch.cuda.is_bf16_supported(),
        "fp16": model_config.get("dtype") == "float16",
        "gradient_checkpointing": True,
        "report_to": "none",
        "seed": seed,
    }
    if "max_steps" in rl_config:
        arguments["max_steps"] = int(rl_config["max_steps"])
    trainer = stack["GRPOTrainer"](
        model=model,
        reward_funcs=[format_reward_func, quality_reward_func],
        args=stack["GRPOConfig"](**arguments),
        train_dataset=dataset,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    trainer.save_state()
    return final_dir

