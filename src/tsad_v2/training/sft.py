from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from ..config import save_effective_config, save_run_metadata, set_seed
from ..data.common import load_training_manifest
from ..intervals import canonicalize, to_json
from ..prompts import interval_prompt


def _load_training_stack():
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install the train extra first: pip install -e '.[train]'") from exc
    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModel": AutoModelForImageTextToText,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _dtype(torch: Any, name: str) -> Any:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_sft_model(config: Dict[str, Any], stack: Dict[str, Any]):
    torch = stack["torch"]
    model_config = config["model"]
    dtype = _dtype(torch, model_config.get("dtype", "bfloat16"))
    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": dtype,
        "attn_implementation": model_config.get("attn_implementation", "sdpa"),
    }
    if bool(model_config.get("load_in_4bit", True)):
        kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = stack["AutoModel"].from_pretrained(model_config["name"], **kwargs)
    if bool(model_config.get("load_in_4bit", True)):
        model = stack["prepare_model_for_kbit_training"](
            model, use_gradient_checkpointing=bool(config["sft"].get("gradient_checkpointing", True))
        )
    lora = stack["LoraConfig"](
        r=int(config["sft"].get("lora_r", 16)),
        lora_alpha=int(config["sft"].get("lora_alpha", 32)),
        lora_dropout=float(config["sft"].get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = stack["get_peft_model"](model, lora)
    model.config.use_cache = False
    return model


class ManifestDataset:
    def __init__(self, records: Sequence[Dict[str, Any]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


class VisionSFTCollator:
    def __init__(self, processor: Any):
        self.processor = processor
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        self.processor.tokenizer.padding_side = "right"

    @staticmethod
    def _messages(record: Dict[str, Any], image: Image.Image, include_answer: bool) -> List[Dict[str, Any]]:
        start = int(record.get("window_start", 0))
        end = int(record.get("window_end", record["length"]))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": interval_prompt(start, end)},
                ],
            }
        ]
        if include_answer:
            truth = canonicalize(record.get("intervals", []), lower=start, upper=end)
            messages.append({"role": "assistant", "content": to_json(truth)})
        return messages

    def __call__(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images, full_texts, prompt_texts = [], [], []
        for record in records:
            with Image.open(record["image_path"]) as source:
                image = source.convert("RGB").copy()
            images.append(image)
            full_texts.append(
                self.processor.apply_chat_template(
                    self._messages(record, image, True), tokenize=False, add_generation_prompt=False
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    self._messages(record, image, False), tokenize=False, add_generation_prompt=True
                )
            )
        batch = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        prompt_batch = self.processor(
            text=prompt_texts, images=images, padding=True, return_tensors="pt"
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row, prompt_length in enumerate(prompt_lengths):
            labels[row, : int(prompt_length)] = -100
        batch["labels"] = labels
        return batch


def train_sft(
    config: Dict[str, Any],
    resume_from_checkpoint: Optional[str] = None,
    limit: Optional[int] = None,
) -> Path:
    started = time.perf_counter()
    print("[train-sft] loading training libraries", flush=True)
    stack = _load_training_stack()
    torch = stack["torch"]
    seed = int(config["project"].get("seed", 3407))
    set_seed(seed)
    sft_config = config["sft"]
    train_records = load_training_manifest(Path(sft_config["train_manifest"]))
    val_records = load_training_manifest(Path(sft_config["val_manifest"]))
    if any(record["split"] != "train" for record in train_records):
        raise ValueError("SFT train manifest contains a non-train split")
    if any(record["split"] != "val" for record in val_records):
        raise ValueError("SFT validation manifest contains a non-val split")
    if limit is not None:
        train_records, val_records = train_records[:limit], val_records[: max(1, limit // 4)]
    output_dir = Path(sft_config["output_dir"])
    print(
        f"[train-sft] train={len(train_records)} val={len(val_records)} "
        f"output={output_dir} resume={resume_from_checkpoint or 'none'}",
        flush=True,
    )
    save_effective_config(config, output_dir)
    save_run_metadata(
        {"stage": "sft", "model": config["model"]["name"], "seed": seed}, output_dir
    )
    print(f"[train-sft] loading processor {config['model']['name']}", flush=True)
    processor = stack["AutoProcessor"].from_pretrained(
        config["model"]["name"],
        min_pixels=int(config["model"].get("min_pixels", 200704)),
        max_pixels=int(config["model"].get("max_pixels", 1003520)),
    )
    print(
        f"[train-sft] loading 4-bit base model {config['model']['name']} and attaching LoRA",
        flush=True,
    )
    model_started = time.perf_counter()
    model = load_sft_model(config, stack)
    print(f"[train-sft] model ready after {time.perf_counter() - model_started:.1f}s", flush=True)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    use_bf16 = config["model"].get("dtype") == "bfloat16" and torch.cuda.is_bf16_supported()
    arguments: Dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": float(sft_config.get("num_train_epochs", 2)),
        "per_device_train_batch_size": int(sft_config.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(sft_config.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(sft_config.get("gradient_accumulation_steps", 16)),
        "learning_rate": float(sft_config.get("learning_rate", 2e-4)),
        "warmup_ratio": float(sft_config.get("warmup_ratio", 0.05)),
        "weight_decay": float(sft_config.get("weight_decay", 0.01)),
        "logging_steps": int(sft_config.get("logging_steps", 5)),
        "logging_first_step": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "bf16": use_bf16,
        "fp16": config["model"].get("dtype") == "float16",
        "gradient_checkpointing": bool(sft_config.get("gradient_checkpointing", True)),
        "remove_unused_columns": False,
        "report_to": "none",
        "disable_tqdm": False,
        "seed": seed,
    }
    if "max_steps" in sft_config:
        arguments["max_steps"] = int(sft_config["max_steps"])
        arguments["save_strategy"] = "steps"
        arguments["eval_strategy"] = "steps"
        arguments["save_steps"] = int(sft_config.get("save_steps", arguments["max_steps"]))
        arguments["eval_steps"] = arguments["save_steps"]
    training_args = stack["TrainingArguments"](**arguments)
    trainer = stack["Trainer"](
        model=model,
        args=training_args,
        train_dataset=ManifestDataset(train_records),
        eval_dataset=ManifestDataset(val_records),
        data_collator=VisionSFTCollator(processor),
    )
    print("[train-sft] trainer starting; step loss and progress follow", flush=True)
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"[train-sft] train metrics={result.metrics}", flush=True)
    final_dir = output_dir / "final_adapter"
    print(f"[train-sft] saving final adapter -> {final_dir}", flush=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    trainer.save_state()
    print(f"[train-sft] complete elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return final_dir
