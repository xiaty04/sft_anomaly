from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from .config import save_effective_config, save_run_metadata, set_seed
from .intervals import parse_interval_output, to_jsonable
from .io import append_jsonl, read_jsonl
from .modalities import text_prompt, validate_modality
from .prompts import interval_prompt


def _torch_dtype(torch: Any, name: str) -> Any:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"unsupported dtype: {name}")
    return values[name]


def _load_model(model_name: str, adapter_path: Optional[str], model_config: Dict[str, Any]):
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install the train extra before model inference: pip install -e '.[train]'") from exc

    processor_source = adapter_path or model_name
    try:
        processor = AutoProcessor.from_pretrained(
            processor_source,
            min_pixels=int(model_config.get("min_pixels", 200704)),
            max_pixels=int(model_config.get("max_pixels", 1003520)),
        )
    except (OSError, ValueError):
        processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=int(model_config.get("min_pixels", 200704)),
            max_pixels=int(model_config.get("max_pixels", 1003520)),
        )
    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": _torch_dtype(torch, model_config.get("dtype", "bfloat16")),
        "attn_implementation": model_config.get("attn_implementation", "sdpa"),
    }
    if bool(model_config.get("load_in_4bit", True)):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=kwargs["torch_dtype"],
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load an adapter") from exc
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, processor, torch


def _generate(
    model: Any,
    processor: Any,
    torch: Any,
    record: Dict[str, Any],
    config: Dict[str, Any],
    modality: str,
) -> str:
    start = int(record.get("window_start", 0))
    end = int(record.get("window_end", record["length"]))
    image = None
    if modality == "vision":
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB").copy()
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": interval_prompt(start, end)},
        ]
    else:
        content = [{"type": "text", "text": text_prompt(record)}]
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
    if image is not None:
        processor_kwargs["images"] = [image]
    inputs = processor(**processor_kwargs)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    temperature = float(config.get("temperature", 0.0))
    generation = model.generate(
        **inputs,
        max_new_tokens=int(config.get("max_new_tokens", 192)),
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-6) if temperature > 0 else None,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    generated = generation[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def run_inference(
    config: Dict[str, Any],
    manifest_path: Path,
    output_path: Path,
    modality: str,
    model_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    validate_modality(modality)
    model_config = dict(config["model"])
    model_name = model_name or model_config["name"]
    seed = int(config["project"].get("seed", 3407))
    set_seed(seed)
    records = read_jsonl(manifest_path)
    if limit is not None:
        records = records[:limit]
    completed = set()
    if output_path.exists():
        completed = {record["sample_id"] for record in read_jsonl(output_path)}
    pending = [record for record in records if record["sample_id"] not in completed]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_effective_config(config, output_path.parent)
    save_run_metadata(
        {
            "model": model_name,
            "adapter_path": adapter_path,
            "manifest": str(manifest_path),
            "modality": modality,
            "python": platform.python_version(),
            "seed": seed,
        },
        output_path.parent,
    )
    print(
        f"[infer] modality={modality} manifest={manifest_path} selected={len(records)} "
        f"completed={len(records) - len(pending)} pending={len(pending)} output={output_path}",
        flush=True,
    )
    if not pending:
        print("[infer] all selected records are already complete; model loading skipped", flush=True)
        return
    load_started = time.perf_counter()
    print(
        f"[infer] loading model={model_name} adapter={adapter_path or 'none'} "
        f"dtype={model_config.get('dtype', 'bfloat16')} 4bit={model_config.get('load_in_4bit', True)}",
        flush=True,
    )
    model, processor, torch = _load_model(model_name, adapter_path, model_config)
    print(
        f"[infer] model ready after {time.perf_counter() - load_started:.1f}s; "
        f"processing {len(pending)} record(s)",
        flush=True,
    )
    run_started = time.perf_counter()
    for processed, record in enumerate(pending, start=1):
        start = int(record.get("window_start", 0))
        end = int(record.get("window_end", record["length"]))
        print(f"[infer] {processed}/{len(pending)} {record['sample_id']} ...", flush=True)
        started = time.perf_counter()
        output = _generate(model, processor, torch, record, model_config, modality)
        parsed = parse_interval_output(output, lower=start, upper=end)
        append_jsonl(
            output_path,
            {
                "sample_id": record["sample_id"],
                "series_id": record.get("series_id", record["sample_id"]),
                "window_start": start,
                "window_end": end,
                "raw_output": output,
                "intervals": to_jsonable(parsed.intervals),
                "parse_valid": parsed.valid,
                "parse_error": parsed.error,
                "latency_seconds": time.perf_counter() - started,
                "model": model_name,
                "modality": modality,
            },
        )
        elapsed = time.perf_counter() - run_started
        average = elapsed / processed
        eta = average * (len(pending) - processed)
        print(
            f"[infer] wrote {record['sample_id']} parse_valid={parsed.valid} "
            f"latency={time.perf_counter() - started:.2f}s "
            f"average={average:.2f}s eta={eta:.1f}s",
            flush=True,
        )
    print(
        f"[infer] done: {len(pending)} new prediction(s) appended to {output_path}; "
        f"elapsed={time.perf_counter() - run_started:.1f}s",
        flush=True,
    )
