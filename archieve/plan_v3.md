# LLM 时序异常检测操作手册 v3（Colab A100 + Drive 持久化）

---

## 0. 固定执行规则

1. 训练、推理、渲染、评估全部在 **Colab A100** 的本地盘 `/content` 运行。
2. 关键数据、API 生成数据、训练模型、checkpoint、最终结果全部持久化到 **Google Drive**。
3. **不要直接从 Drive 读取大量小文件训练。**
4. 图像目录、checkpoint 目录、评估输出目录在写入 Drive 前统一打包为 `.tar`。
5. `final_holdout` 生成前，不要生成 SFT 数据，不要训练，不要调参。
6. 所有实验都追加记录到 `run_log.csv`。

---

## 1. 初始化 Notebook

### 1.1 挂载 Drive

在第一个代码单元执行：

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 1.2 定义本地运行目录与 Drive 持久化目录

在第二个代码单元执行：

```python
from pathlib import Path

RUNTIME = Path("/content/tsad_runtime")
DRIVE_ROOT = Path("/content/drive/MyDrive/tsad_anomaly")

RT_CODE = RUNTIME / "code"
RT_DATA = RUNTIME / "data"
RT_IMAGES = RUNTIME / "images"
RT_SFT = RUNTIME / "sft"
RT_CKPT = RUNTIME / "checkpoints"
RT_RESULTS = RUNTIME / "results"

DRV_RAW = DRIVE_ROOT / "raw"
DRV_PACK = DRIVE_ROOT / "packs"
DRV_SFT = DRIVE_ROOT / "sft"
DRV_MODELS = DRIVE_ROOT / "models"
DRV_CKPT = DRIVE_ROOT / "checkpoints"
DRV_RESULTS = DRIVE_ROOT / "results"
DRV_LOGS = DRIVE_ROOT / "logs"

for p in [
    RUNTIME, RT_CODE, RT_DATA, RT_IMAGES, RT_SFT, RT_CKPT, RT_RESULTS,
    DRV_RAW, DRV_PACK, DRV_SFT, DRV_MODELS, DRV_CKPT, DRV_RESULTS, DRV_LOGS,
]:
    p.mkdir(parents=True, exist_ok=True)
```

### 1.3 安装依赖

在第三个代码单元执行：

```bash
pip install -U pip -q
pip install "unsloth[colab-new]" -q
pip install trl datasets matplotlib pillow openai accelerate bitsandbytes scikit-learn pandas -q
```

### 1.4 定义打包与恢复工具

在第四个代码单元执行：

```python
import os
import tarfile
import shutil
from pathlib import Path

def pack_dir(src_dir, dst_tar):
    src_dir = Path(src_dir)
    dst_tar = Path(dst_tar)
    dst_tar.parent.mkdir(parents=True, exist_ok=True)
    if dst_tar.exists():
        dst_tar.unlink()
    with tarfile.open(dst_tar, "w") as tar:
        tar.add(src_dir, arcname=src_dir.name)

def unpack_tar(src_tar, dst_parent):
    src_tar = Path(src_tar)
    dst_parent = Path(dst_parent)
    dst_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src_tar, "r") as tar:
        tar.extractall(dst_parent)

def sync_file(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def reset_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
```

### 1.5 检查 GPU

在第五个代码单元执行：

```python
import random
import numpy as np
import torch

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM_GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
print("BF16:", torch.cuda.is_bf16_supported())
```

### 1.6 检查点

满足以下条件后继续：

- `GPU` 显示 A100
- `BF16` 为 `True`

---

## 2. 获取代码与原始数据

### 2.1 恢复或克隆仓库

优先恢复本地代码目录。若不存在则克隆：

```bash
cd /content
if [ ! -d /content/tsad_runtime/code/AnomLLM ]; then
  git clone https://github.com/Rose-STL-Lab/AnomLLM.git /content/tsad_runtime/code/AnomLLM
fi
cd /content/tsad_runtime/code/AnomLLM
export PYTHONPATH=$PYTHONPATH:/content/tsad_runtime/code/AnomLLM/src
```

### 2.2 恢复或下载原始数据

优先从 Drive 恢复：

```bash
if [ -f /content/drive/MyDrive/tsad_anomaly/packs/anomllm_raw_data.tar ]; then
  mkdir -p /content/tsad_runtime/code/AnomLLM
  tar -xf /content/drive/MyDrive/tsad_anomaly/packs/anomllm_raw_data.tar -C /content/tsad_runtime/code/AnomLLM
fi
```

若本地仍无数据，则下载：

```bash
cd /content/tsad_runtime/code/AnomLLM
s5cmd --no-sign-request --endpoint-url https://s3-west.nrp-nautilus.io cp "s3://anomllm/data/*" data/
```

### 2.3 将原始数据打包到 Drive

下载完成后执行：

```bash
cd /content/tsad_runtime/code/AnomLLM
tar -cf /content/drive/MyDrive/tsad_anomaly/packs/anomllm_raw_data.tar -C /content/tsad_runtime/code/AnomLLM data
```

### 2.4 扫描数据目录

执行：

```bash
cd /content/tsad_runtime/code/AnomLLM
find data -maxdepth 2 -type d | sort
```

### 2.5 固定数据子集

在新代码单元执行：

```python
DATASETS = [
    "flat-trend",
    "range",
    "point",
    "freq",
    "noisy-point",
    "noisy-freq",
    "noisy-trend",
]
```

---

## 3. 生成 manifest 与 split

### 3.1 生成主 manifest

创建本地文件：

```text
/content/tsad_runtime/data/master_manifest.csv
```

固定列名：

```text
sample_id,source_subset,series_path,image_path,label,anomaly_type
```

### 3.2 生成 split

在新代码单元执行：

```python
import pandas as pd
from sklearn.model_selection import train_test_split

manifest = pd.read_csv("/content/tsad_runtime/data/master_manifest.csv")
manifest["strata"] = manifest["source_subset"].astype(str) + "__" + manifest["anomaly_type"].astype(str)

train_pool, final_holdout = train_test_split(
    manifest,
    test_size=0.20,
    random_state=3407,
    stratify=manifest["strata"],
)

sft_pool = train_pool.sample(n=min(1500, len(train_pool)), random_state=3407)
rest_pool = train_pool.drop(sft_pool.index)

train_df, temp_df = train_test_split(
    sft_pool,
    test_size=0.20,
    random_state=3407,
    stratify=sft_pool["strata"],
)

val_df, dev_test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=3407,
    stratify=temp_df["strata"],
)

train_pool.to_csv("/content/tsad_runtime/data/train_pool.csv", index=False)
final_holdout.to_csv("/content/tsad_runtime/data/final_holdout.csv", index=False)
sft_pool.to_csv("/content/tsad_runtime/data/sft_pool.csv", index=False)
train_df.to_csv("/content/tsad_runtime/data/train.csv", index=False)
val_df.to_csv("/content/tsad_runtime/data/val.csv", index=False)
dev_test_df.to_csv("/content/tsad_runtime/data/dev_test.csv", index=False)
rest_pool.to_csv("/content/tsad_runtime/data/rest_pool.csv", index=False)
```

### 3.3 将 split 文件同步到 Drive

在新代码单元执行：

```python
for name in [
    "master_manifest.csv",
    "train_pool.csv",
    "final_holdout.csv",
    "sft_pool.csv",
    "train.csv",
    "val.csv",
    "dev_test.csv",
    "rest_pool.csv",
]:
    sync_file(RT_DATA / name, DRV_RAW / name)
```

---

## 4. 渲染图像并打包到 Drive

### 4.1 固定渲染函数

在新代码单元执行：

```python
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def ts_to_image(values):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=112)  # 672 x 336
    ax.plot(values, color="#2f5d8a", linewidth=1.4)
    ax.set_xlim(0, len(values) - 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.15)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert("RGB")
```

### 4.2 渲染训练与评估图像到本地

本地图像目录固定为：

- `/content/tsad_runtime/images/sft`
- `/content/tsad_runtime/images/eval`

执行要求：

1. `train.csv`、`val.csv`、`dev_test.csv` 的图像保存到 `/content/tsad_runtime/images/sft/<sample_id>.png`
2. `final_holdout.csv` 的图像保存到 `/content/tsad_runtime/images/eval/<sample_id>.png`
3. 更新各自 CSV 中的 `image_path`

### 4.3 打包图像到 Drive

在新代码单元执行：

```python
pack_dir("/content/tsad_runtime/images/sft", "/content/drive/MyDrive/tsad_anomaly/packs/images_sft.tar")
pack_dir("/content/tsad_runtime/images/eval", "/content/drive/MyDrive/tsad_anomaly/packs/images_eval.tar")

for name in ["train.csv", "val.csv", "dev_test.csv", "final_holdout.csv"]:
    sync_file(RT_DATA / name, DRV_RAW / name)
```

### 4.4 恢复图像到本地

新 session 中如需恢复图像，执行：

```bash
mkdir -p /content/tsad_runtime/images
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/images_sft.tar -C /content/tsad_runtime/images
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/images_eval.tar -C /content/tsad_runtime/images
```

---

## 5. 运行官方 benchmark 的 zero-shot 基线

### 5.1 固定模型参数

在新代码单元执行：

```python
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 640 * 28 * 28
MAX_NEW_TOKENS = 256
```

### 5.2 启动 OpenAI-compatible endpoint

按附录 B 启动。  
服务地址固定为：

```text
http://127.0.0.1:8000/v1
```

### 5.3 写入 `credentials.yml`

在新代码单元执行：

```python
from pathlib import Path

credentials = """
qwen-local:
  api_key: dummy
  base_url: "http://127.0.0.1:8000/v1"
"""

Path("/content/tsad_runtime/code/AnomLLM/credentials.yml").write_text(credentials)
```

### 5.4 跑最小闭环

在 shell 单元执行：

```bash
cd /content/tsad_runtime/code/AnomLLM
python src/online_api.py --data flat-trend --model qwen-local --variant 0shot-vision
python src/result_agg.py --data flat-trend
```

### 5.5 跑主基线

在 shell 单元执行：

```bash
cd /content/tsad_runtime/code/AnomLLM

DATASETS=("flat-trend" "range" "point" "freq" "noisy-point" "noisy-freq" "noisy-trend")
VARIANTS=("0shot-vision" "0shot-text" "0shot-vision-cot")

for datum in "${DATASETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    python src/online_api.py --data "$datum" --model qwen-local --variant "$variant"
  done
done
```

### 5.6 打包 benchmark 结果到 Drive

在 shell 单元执行：

```bash
cd /content/tsad_runtime/code/AnomLLM
tar -cf /content/drive/MyDrive/tsad_anomaly/packs/official_baseline.tar results
```

---

## 6. 生成 SFT 教师数据并持久化到 Drive

### 6.1 设置 API Key

在新代码单元执行：

```python
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-xxxx"
```

### 6.2 固定教师模型参数

在新代码单元执行：

```python
TEACHER_MODEL = "qwen3.5-plus-2026-02-15"
TEACHER_FALLBACK = "qwen3.5-plus"
TEACHER_TEMPERATURE = 0.2
```

### 6.3 固定输出 JSON schema

教师输出必须包含：

```json
{
  "summary": "brief description",
  "is_anomaly": true,
  "anomaly_type": "point|range|freq|trend|none",
  "start_idx": 81,
  "end_idx": 96,
  "rationale": "short rationale"
}
```

### 6.4 生成原始教师数据到本地

在新代码单元执行：

```python
import os
import json
import base64
import pandas as pd
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = (
    "You are annotating time-series anomaly plots for supervised fine-tuning. "
    "Return strict JSON only."
)

USER_TEXT = (
    "Analyze this time-series plot and return JSON with keys: "
    "summary, is_anomaly, anomaly_type, start_idx, end_idx, rationale. "
    "Keep summary short. Keep rationale short. Do not output markdown."
)

train_df = pd.read_csv("/content/tsad_runtime/data/train.csv")
val_df = pd.read_csv("/content/tsad_runtime/data/val.csv")
dev_test_df = pd.read_csv("/content/tsad_runtime/data/dev_test.csv")
sft_df = pd.concat([train_df, val_df, dev_test_df], ignore_index=True)

rows = []

for _, row in sft_df.iterrows():
    with open(row["image_path"], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    try:
        resp = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": USER_TEXT},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
            temperature=TEACHER_TEMPERATURE,
        )
    except Exception:
        resp = client.chat.completions.create(
            model=TEACHER_FALLBACK,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": USER_TEXT},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
            temperature=TEACHER_TEMPERATURE,
        )

    parsed = json.loads(resp.choices[0].message.content)
    parsed["sample_id"] = row["sample_id"]
    parsed["image_path"] = row["image_path"]
    parsed["ground_truth"] = row["label"]
    parsed["ground_truth_type"] = row["anomaly_type"]
    rows.append(parsed)

with open("/content/tsad_runtime/sft/sft_raw.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

### 6.5 同步 API 结果到 Drive

在新代码单元执行：

```python
sync_file("/content/tsad_runtime/sft/sft_raw.jsonl", "/content/drive/MyDrive/tsad_anomaly/sft/sft_raw.jsonl")
```

---

## 7. 清洗 SFT 数据并打包

### 7.1 自动清洗

在新代码单元执行：

```python
import json

clean_rows = []
reject_rows = []

with open("/content/tsad_runtime/sft/sft_raw.jsonl", "r") as f:
    for line in f:
        row = json.loads(line)
        pred = row["is_anomaly"]
        gt = bool(row["ground_truth"])
        if pred == gt:
            clean_rows.append(row)
        else:
            reject_rows.append(row)

with open("/content/tsad_runtime/sft/sft_clean.jsonl", "w") as f:
    for r in clean_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open("/content/tsad_runtime/sft/sft_reject.jsonl", "w") as f:
    for r in reject_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

### 7.2 人工抽检

执行要求：

1. 从 `sft_clean.jsonl` 抽样 100 条
2. 检查 `is_anomaly`、`anomaly_type`、`start_idx/end_idx`、`rationale`
3. 输出：
   - `/content/tsad_runtime/sft/sft_reject_manual.jsonl`
   - `/content/tsad_runtime/sft/sft_final.jsonl`

### 7.3 同步清洗结果到 Drive

在新代码单元执行：

```python
for name in [
    "sft_clean.jsonl",
    "sft_reject.jsonl",
    "sft_reject_manual.jsonl",
    "sft_final.jsonl",
]:
    sync_file(RT_SFT / name, DRV_SFT / name)
```

---

## 8. 转换为训练格式并打包训练数据

### 8.1 固定回答模板

训练目标文本固定为：

```text
<summary>{summary}</summary>
<label>{ANOMALY|NORMAL}</label>
<type>{point|range|freq|trend|none}</type>
<region>{[start_idx,end_idx] or null}</region>
<rationale>{rationale}</rationale>
```

### 8.2 生成训练文件到本地

在新代码单元执行：

```python
import json
import pandas as pd

split_map = {}
for split_name in ["train", "val", "dev_test"]:
    df = pd.read_csv(f"/content/tsad_runtime/data/{split_name}.csv")
    for _, row in df.iterrows():
        split_map[row["sample_id"]] = split_name

writers = {
    "train": open("/content/tsad_runtime/sft/train.jsonl", "w"),
    "val": open("/content/tsad_runtime/sft/val.jsonl", "w"),
    "dev_test": open("/content/tsad_runtime/sft/dev_test.jsonl", "w"),
}

with open("/content/tsad_runtime/sft/sft_final.jsonl", "r") as f:
    for line in f:
        row = json.loads(line)
        label_text = "ANOMALY" if row["is_anomaly"] else "NORMAL"
        region_text = (
            f"[{row['start_idx']},{row['end_idx']}]"
            if row["start_idx"] is not None and row["end_idx"] is not None
            else "null"
        )
        assistant_text = (
            f"<summary>{row['summary']}</summary>\n"
            f"<label>{label_text}</label>\n"
            f"<type>{row['anomaly_type']}</type>\n"
            f"<region>{region_text}</region>\n"
            f"<rationale>{row['rationale']}</rationale>"
        )
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": row["image_path"]},
                        {"type": "text", "text": "Detect anomalies in this time-series plot."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": assistant_text,
                },
            ]
        }
        split_name = split_map[row["sample_id"]]
        writers[split_name].write(json.dumps(record, ensure_ascii=False) + "\n")

for f in writers.values():
    f.close()
```

### 8.3 将训练数据与元数据打包到 Drive

在新代码单元执行：

```python
for name in ["train.jsonl", "val.jsonl", "dev_test.jsonl", "sft_final.jsonl"]:
    sync_file(RT_SFT / name, DRV_SFT / name)

pack_dir("/content/tsad_runtime/sft", "/content/drive/MyDrive/tsad_anomaly/packs/sft_bundle.tar")
```

### 8.4 恢复训练数据到本地

新 session 中如需恢复，执行：

```bash
mkdir -p /content/tsad_runtime
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/sft_bundle.tar -C /content/tsad_runtime
```

---

## 9. 执行 QLoRA SFT

### 9.1 训练前恢复本地图像与训练数据

执行：

```bash
mkdir -p /content/tsad_runtime/images /content/tsad_runtime
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/images_sft.tar -C /content/tsad_runtime/images
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/sft_bundle.tar -C /content/tsad_runtime
```

### 9.2 定义 checkpoint 自动备份回调

在新代码单元执行：

```python
import tarfile
from pathlib import Path
from transformers import TrainerCallback

class DriveCheckpointCallback(TrainerCallback):
    def __init__(self, drive_root):
        self.drive_root = Path(drive_root)
        self.drive_root.mkdir(parents=True, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt_dir.exists():
            tar_path = self.drive_root / f"checkpoint-{state.global_step}.tar"
            if tar_path.exists():
                tar_path.unlink()
            with tarfile.open(tar_path, "w") as tar:
                tar.add(ckpt_dir, arcname=ckpt_dir.name)
        return control
```

### 9.3 加载模型与数据

在新代码单元执行：

```python
from datasets import load_dataset
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)

train_dataset = load_dataset("json", data_files="/content/tsad_runtime/sft/train.jsonl", split="train")
eval_dataset = load_dataset("json", data_files="/content/tsad_runtime/sft/val.jsonl", split="train")
```

### 9.4 配置训练输出到本地、备份到 Drive

在新代码单元执行：

```python
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

sft_args = SFTConfig(
    output_dir="/content/tsad_runtime/checkpoints/qwen3vl-tsad",
    max_length=None,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=0.3,
    report_to="none",
    remove_unused_columns=False,
    dataset_num_proc=2,
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
)

trainer.add_callback(DriveCheckpointCallback("/content/drive/MyDrive/tsad_anomaly/checkpoints/qwen3vl-tsad"))
```

### 9.5 启动训练

在新代码单元执行：

```python
trainer.train()
```

### 9.6 中断后恢复训练

选择最新 checkpoint tar，先解包到本地，再恢复：

```bash
mkdir -p /content/tsad_runtime/checkpoints/qwen3vl-tsad
tar -xf /content/drive/MyDrive/tsad_anomaly/checkpoints/qwen3vl-tsad/checkpoint-XXXX.tar -C /content/tsad_runtime/checkpoints/qwen3vl-tsad
```

然后在代码单元执行：

```python
trainer.train(resume_from_checkpoint="/content/tsad_runtime/checkpoints/qwen3vl-tsad/checkpoint-XXXX")
```

### 9.7 保存最终模型并归档到 Drive

在新代码单元执行：

```python
model.save_pretrained_merged("/content/tsad_runtime/sft/qwen3vl-tsad-merged", tokenizer)
model.save_pretrained("/content/tsad_runtime/sft/qwen3vl-tsad-adapter")
```

然后执行：

```python
pack_dir("/content/tsad_runtime/sft/qwen3vl-tsad-merged", "/content/drive/MyDrive/tsad_anomaly/models/qwen3vl-tsad-merged.tar")
pack_dir("/content/tsad_runtime/sft/qwen3vl-tsad-adapter", "/content/drive/MyDrive/tsad_anomaly/models/qwen3vl-tsad-adapter.tar")
```

---

## 10. 在 dev_test 上做验证

### 10.1 恢复模型到本地

执行：

```bash
mkdir -p /content/tsad_runtime/models
tar -xf /content/drive/MyDrive/tsad_anomaly/models/qwen3vl-tsad-merged.tar -C /content/tsad_runtime/models
```

### 10.2 恢复训练数据与图像到本地

执行：

```bash
mkdir -p /content/tsad_runtime/images /content/tsad_runtime
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/images_sft.tar -C /content/tsad_runtime/images
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/sft_bundle.tar -C /content/tsad_runtime
```

### 10.3 输出验证结果

固定输出文件：

```text
/content/tsad_runtime/results/dev_test_sft_predictions.csv
/content/tsad_runtime/results/dev_test_metrics.csv
```

### 10.4 将验证结果同步到 Drive

在新代码单元执行：

```python
sync_file("/content/tsad_runtime/results/dev_test_sft_predictions.csv", "/content/drive/MyDrive/tsad_anomaly/results/dev_test_sft_predictions.csv")
sync_file("/content/tsad_runtime/results/dev_test_metrics.csv", "/content/drive/MyDrive/tsad_anomaly/results/dev_test_metrics.csv")
```

---

## 11. 在 final_holdout 上执行最终评估

### 11.1 恢复最终评估图像到本地

执行：

```bash
mkdir -p /content/tsad_runtime/images
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/images_eval.tar -C /content/tsad_runtime/images
```

### 11.2 固定评估对象

只评估：

1. `Qwen3-VL-8B zero-shot`
2. `Qwen3-VL-8B + SFT`
3. `Qwen3-VL-8B + SFT + GRPO`（如已训练）

### 11.3 输出最终评估文件

固定输出：

```text
/content/tsad_runtime/results/final_holdout_<model_name>.csv
/content/tsad_runtime/results/final_summary.csv
```

### 11.4 将最终结果同步到 Drive

在新代码单元执行：

```python
for p in Path("/content/tsad_runtime/results").glob("final_holdout_*.csv"):
    sync_file(p, Path("/content/drive/MyDrive/tsad_anomaly/results") / p.name)

sync_file("/content/tsad_runtime/results/final_summary.csv", "/content/drive/MyDrive/tsad_anomaly/results/final_summary.csv")
```

---

## 12. 可选路径：执行 GRPO

### 12.1 训练前恢复本地图像与训练数据

执行：

```bash
mkdir -p /content/tsad_runtime/images /content/tsad_runtime
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/images_sft.tar -C /content/tsad_runtime/images
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/sft_bundle.tar -C /content/tsad_runtime
```

### 12.2 Warm-up SFT 基座

在新代码单元执行：

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Thinking-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
```

### 12.3 固定输出模板

```text
<summary>...</summary>
<label>ANOMALY|NORMAL</label>
<type>point|range|freq|trend|none</type>
<region>[start,end] or null</region>
```

### 12.4 固定 reward 权重

- `0.2`：格式正确
- `0.5`：`label` 正确
- `0.2`：`type` 正确
- `0.1`：`region` 与标注区间有重叠

### 12.5 GRPO 输出目录

本地：

```text
/content/tsad_runtime/checkpoints/qwen3vl-tsad-grpo
```

Drive：

```text
/content/drive/MyDrive/tsad_anomaly/checkpoints/qwen3vl-tsad-grpo
```

### 12.6 训练结束后归档模型到 Drive

输出归档：

- `/content/drive/MyDrive/tsad_anomaly/models/qwen3vl-tsad-grpo-merged.tar`

---

## 13. 运行日志

### 13.1 固定日志文件

日志文件固定为：

```text
/content/drive/MyDrive/tsad_anomaly/logs/run_log.csv
```

### 13.2 固定字段

```text
run_id,stage,model_name,split_version,image_size,min_pixels,max_pixels,lora_r,lr,batch_size,grad_accum,vision_layers,checkpoint_name,val_f1,val_precision,val_recall,notes
```

---

## 14. 交付物清单

执行结束后确认以下 Drive 文件存在：

- `/content/drive/MyDrive/tsad_anomaly/packs/anomllm_raw_data.tar`
- `/content/drive/MyDrive/tsad_anomaly/packs/images_sft.tar`
- `/content/drive/MyDrive/tsad_anomaly/packs/images_eval.tar`
- `/content/drive/MyDrive/tsad_anomaly/packs/sft_bundle.tar`
- `/content/drive/MyDrive/tsad_anomaly/sft/sft_raw.jsonl`
- `/content/drive/MyDrive/tsad_anomaly/sft/sft_final.jsonl`
- `/content/drive/MyDrive/tsad_anomaly/models/qwen3vl-tsad-merged.tar`
- `/content/drive/MyDrive/tsad_anomaly/models/qwen3vl-tsad-adapter.tar`
- `/content/drive/MyDrive/tsad_anomaly/results/dev_test_metrics.csv`
- `/content/drive/MyDrive/tsad_anomaly/results/final_summary.csv`
- `/content/drive/MyDrive/tsad_anomaly/logs/run_log.csv`

---

## 附录 A. 存疑项：AnomLLM 数据目录与字段

执行前先核实：

1. `DATASETS` 的实际名称是否与以下一致：
   - `flat-trend`
   - `range`
   - `point`
   - `freq`
   - `noisy-point`
   - `noisy-freq`
   - `noisy-trend`
2. 原始数据文件格式是否可直接映射到 `series_path`
3. `label` 与 `anomaly_type` 是否能从原始元数据稳定读取

如不一致，先修改 `master_manifest.csv` 生成逻辑，再回到主流程。

---

## 附录 B. 存疑项：OpenAI-compatible endpoint

主流程默认使用 OpenAI-compatible endpoint。  
可选实现：

1. `vLLM`
2. `SGLang`
3. 自定义 FastAPI / Flask wrapper

固定要求：

- 地址：`http://127.0.0.1:8000/v1`
- 支持图像输入
- 支持基线模型与 merged SFT 模型

如无法服务化，改为 notebook 直接推理，并将结果写到 `RT_RESULTS` 后同步到 Drive。

---

## 附录 C. 存疑项：百炼模型快照

优先使用：

```text
qwen3.5-plus-2026-02-15
```

不可用时改为：

```text
qwen3.5-plus
```

固定参数：

- `base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"`
- `extra_body={"enable_thinking": False}`
- `response_format={"type": "json_object"}`

---

## 附录 D. 存疑项：OOM 回退顺序

发生显存不足时，按顺序回退：

1. `per_device_train_batch_size: 2 -> 1`
2. `gradient_accumulation_steps: 8 -> 16`
3. `finetune_vision_layers: True -> False`
4. `max_pixels: 640 * 28 * 28 -> 512 * 28 * 28`
5. `r: 16 -> 8`

每次只改一个变量，并更新 `run_log.csv`。

---

## 附录 E. 存疑项：官方 benchmark variant 名称

优先使用：

- `0shot-vision`
- `0shot-text`
- `0shot-vision-cot`

如不匹配，先执行：

```bash
cd /content/tsad_runtime/code/AnomLLM
sed -n '1,220p' test.sh
```

再按实际 variant 修改命令。

---

## 附录 F. 存疑项：`s5cmd` 下载失败

如 `s5cmd` 不可用：

1. 使用 AnomLLM README 中的 Google Drive 链接下载原始数据
2. 将数据放入：

```text
/content/tsad_runtime/code/AnomLLM/data
```

3. 执行：

```bash
cd /content/tsad_runtime/code/AnomLLM
tar -cf /content/drive/MyDrive/tsad_anomaly/packs/anomllm_raw_data.tar -C /content/tsad_runtime/code/AnomLLM data
```
