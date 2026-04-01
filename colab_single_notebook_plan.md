# Colab 单 Notebook 执行计划

## 1. 文档定位

这份文档把项目整理成一个适合直接落到 `Colab notebook` 的执行计划，并明确三层方法优先级：

- `SFT` 之前的 baseline，优先复现 `AnomLLM` 官方仓库和 README 的 git 使用方式。
- `SFT` 和 `GRPO`，优先对齐 `Unsloth` 官方 Vision fine-tuning / Qwen3-VL / VLM RL 文档与 notebook 范式。
- manifest、split、教师标注、清洗、最终评估这些官方方案没有直接覆盖的部分，保留为本项目的自定义补充代码。

目标不是只给流程说明，而是同时提供：

- 单 notebook 的阶段设计。
- 每一阶段优先依赖的官方来源。
- 可直接放进 Colab 的规范代码单元。
- 断线恢复、Drive 持久化和检查点规则。

默认前提：

- 只使用一个 notebook。
- 训练、推理、渲染、评估都在 Colab 本地盘 `/content` 运行。
- 所有跨 session 需要保留的产物都同步到 Google Drive。

## 2. 官方与源码参考链接

以下链接已在 `2026-04-01` 查阅，用于约束当前 notebook 设计。

### 平台与运行环境

- Google Colab FAQ: [https://research.google.com/colaboratory/faq.html](https://research.google.com/colaboratory/faq.html)
- PyTorch `torch.cuda.is_bf16_supported`: [https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html)

### Baseline 官方链路

- AnomLLM 仓库: [https://github.com/Rose-STL-Lab/AnomLLM](https://github.com/Rose-STL-Lab/AnomLLM)
- AnomLLM 论文: [https://arxiv.org/abs/2410.05440](https://arxiv.org/abs/2410.05440)
- vLLM OpenAI-Compatible Server: [https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

### SFT / GRPO 官方链路

- Unsloth Vision Fine-tuning: [https://docs.unsloth.ai/basics/vision-fine-tuning](https://docs.unsloth.ai/basics/vision-fine-tuning)
- Unsloth Qwen3-VL Guide: [https://docs.unsloth.ai/models/qwen3-vl](https://docs.unsloth.ai/models/qwen3-vl)
- Unsloth VLM RL Guide: [https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl)
- Qwen3-VL-8B-Instruct 模型卡: [https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

### 自定义补充代码依赖

- Hugging Face Datasets `load_dataset`: [https://huggingface.co/docs/datasets/main/loading](https://huggingface.co/docs/datasets/main/loading)

### 为什么这些链接必须放进 notebook 计划

- Colab FAQ 明确提示 Drive 目录中的大量小文件读写容易出问题，因此本方案坚持“Drive 持久化、本地盘训练、目录先打包再恢复”。
- `AnomLLM` README 明确给出了数据下载、`credentials.yml`、`online_api.py` 和 `result_agg.py` 的官方用法，因此 baseline 直接按官方 git 链路组织。
- `Unsloth` 官方文档已经给出了 Vision SFT、Qwen3-VL 和 VLM RL 的标准范式，因此 `SFT` 和 `GRPO` 不再把 `TRL` 文档当作主参考，而是只把 `TRL` 视为 Unsloth 底层训练组件之一。

## 3. 代码规范约定

这个 notebook 中的代码统一遵循以下约定：

- 路径统一使用 `pathlib.Path`。
- 固定常量统一大写命名。
- 一个 cell 只做一类事情：初始化、下载、切分、训练、评估不要混写。
- 所有关键产物在写入前先 `mkdir(parents=True, exist_ok=True)`。
- 对外部依赖和关键文件增加存在性检查。
- 对中间产物统一打印路径、行数、数量或文件大小，保证 notebook 可检查。
- 对仍需根据 `AnomLLM` 实际字段补齐的地方，用 `TODO` 明确标记，不在 notebook 中埋隐式假设。
- 文中所有“官方”一词仅指 `AnomLLM` baseline 流程或 `Unsloth` SFT/GRPO 流程，不指本项目的自定义数据桥接代码。

## 4. 固定执行规则

这些规则直接继承自 [plan_v3.md](/Users/tianyuxia/Desktop/sft_anomaly/archieve/plan_v3.md)，是整个 notebook 的硬约束：

1. 训练、推理、渲染、评估全部在 Colab 本地盘 `/content` 运行。
2. 关键数据、baseline 结果、SFT 数据、训练模型、checkpoint、最终结果全部持久化到 Google Drive。
3. 不直接从 Drive 读取大量小文件训练。
4. 图像目录、checkpoint 目录、评估输出目录在写入 Drive 前统一打包。
5. `final_holdout` 生成前，不允许生成 SFT 数据，不允许训练，不允许调参。
6. 所有实验都追加记录到 `run_log.csv`。

## 5. Notebook 阶段总览

| 阶段 | 目标 | 产物定位 |
| --- | --- | --- |
| 0 | 初始化环境 | Colab 与 Drive 运行时 |
| 1 | 获取 AnomLLM 代码与原始数据 | 官方 baseline 前置 |
| 2 | 生成 manifest 与 split | 自定义补充 |
| 3 | 渲染图像并打包 | 自定义补充 |
| 4 | 运行 AnomLLM 官方 benchmark baseline | `official_baseline.tar` |
| 5 | 生成教师 SFT 数据 | 自定义补充 |
| 6 | 清洗并确定最终 SFT 数据 | 自定义补充 |
| 7 | 转成 Unsloth 训练可用格式 | 自定义补充 |
| 8 | 执行 Unsloth Vision QLoRA SFT | 官方 SFT 主线 |
| 9 | 在 `dev_test` 上验证 | 自定义评估 |
| 10 | 在 `final_holdout` 上最终评估 | 自定义评估 |
| 11 | 可选 Unsloth VLM RL / GRPO | 官方 RL 主线 + 自定义 reward |

## 6. Cell-by-Cell 执行计划

### 阶段 0. 初始化环境

#### Cell 0.1 挂载 Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

#### Cell 0.2 定义运行目录与 Drive 持久化目录

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

for path in [
    RUNTIME, RT_CODE, RT_DATA, RT_IMAGES, RT_SFT, RT_CKPT, RT_RESULTS,
    DRV_RAW, DRV_PACK, DRV_SFT, DRV_MODELS, DRV_CKPT, DRV_RESULTS, DRV_LOGS,
]:
    path.mkdir(parents=True, exist_ok=True)
```

#### Cell 0.3 安装依赖

```bash
pip install -U pip -q
pip install "unsloth[colab-new]" -q
pip install vllm matplotlib pillow openai accelerate bitsandbytes scikit-learn pandas pyyaml datasets -q
```

#### Cell 0.4 定义公共工具函数

```python
import shutil
import tarfile
from pathlib import Path


def pack_dir(src_dir: str | Path, dst_tar: str | Path) -> None:
    src_dir = Path(src_dir)
    dst_tar = Path(dst_tar)
    dst_tar.parent.mkdir(parents=True, exist_ok=True)
    if dst_tar.exists():
        dst_tar.unlink()
    with tarfile.open(dst_tar, "w") as tar:
        tar.add(src_dir, arcname=src_dir.name)


def unpack_tar(src_tar: str | Path, dst_parent: str | Path) -> None:
    src_tar = Path(src_tar)
    dst_parent = Path(dst_parent)
    dst_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src_tar, "r") as tar:
        tar.extractall(dst_parent)


def sync_file(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
```

#### Cell 0.5 固定随机种子并检查 GPU

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

#### 阶段 0 检查点

- GPU 显示为 A100。
- `BF16` 为 `True`。
- 所有运行目录已创建。

### 阶段 1. 获取代码与原始数据

这一阶段完全按 `AnomLLM` 官方 baseline 链路组织，不重新设计仓库结构和入口脚本。

#### Cell 1.1 恢复或克隆 `AnomLLM`

```bash
cd /content
if [ ! -d /content/tsad_runtime/code/AnomLLM ]; then
  git clone https://github.com/Rose-STL-Lab/AnomLLM.git /content/tsad_runtime/code/AnomLLM
fi
cd /content/tsad_runtime/code/AnomLLM
export PYTHONPATH=$PYTHONPATH:/content/tsad_runtime/code/AnomLLM/src
```

#### Cell 1.2 优先从 Drive 恢复原始数据

```bash
if [ -f /content/drive/MyDrive/tsad_anomaly/packs/anomllm_raw_data.tar ]; then
  mkdir -p /content/tsad_runtime/code/AnomLLM
  tar -xf /content/drive/MyDrive/tsad_anomaly/packs/anomllm_raw_data.tar -C /content/tsad_runtime/code/AnomLLM
fi
```

#### Cell 1.3 Drive 中无备份时，按 AnomLLM README 下载

```bash
cd /content/tsad_runtime/code/AnomLLM
s5cmd --no-sign-request --endpoint-url https://s3-west.nrp-nautilus.io cp "s3://anomllm/data/*" data/
```

#### Cell 1.4 将原始数据重新打包到 Drive

```bash
cd /content/tsad_runtime/code/AnomLLM
tar -cf /content/drive/MyDrive/tsad_anomaly/packs/anomllm_raw_data.tar -C /content/tsad_runtime/code/AnomLLM data
```

#### Cell 1.5 固定使用的数据子集

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

### 阶段 2. 生成 manifest 与 split

这部分不是 `AnomLLM` 官方 baseline，也不是 `Unsloth` 官方训练内容，而是把官方 baseline 接到本项目自定义 SFT 数据管线的桥接层。

#### Cell 2.1 固定 manifest 列名

```python
MANIFEST_COLUMNS = [
    "sample_id",
    "source_subset",
    "series_path",
    "image_path",
    "label",
    "anomaly_type",
]
```

#### Cell 2.2 生成主 manifest

```python
from pathlib import Path

import pandas as pd

ANOMLLM_DATA_ROOT = RT_CODE / "AnomLLM" / "data"
MASTER_MANIFEST_PATH = RT_DATA / "master_manifest.csv"


def build_master_manifest() -> pd.DataFrame:
    """
    TODO:
    根据 AnomLLM 实际文件结构，把原始样本映射为：
    sample_id, source_subset, series_path, image_path, label, anomaly_type
    """
    raise NotImplementedError("请按 AnomLLM 实际数据结构补齐 manifest 构造逻辑。")
```

#### Cell 2.3 生成 split

```python
from sklearn.model_selection import train_test_split

manifest = pd.read_csv(MASTER_MANIFEST_PATH)
manifest["strata"] = (
    manifest["source_subset"].astype(str) + "__" + manifest["anomaly_type"].astype(str)
)

train_pool, final_holdout = train_test_split(
    manifest,
    test_size=0.20,
    random_state=SEED,
    stratify=manifest["strata"],
)

sft_pool = train_pool.sample(n=min(1500, len(train_pool)), random_state=SEED)
rest_pool = train_pool.drop(sft_pool.index)

train_df, temp_df = train_test_split(
    sft_pool,
    test_size=0.20,
    random_state=SEED,
    stratify=sft_pool["strata"],
)

val_df, dev_test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["strata"],
)
```

#### 阶段 2 检查点

- `final_holdout.csv` 先于 SFT 数据生成并固定。
- `train.csv`、`val.csv`、`dev_test.csv`、`final_holdout.csv` 全部存在。

### 阶段 3. 渲染图像并打包

这部分仍然是自定义补充，用来把时序样本转成后续 baseline 对比、教师标注和 VLM 微调都能复用的图像输入。

#### Cell 3.1 定义时序读取和渲染函数

```python
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def load_series(series_path: str | Path) -> np.ndarray:
    series_path = Path(series_path)

    if series_path.suffix == ".npy":
        values = np.load(series_path)
    elif series_path.suffix == ".csv":
        frame = pd.read_csv(series_path)
        values = frame.iloc[:, -1].to_numpy()
    else:
        raise ValueError(f"Unsupported series format: {series_path}")

    return np.asarray(values, dtype=float).reshape(-1)


def ts_to_image(values: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=112)
    ax.plot(values, color="#2F5D8A", linewidth=1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0.15)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")
```

#### Cell 3.2 批量渲染 split 图像

```python
def render_split_images(csv_name: str, image_subdir: str) -> None:
    csv_path = RT_DATA / csv_name
    output_dir = RT_IMAGES / image_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    updated_paths = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        image_path = output_dir / f"{sample_id}.png"
        values = load_series(row["series_path"])
        image = ts_to_image(values)
        image.save(image_path)
        updated_paths.append(str(image_path))

    df["image_path"] = updated_paths
    df.to_csv(csv_path, index=False)


render_split_images("train.csv", "sft")
render_split_images("val.csv", "sft")
render_split_images("dev_test.csv", "sft")
render_split_images("final_holdout.csv", "eval")
```

#### Cell 3.3 打包图像并同步 CSV

```python
pack_dir(RT_IMAGES / "sft", DRV_PACK / "images_sft.tar")
pack_dir(RT_IMAGES / "eval", DRV_PACK / "images_eval.tar")

for name in ["train.csv", "val.csv", "dev_test.csv", "final_holdout.csv"]:
    sync_file(RT_DATA / name, DRV_RAW / name)
```

### 阶段 4. 运行 AnomLLM 官方 benchmark baseline

这一段坚持按 `AnomLLM` 官方 git 用法走，不把官方 baseline 和后续自定义 split 评估混成同一套 benchmark。

#### Cell 4.1 固定 baseline 模型与 endpoint

```python
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
MAX_NEW_TOKENS = 256
```

#### Cell 4.2 启动 vLLM OpenAI-compatible server

默认 endpoint 固定为 `vLLM`，不再保留 `SGLang / 自定义 wrapper` 分支。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 \
  --port 8000
```

#### Cell 4.3 写入 `credentials.yml`

```python
credentials_text = """
qwen-local:
  api_key: dummy
  base_url: "http://127.0.0.1:8000/v1"
""".strip() + "\n"

(RT_CODE / "AnomLLM" / "credentials.yml").write_text(credentials_text)
```

#### Cell 4.4 跑最小闭环

```bash
cd /content/tsad_runtime/code/AnomLLM
python src/online_api.py --data flat-trend --model qwen-local --variant 0shot-vision
python src/result_agg.py --data flat-trend
```

#### Cell 4.5 跑完整官方 baseline

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

#### Cell 4.6 打包官方 baseline 结果

```bash
cd /content/tsad_runtime/code/AnomLLM
tar -cf /content/drive/MyDrive/tsad_anomaly/packs/official_baseline.tar results
```

#### 阶段 4 检查点

- `online_api.py` 和 `result_agg.py` 的最小闭环可运行。
- `/content/tsad_runtime/code/AnomLLM/results` 中已有官方 benchmark 输出。
- Drive 中存在 `official_baseline.tar`。

### 阶段 5. 生成教师 SFT 数据

从这里开始进入本项目自定义数据桥接层。这里不是 `AnomLLM` 官方 baseline，也不是 `Unsloth` 官方训练代码，而是为后续 Unsloth SFT 准备监督数据。

#### Cell 5.1 设置 API Key

```python
import os

os.environ["DASHSCOPE_API_KEY"] = "sk-xxxx"
```

#### Cell 5.2 设置教师模型参数与输出 schema

```python
TEACHER_MODEL = "qwen3.5-plus-2026-02-15"
TEACHER_TEMPERATURE = 0.2

TARGET_SCHEMA = {
    "summary": "brief description",
    "is_anomaly": True,
    "anomaly_type": "point|range|freq|trend|none",
    "start_idx": 81,
    "end_idx": 96,
    "rationale": "short rationale",
}
```

#### Cell 5.3 生成原始教师标注

```python
import base64
import json
import os

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

train_df = pd.read_csv(RT_DATA / "train.csv")
val_df = pd.read_csv(RT_DATA / "val.csv")
dev_test_df = pd.read_csv(RT_DATA / "dev_test.csv")
sft_df = pd.concat([train_df, val_df, dev_test_df], ignore_index=True)

rows = []

for _, row in sft_df.iterrows():
    with open(row["image_path"], "rb") as file_obj:
        image_b64 = base64.b64encode(file_obj.read()).decode()

    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": USER_TEXT},
                ],
            },
        ],
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
        temperature=TEACHER_TEMPERATURE,
    )

    parsed = json.loads(response.choices[0].message.content)
    parsed["sample_id"] = row["sample_id"]
    parsed["image_path"] = row["image_path"]
    parsed["ground_truth"] = row["label"]
    parsed["ground_truth_type"] = row["anomaly_type"]
    rows.append(parsed)

with open(RT_SFT / "sft_raw.jsonl", "w") as file_obj:
    for record in rows:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
```

#### Cell 5.4 同步教师原始标注

```python
sync_file(RT_SFT / "sft_raw.jsonl", DRV_SFT / "sft_raw.jsonl")
```

### 阶段 6. 清洗 SFT 数据

这部分保持自定义。目标是把教师输出清洗成可控的监督信号，而不是让 `AnomLLM` 或 `Unsloth` 负责数据质量兜底。

#### Cell 6.1 自动清洗

```python
import json

clean_rows = []
reject_rows = []

with open(RT_SFT / "sft_raw.jsonl", "r") as file_obj:
    for line in file_obj:
        row = json.loads(line)
        pred = row["is_anomaly"]
        gt = bool(row["ground_truth"])
        if pred == gt:
            clean_rows.append(row)
        else:
            reject_rows.append(row)

with open(RT_SFT / "sft_clean.jsonl", "w") as file_obj:
    for row in clean_rows:
        file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")

with open(RT_SFT / "sft_reject.jsonl", "w") as file_obj:
    for row in reject_rows:
        file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
```

#### Cell 6.2 人工抽检与最终版落盘

保留一个 Markdown 单元，明确抽检要求：

- 从 `sft_clean.jsonl` 抽样检查。
- 人工输出 `sft_reject_manual.jsonl`。
- 最终确认版写为 `sft_final.jsonl`。

#### Cell 6.3 同步清洗结果

```python
for name in [
    "sft_clean.jsonl",
    "sft_reject.jsonl",
    "sft_reject_manual.jsonl",
    "sft_final.jsonl",
]:
    sync_file(RT_SFT / name, DRV_SFT / name)
```

### 阶段 7. 转换为 Unsloth 训练格式

这部分仍然是自定义补充，但输出格式要主动对齐 `Unsloth Vision Fine-tuning` 的 `messages` / conversation 样式。

#### Cell 7.1 生成训练 JSONL

```python
record = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/content/tsad_runtime/images/sft/example.png"},
                {"type": "text", "text": "Detect anomalies in this time-series plot."},
            ],
        },
        {
            "role": "assistant",
            "content": "<summary>...</summary><label>ANOMALY</label>",
        },
    ]
}
```

说明：

- `train.jsonl`、`val.jsonl`、`dev_test.jsonl` 是本项目的持久化格式。
- 真正的训练范式仍然以 `Unsloth` 官方 Vision SFT 为主，不把这层 JSONL 持久化格式误写成官方训练实现本身。

#### Cell 7.2 按 split 生成训练 JSONL

```python
import json

split_map = {}
for split_name in ["train", "val", "dev_test"]:
    split_df = pd.read_csv(RT_DATA / f"{split_name}.csv")
    for _, row in split_df.iterrows():
        split_map[row["sample_id"]] = split_name

writers = {
    "train": open(RT_SFT / "train.jsonl", "w"),
    "val": open(RT_SFT / "val.jsonl", "w"),
    "dev_test": open(RT_SFT / "dev_test.jsonl", "w"),
}

with open(RT_SFT / "sft_final.jsonl", "r") as file_obj:
    for line in file_obj:
        row = json.loads(line)
        label_text = "ANOMALY" if row["is_anomaly"] else "NORMAL"
        region_text = (
            f"[{row['start_idx']},{row['end_idx']}]"
            if row["start_idx"] is not None and row["end_idx"] is not None
            else "null"
        )
        assistant_text = (
            f"<summary>{row['summary']}</summary>"
            f"<label>{label_text}</label>"
            f"<type>{row['anomaly_type']}</type>"
            f"<region>{region_text}</region>"
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
                {"role": "assistant", "content": assistant_text},
            ]
        }

        split_name = split_map[row["sample_id"]]
        writers[split_name].write(json.dumps(record, ensure_ascii=False) + "\n")

for writer in writers.values():
    writer.close()
```

#### Cell 7.3 同步并打包 SFT 数据

```python
for name in ["train.jsonl", "val.jsonl", "dev_test.jsonl", "sft_final.jsonl"]:
    sync_file(RT_SFT / name, DRV_SFT / name)

pack_dir(RT_SFT, DRV_PACK / "sft_bundle.tar")
```

### 阶段 8. 执行 Unsloth Vision QLoRA SFT

这一段尽量贴近 `Unsloth` 官方 Vision fine-tuning 和 `Qwen3-VL` notebook。

#### Cell 8.1 恢复本地图像与训练数据

```bash
mkdir -p /content/tsad_runtime/images /content/tsad_runtime
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/images_sft.tar -C /content/tsad_runtime/images
tar -xf /content/drive/MyDrive/tsad_anomaly/packs/sft_bundle.tar -C /content/tsad_runtime
```

#### Cell 8.2 加载模型并按 Unsloth 官方方式挂 LoRA

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
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules="all-linear",
)

train_dataset = load_dataset("json", data_files=str(RT_SFT / "train.jsonl"), split="train")
eval_dataset = load_dataset("json", data_files=str(RT_SFT / "val.jsonl"), split="train")
```

#### Cell 8.3 配置 Unsloth Vision SFTTrainer

```python
from trl import SFTConfig, SFTTrainer
from unsloth.trainer import UnslothVisionDataCollator

sft_args = SFTConfig(
    output_dir=str(RT_CKPT / "qwen3vl-tsad"),
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
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
)
```

#### Cell 8.4 启动训练并导出模型

```python
trainer.train()

model.save_pretrained_merged(RT_SFT / "qwen3vl-tsad-merged", tokenizer)
model.save_pretrained(RT_SFT / "qwen3vl-tsad-adapter")
```

#### 阶段 8 约束

- `messages` / conversation 格式保持不变。
- `max_length=None` 保持不变。
- 默认采用 `Unsloth` 官方 Vision 配置：`target_modules="all-linear"`，并同时打开 vision、language、attention、MLP 层。
- `TRL` 在这里是底层组件，不作为本阶段主方法来源。

### 阶段 9. 在 `dev_test` 上验证

这一阶段保持自定义评估。注意和阶段 4 的 `AnomLLM` 官方 baseline 分开解释：

- baseline 指标来自 `AnomLLM/results`。
- `dev_test` 指标来自本项目自定义 `dev_test.csv` 和结构化输出解析。

固定输出：

- `/content/tsad_runtime/results/dev_test_sft_predictions.csv`
- `/content/tsad_runtime/results/dev_test_metrics.csv`

### 阶段 10. 在 `final_holdout` 上执行最终评估

这一步也保持自定义评估，但比较对象必须清楚区分来源：

- `AnomLLM official baseline`
- `Qwen3-VL + Unsloth SFT`
- `Qwen3-VL + Unsloth SFT + GRPO`

两类结果用途不同，不直接混成同一套官方 benchmark。

### 阶段 11. 可选 Unsloth VLM RL / GRPO

这一段按 `Unsloth` 官方 VLM RL / GRPO 路线组织，reward 逻辑仍保留本项目自定义。

#### Cell 11.1 启用 vLLM fast inference 与 Standby

```python
import os

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
```

#### Cell 11.2 按 Unsloth 官方方式加载 GRPO 基座

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    fast_inference=True,
    gpu_memory_utilization=0.8,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules="all-linear",
)
```

说明：

- 默认使用 `GRPO`，不把 `GSPO` 作为主路径。
- 保留 `GSPO` 作为后续可选开关。
- 在 `fast_inference=True` 的 vLLM 路线下，默认 `finetune_vision_layers=False`，因为官方文档明确指出 vLLM fast inference 目前不支持 vision/encoder LoRA。

#### Cell 11.3 固定输出模板与自定义 reward

```python
GRPO_OUTPUT_TEMPLATE = """<summary>...</summary>
<label>ANOMALY|NORMAL</label>
<type>point|range|freq|trend|none</type>
<region>[start,end] or null</region>"""

GRPO_REWARD_WEIGHTS = {
    "format_correct": 0.2,
    "label_correct": 0.5,
    "type_correct": 0.2,
    "region_overlap": 0.1,
}
```

这里的 reward 设计是“套在 `Unsloth` 官方 GRPO 骨架上的任务特异逻辑”，不是重写 RL 训练主干。

## 7. 运行日志

```python
RUN_LOG_PATH = DRV_LOGS / "run_log.csv"
RUN_LOG_COLUMNS = [
    "run_id",
    "stage",
    "model_name",
    "split_version",
    "image_size",
    "min_pixels",
    "max_pixels",
    "lora_r",
    "lr",
    "batch_size",
    "grad_accum",
    "vision_layers",
    "checkpoint_name",
    "val_f1",
    "val_precision",
    "val_recall",
    "notes",
]
```

## 8. 最终交付检查

Drive 中至少应存在以下关键文件：

- `/content/drive/MyDrive/tsad_anomaly/packs/anomllm_raw_data.tar`
- `/content/drive/MyDrive/tsad_anomaly/packs/images_sft.tar`
- `/content/drive/MyDrive/tsad_anomaly/packs/images_eval.tar`
- `/content/drive/MyDrive/tsad_anomaly/packs/official_baseline.tar`
- `/content/drive/MyDrive/tsad_anomaly/packs/sft_bundle.tar`
- `/content/drive/MyDrive/tsad_anomaly/sft/sft_raw.jsonl`
- `/content/drive/MyDrive/tsad_anomaly/sft/sft_final.jsonl`
- `/content/drive/MyDrive/tsad_anomaly/models/qwen3vl-tsad-merged.tar`
- `/content/drive/MyDrive/tsad_anomaly/models/qwen3vl-tsad-adapter.tar`
- `/content/drive/MyDrive/tsad_anomaly/results/dev_test_metrics.csv`
- `/content/drive/MyDrive/tsad_anomaly/results/final_summary.csv`
- `/content/drive/MyDrive/tsad_anomaly/logs/run_log.csv`

## 9. 当前文档中的有意留白

以下内容仍然需要你结合 `AnomLLM` 实际数据结构或任务定义补齐：

- `master_manifest.csv` 的构造逻辑。
- 教师标注 prompt 与清洗细则。
- `dev_test` 和 `final_holdout` 的具体推理解析与评估实现。

这些留白是有意保留的：

- baseline 已尽量按 `AnomLLM` 官方 git 方法固定。
- `SFT` 和 `GRPO` 已尽量按 `Unsloth` 官方方法固定。
- 只把官方没有直接覆盖的桥接环节留给本项目自定义代码。
