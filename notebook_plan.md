# Colab Notebook 执行计划

## 参考链接

- AnomLLM: https://github.com/Rose-STL-Lab/AnomLLM
- Unsloth Vision SFT: https://docs.unsloth.ai/basics/vision-fine-tuning
- Unsloth Qwen3-VL: https://docs.unsloth.ai/models/qwen3-vl
- Unsloth VLM RL: https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl
- vLLM Server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

## 数据结构（已确认）

```
AnomLLM/
  data/synthetic/<subset>/
    eval/
      data.pkl          # {'series': [ndarray(1000,1), ...×400], 'anom': [[[start,end],...], ...×400]}
      figs/001.png ... 400.png   # 对应 series[0]~series[399] 的折线图
    train/
      data.pkl          # 同格式，seed 不同
  results/synthetic/<subset>/<model>/
    <variant>.jsonl     # {'custom_id':..., 'request':..., 'response':'[{"start":0,"end":50},...]'}
  results/agg/<subset>.pkl       # result_agg.py 输出
```

子集：`range / point / freq / trend / flat-trend / noisy-point / noisy-freq / noisy-trend`

每个 series：长度 1000，单通道，label = `anom[i][0]` 是 `[(start,end),...]` 列表（空列表=正常）。

图像：`figs/` 下已有预渲染 PNG（1-indexed），无需重新渲染。

---

## 阶段总览

| 阶段 | 目标 | 主要产物 |
| --- | --- | --- |
| 0 | 初始化环境 | 运行目录、依赖 |
| 1 | 获取 AnomLLM 代码和数据 | 仓库、data.pkl × 子集 |
| 2 | VLM baseline | `results/synthetic/<subset>/qwen-local/*.jsonl` |
| 3 | 统计 baseline | `results/synthetic/<subset>/isolation-forest/0shot.jsonl` |
| 4 | 汇总 baseline 对比 | `results/agg/*.pkl`、`baseline_compare.csv` |
| 5 | 构建 SFT 样本清单 | `sft_manifest.csv`（含 train/val/eval split） |
| 6 | 教师蒸馏 | `sft_raw.jsonl` → `sft_final.jsonl` |
| 7 | 转成 Unsloth 训练格式 | `train.jsonl`、`val.jsonl`、`eval.jsonl` |
| 8 | SFT | adapter、merged model |
| 9 | 对比 SFT 与 baseline | `sft_eval_metrics.csv` |
| 10 | 可选 GRPO | grpo 模型与结果 |

---

## 阶段 0. 初始化环境

**Cell 0.1** 挂载 Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

**Cell 0.2** 定义运行目录

```python
from pathlib import Path

RUNTIME    = Path("/content/tsad_runtime")
DRIVE_ROOT = Path("/content/drive/MyDrive/tsad_anomaly")

RT_CODE    = RUNTIME / "code"
RT_SFT     = RUNTIME / "sft"
RT_CKPT    = RUNTIME / "checkpoints"
RT_RESULTS = RUNTIME / "results"

DRV_PACK   = DRIVE_ROOT / "packs"
DRV_SFT    = DRIVE_ROOT / "sft"
DRV_CKPT   = DRIVE_ROOT / "checkpoints"
DRV_RESULTS= DRIVE_ROOT / "results"

for p in [RUNTIME, RT_CODE, RT_SFT, RT_CKPT, RT_RESULTS,
          DRV_PACK, DRV_SFT, DRV_CKPT, DRV_RESULTS]:
    p.mkdir(parents=True, exist_ok=True)

ANOMLLM = RT_CODE / "AnomLLM"
```

**Cell 0.3** 安装依赖

```bash
pip install -U pip -q
pip install "unsloth[colab-new]" -q
pip install vllm openai accelerate bitsandbytes scikit-learn pandas pyyaml datasets matplotlib pillow trl -q
```

---

## 阶段 1. 获取 AnomLLM 代码和数据

**Cell 1.1** 克隆仓库

```bash
if [ ! -d /content/tsad_runtime/code/AnomLLM ]; then
  git clone https://github.com/Rose-STL-Lab/AnomLLM.git /content/tsad_runtime/code/AnomLLM
fi
export PYTHONPATH=/content/tsad_runtime/code/AnomLLM/src
```

**Cell 1.2** 恢复或下载数据

```bash
# 优先从 Drive 恢复
if [ -f /content/drive/MyDrive/tsad_anomaly/packs/anomllm_data.tar ]; then
  tar -xf /content/drive/MyDrive/tsad_anomaly/packs/anomllm_data.tar \
      -C /content/tsad_runtime/code/AnomLLM
else
  # 方案 A：从 S3 下载预生成数据
  pip install s5cmd -q
  cd /content/tsad_runtime/code/AnomLLM
  s5cmd --no-sign-request --endpoint-url https://s3-west.nrp-nautilus.io \
      cp "s3://anomllm/data/*" data/

  # 方案 B：本地生成（若 S3 不可用）
  # cd /content/tsad_runtime/code/AnomLLM && bash synthesize.sh

  # 打包备份到 Drive
  tar -cf /content/drive/MyDrive/tsad_anomaly/packs/anomllm_data.tar \
      -C /content/tsad_runtime/code/AnomLLM data
fi
```

**Cell 1.3** 确认数据结构

```python
import pickle, os

for subset in ["point", "range", "freq", "flat-trend"]:
    pkl = f"/content/tsad_runtime/code/AnomLLM/data/synthetic/{subset}/eval/data.pkl"
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    print(subset, "series:", len(d["series"]), d["series"][0].shape,
          "| has_figs:", os.path.isdir(pkl.replace("data.pkl","figs")))
```

**Cell 1.4** 固定先跑的数据子集

```python
DATASETS = ["flat-trend", "range", "point", "freq"]
```

---

## 阶段 2. VLM Baseline

**Cell 2.1** 启动本地 Qwen3-VL endpoint（后台运行）

`--served-model-name qwen-local` 必须加：`online_api.py` 把 `--model` 参数直接发给 API，vLLM 只认注册名。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen-local \
  --host 127.0.0.1 --port 8000 &
```

**Cell 2.2** 写 `credentials.yml`（脚本从 AnomLLM 根目录读取）

```python
creds = """\
qwen-local:
  api_key: dummy
  base_url: "http://127.0.0.1:8000/v1"
"""
(ANOMLLM / "credentials.yml").write_text(creds)
```

**Cell 2.3** 跑 VLM baseline（AnomLLM 官方脚本）

```bash
cd /content/tsad_runtime/code/AnomLLM
for datum in flat-trend range point freq; do
  python src/online_api.py --data "$datum" --model qwen-local --variant 0shot-vision
done
```

输出：`results/synthetic/<datum>/qwen-local/0shot-vision.jsonl`
格式：每行 `{"custom_id":..., "request":..., "response":"[{\"start\":0,\"end\":50},...]"}`

---

## 阶段 3. 统计 Baseline

**Cell 3.1** 跑 isolation forest（AnomLLM 官方脚本）

```bash
cd /content/tsad_runtime/code/AnomLLM
for datum in flat-trend range point freq; do
  python src/baselines/isoforest.py --data "$datum" --model isolation-forest
done
```

输出：`results/synthetic/<datum>/isolation-forest/0shot.jsonl`

---

## 阶段 4. 汇总 Baseline 对比

**Cell 4.1** 用官方 result_agg.py 生成每个子集的指标

```bash
cd /content/tsad_runtime/code/AnomLLM
mkdir -p results/agg

python src/result_agg.py --data_name flat-trend --label_name flat-trend-exp \
    --table_caption "Trend anomalies (flat)"
python src/result_agg.py --data_name range --label_name range-exp \
    --table_caption "Out-of-range anomalies"
python src/result_agg.py --data_name point --label_name point-exp \
    --table_caption "Point anomalies"
python src/result_agg.py --data_name freq --label_name freq-exp \
    --table_caption "Frequency anomalies"
```

输出：`results/agg/<datum>.pkl`，每个 pkl 是 pandas DataFrame（index=model+variant，columns=precision/recall/f1/affi-*）

**Cell 4.2** 合并成对比 CSV

```python
import pickle, pandas as pd

frames = []
for datum in ["flat-trend", "range", "point", "freq"]:
    with open(f"/content/tsad_runtime/code/AnomLLM/results/agg/{datum}.pkl", "rb") as f:
        df = pickle.load(f)
    df.insert(0, "dataset", datum)
    frames.append(df)

pd.concat(frames).to_csv("/content/tsad_runtime/results/baseline_compare.csv")
```

---

## 阶段 5. 构建 SFT 样本清单

数据来源：已加载的 `SyntheticDataset`，无需重新渲染图像（`figs/` 已有 PNG）。

**Cell 5.1** 生成 `sft_manifest.csv`

```python
import pickle, pandas as pd
from sklearn.model_selection import train_test_split

rows = []
for subset in ["flat-trend", "range", "point", "freq"]:
    pkl_path = f"/content/tsad_runtime/code/AnomLLM/data/synthetic/{subset}/eval/data.pkl"
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    anom_type = {"flat-trend": "trend", "range": "range",
                 "point": "point", "freq": "freq"}[subset]
    for i, anom_list in enumerate(d["anom"]):
        intervals = anom_list[0]   # sensor 0
        label = 1 if len(intervals) > 0 else 0
        rows.append({
            "sample_id":   f"{subset}_{i:03d}",
            "subset":      subset,
            "pkl_path":    pkl_path,
            "pkl_idx":     i,
            "image_path":  f"/content/tsad_runtime/code/AnomLLM/data/synthetic/{subset}/eval/figs/{i+1:03d}.png",
            "label":       label,
            "anomaly_type": anom_type if label else "none",
            "intervals":   str(intervals),
        })

manifest = pd.DataFrame(rows)

# stratified split: 1200 train / 150 val / 150 eval
train_df, tmp_df = train_test_split(
    manifest, test_size=0.20, random_state=3407,
    stratify=manifest["subset"].astype(str) + "_" + manifest["label"].astype(str)
)
val_df, eval_df = train_test_split(
    tmp_df, test_size=0.50, random_state=3407,
    stratify=tmp_df["subset"].astype(str) + "_" + tmp_df["label"].astype(str)
)
train_df = train_df.assign(split="train")
val_df   = val_df.assign(split="val")
eval_df  = eval_df.assign(split="eval")

manifest = pd.concat([train_df, val_df, eval_df]).sort_values("sample_id")
manifest.to_csv("/content/tsad_runtime/sft/sft_manifest.csv", index=False)
print(manifest["split"].value_counts())
print(manifest.groupby(["split","label"]).size())
```

---

## 阶段 6. 教师蒸馏与清洗

**Cell 6.1** 设置教师模型

```python
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-xxxx"

TEACHER_MODEL = "qwen3.5-plus"
TEACHER_TEMPERATURE = 0.2
```

**Cell 6.2** 生成蒸馏数据（只对 train + val 调用教师）

```python
import base64, json, pandas as pd
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM = "You are annotating time-series anomaly plots for supervised fine-tuning. Return strict JSON only."
# intervals 与 AnomLLM 输出格式对齐，支持多段异常区间
USER   = ("Analyze this time-series plot. Return JSON with keys: "
          "is_anomaly (bool), anomaly_type (point|range|freq|trend|none), "
          "intervals ([{\"start\":int,\"end\":int}, ...] or []), rationale (str). "
          "No markdown.")

manifest = pd.read_csv("/content/tsad_runtime/sft/sft_manifest.csv")
sft_df   = manifest[manifest["split"].isin(["train", "val"])]

out_path = "/content/tsad_runtime/sft/sft_raw.jsonl"
with open(out_path, "a") as fout:
    for _, row in sft_df.iterrows():
        with open(row["image_path"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        resp = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": USER},
                ]},
            ],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
            temperature=TEACHER_TEMPERATURE,
        )
        result = json.loads(resp.choices[0].message.content)
        result["sample_id"]      = row["sample_id"]
        result["image_path"]     = row["image_path"]
        result["ground_truth"]   = int(row["label"])
        result["split"]          = row["split"]
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
```

**Cell 6.3** 自动清洗（label 一致 + intervals 格式合法才保留）

```python
import json

clean, reject = [], []
with open("/content/tsad_runtime/sft/sft_raw.jsonl") as f:
    for line in f:
        r = json.loads(line)
        teacher_anom = bool(r.get("is_anomaly", len(r.get("intervals", [])) > 0))
        ground_truth = bool(r["ground_truth"])
        intervals_ok = isinstance(r.get("intervals"), list)
        if teacher_anom == ground_truth and intervals_ok:
            clean.append(r)
        else:
            reject.append(r)

with open("/content/tsad_runtime/sft/sft_final.jsonl", "w") as f:
    for r in clean:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"clean={len(clean)}, reject={len(reject)}, "
      f"keep_rate={len(clean)/(len(clean)+len(reject)):.1%}")
```

---

## 阶段 7. 转成 Unsloth 训练格式

**Cell 7.1** 生成 messages 格式 JSONL

```python
import json

USER_TEXT = "Detect anomalies in this time-series plot."

writers = {
    "train": open("/content/tsad_runtime/sft/train.jsonl", "w"),
    "val":   open("/content/tsad_runtime/sft/val.jsonl",   "w"),
}

with open("/content/tsad_runtime/sft/sft_final.jsonl") as f:
    for line in f:
        r = json.loads(line)
        label_str  = "ANOMALY" if r["is_anomaly"] else "NORMAL"
        # intervals 与 AnomLLM 格式一致：[{"start":..., "end":...}, ...]
        region_str = json.dumps(r.get("intervals", []))
        assistant  = (f"<label>{label_str}</label>\n"
                      f"<type>{r['anomaly_type']}</type>\n"
                      f"<region>{region_str}</region>\n"
                      f"<rationale>{r['rationale']}</rationale>")
        record = {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": r["image_path"]},
                {"type": "text",  "text":  USER_TEXT},
            ]},
            {"role": "assistant", "content": assistant},
        ]}
        writers[r["split"]].write(json.dumps(record, ensure_ascii=False) + "\n")

for f in writers.values():
    f.close()
```

**Cell 7.2** eval split 单独转（只含 user turn，用于推理）

```python
import json, pandas as pd

manifest = pd.read_csv("/content/tsad_runtime/sft/sft_manifest.csv")
eval_rows = manifest[manifest["split"] == "eval"]

with open("/content/tsad_runtime/sft/eval.jsonl", "w") as f:
    for _, row in eval_rows.iterrows():
        record = {
            "sample_id":   row["sample_id"],
            "image_path":  row["image_path"],
            "label":       int(row["label"]),
            "anomaly_type": row["anomaly_type"],
            "intervals":   row["intervals"],
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

---

## 阶段 8. SFT（Unsloth 官方范式）

**Cell 8.1** 加载模型

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
val_dataset   = load_dataset("json", data_files=str(RT_SFT / "val.jsonl"),   split="train")
```

**Cell 8.2** 训练

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
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    optim="adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
)

trainer.train()
```

**Cell 8.3** 导出模型

```python
model.save_pretrained_merged(RT_SFT / "qwen3vl-tsad-merged", tokenizer)
model.save_pretrained(RT_SFT / "qwen3vl-tsad-adapter")
```

---

## 阶段 9. 对比 SFT 与 Baseline

baseline 在 400 条上算过指标，SFT eval 只有 150 条（跨 4 个子集）。
**必须把 baseline 结果过滤到相同的 150 条**，才能公平对比。

**Cell 9.1** 启动 SFT 模型 vLLM server 并推理

```bash
# 用与 baseline 完全相同的调用方式：online_api.py + credentials.yml
python -m vllm.entrypoints.openai.api_server \
  --model /content/tsad_runtime/sft/qwen3vl-tsad-merged \
  --served-model-name sft-model \
  --host 127.0.0.1 --port 8001 &
```

```python
# 追加 sft-model 到 credentials.yml
with open(ANOMLLM / "credentials.yml", "a") as f:
    f.write("\nsft-model:\n  api_key: dummy\n  base_url: \"http://127.0.0.1:8001/v1\"\n")
```

```bash
# 对全部 4 个子集跑推理（结果保存到 results/synthetic/<datum>/sft-model/）
cd /content/tsad_runtime/code/AnomLLM
for datum in flat-trend range point freq; do
  python src/online_api.py --data "$datum" --model sft-model --variant 0shot-vision
done
```

**Cell 9.2** 在相同的 eval 150 条上计算 SFT 与 baseline 指标

```python
import json, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/content/tsad_runtime/code/AnomLLM/src")
from utils import compute_metrics, interval_to_vector, load_results

# 1. 读取 eval split，建立 {subset: [pkl_idx, ...]} 映射
eval_manifest = pd.read_csv("/content/tsad_runtime/sft/sft_manifest.csv")
eval_manifest = eval_manifest[eval_manifest["split"] == "eval"]
eval_indices = {}
for subset, grp in eval_manifest.groupby("subset"):
    eval_indices[subset] = grp["pkl_idx"].tolist()   # 0-indexed

# 2. 加载 eval 数据集的 ground truth
gt_map = {}   # (subset, pkl_idx) -> 0/1 label vector (length 1000)
for subset in ["flat-trend", "range", "point", "freq"]:
    with open(f"/content/tsad_runtime/code/AnomLLM/data/synthetic/{subset}/eval/data.pkl","rb") as f:
        d = pickle.load(f)
    for idx in eval_indices.get(subset, []):
        intervals = d["anom"][idx][0]   # sensor 0
        gt_map[(subset, idx)] = interval_to_vector(
            [{"start": s, "end": e} for s, e in intervals]
        ).flatten()

# 3. 对每个方法提取 eval 样本的预测，计算指标
#    online_api.py 的 custom_id 格式：{data}_{model}_{variant}_{i+1:05d}
#    即 pkl_idx=k 对应 custom_id 末尾 _{k+1:05d}

def eval_metrics(method_label, result_fn, eval_indices_map, gt_map):
    results_raw = load_results(result_fn, raw=False)  # list, index=pkl_idx (0-based)
    rows = []
    for subset, idxs in eval_indices_map.items():
        for idx in idxs:
            pred = results_raw[idx]
            gt   = gt_map[(subset, idx)]
            if pred is None:
                pred = np.zeros_like(gt)
            m = compute_metrics(gt.reshape(-1,1), pred.reshape(-1,1).astype(int))
            rows.append({"method": method_label, "subset": subset, **m})
    return pd.DataFrame(rows)

BASE_DIR = "/content/tsad_runtime/code/AnomLLM/results/synthetic"
all_frames = []
for subset in ["flat-trend", "range", "point", "freq"]:
    for method, fn in [
        ("isolation-forest", f"{BASE_DIR}/{subset}/isolation-forest/0shot.jsonl"),
        ("qwen-local-0shot",  f"{BASE_DIR}/{subset}/qwen-local/0shot-vision.jsonl"),
        ("sft-0shot",         f"{BASE_DIR}/{subset}/sft-model/0shot-vision.jsonl"),
    ]:
        all_frames.append(eval_metrics(method, fn, {subset: eval_indices[subset]}, gt_map))

result_df = pd.concat(all_frames)
summary = result_df.groupby("method")[["f1", "affi f1"]].mean().round(3)
print(summary)
summary.to_csv("/content/tsad_runtime/results/sft_eval_metrics.csv")
```

---

## 阶段 10. 可选 GRPO

条件：baseline 对比完成 + SFT 有可见提升 + 蒸馏数据质量稳定。

沿用 Unsloth 官方 VLM RL/GRPO 范式，同一份 `train/val/eval` 数据。
