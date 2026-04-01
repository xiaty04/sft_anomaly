# 使用方法说明

## 1. 这份文档解决什么问题

这份文档主要回答五个实际问题：

- `AnomLLM` 官方 baseline 现在是否已经跑通？
- 数据现在长什么样，怎么检查？
- `Unsloth SFT` 训练数据长什么样，怎么检查？
- 模型和结果文件写到哪里，怎么读？
- 本地 `/content` 和 Drive 目录分别存什么？

默认沿用当前方案中的目录约定，并区分三类产物：

- `AnomLLM` 官方 baseline 产物
- 本项目自定义数据桥接产物
- `Unsloth` 官方 `SFT/GRPO` 训练产物

## 2. 路径总览

### Colab 本地运行路径

| 路径 | 作用 |
| --- | --- |
| `/content/tsad_runtime/code/AnomLLM` | `AnomLLM` 官方 baseline 代码目录 |
| `/content/tsad_runtime/code/AnomLLM/results` | `AnomLLM` 官方 benchmark 结果目录 |
| `/content/tsad_runtime/data` | manifest、split CSV |
| `/content/tsad_runtime/images/sft` | `train/val/dev_test` 图像 |
| `/content/tsad_runtime/images/eval` | `final_holdout` 图像 |
| `/content/tsad_runtime/sft` | 教师标注、清洗结果、训练 JSONL、模型导出目录 |
| `/content/tsad_runtime/checkpoints` | SFT 或 GRPO 中间 checkpoint |
| `/content/tsad_runtime/results` | `dev_test` 和 `final_holdout` 自定义评估结果 |

### Google Drive 持久化路径

| 路径 | 作用 |
| --- | --- |
| `/content/drive/MyDrive/tsad_anomaly/raw` | CSV 元数据与 split |
| `/content/drive/MyDrive/tsad_anomaly/packs` | 原始数据包、图像包、官方 baseline 结果包、SFT 数据包 |
| `/content/drive/MyDrive/tsad_anomaly/sft` | 教师标注与最终训练数据 |
| `/content/drive/MyDrive/tsad_anomaly/models` | `Unsloth` 导出的 merged model、adapter、GRPO 模型包 |
| `/content/drive/MyDrive/tsad_anomaly/checkpoints` | SFT / GRPO checkpoint 归档 |
| `/content/drive/MyDrive/tsad_anomaly/results` | `dev_test` 与最终评估结果 |
| `/content/drive/MyDrive/tsad_anomaly/logs` | `run_log.csv` |

## 3. 如何查看官方 baseline 是否跑通

### 查看 `credentials.yml`

```python
from pathlib import Path

credentials_path = Path("/content/tsad_runtime/code/AnomLLM/credentials.yml")
print("exists:", credentials_path.exists())
if credentials_path.exists():
    print(credentials_path.read_text())
```

### 查看 `AnomLLM/results`

```python
from pathlib import Path

results_root = Path("/content/tsad_runtime/code/AnomLLM/results")
print("exists:", results_root.exists())
for p in sorted(results_root.glob("*"))[:20]:
    print(p.name)
```

### 查看 `official_baseline.tar`

```python
from pathlib import Path

baseline_tar = Path("/content/drive/MyDrive/tsad_anomaly/packs/official_baseline.tar")
print("exists:", baseline_tar.exists())
print("size_bytes:", baseline_tar.stat().st_size if baseline_tar.exists() else 0)
```

如果这三项都正常，说明 `AnomLLM` 官方 baseline 基本已经跑通并完成打包。

## 4. 如何查看数据形状和样子

### 查看 CSV 的行列数

```python
import pandas as pd

manifest = pd.read_csv("/content/tsad_runtime/data/master_manifest.csv")
train_df = pd.read_csv("/content/tsad_runtime/data/train.csv")
val_df = pd.read_csv("/content/tsad_runtime/data/val.csv")
dev_test_df = pd.read_csv("/content/tsad_runtime/data/dev_test.csv")
final_holdout_df = pd.read_csv("/content/tsad_runtime/data/final_holdout.csv")

print("manifest:", manifest.shape)
print("train:", train_df.shape)
print("val:", val_df.shape)
print("dev_test:", dev_test_df.shape)
print("final_holdout:", final_holdout_df.shape)
```

### 查看列名和前几行

```python
print(manifest.columns.tolist())
display(manifest.head(3))
display(train_df.head(3))
```

如果流程和当前方案一致，`master_manifest.csv` 至少应包含：

- `sample_id`
- `source_subset`
- `series_path`
- `image_path`
- `label`
- `anomaly_type`

### 查看标签分布

```python
print(train_df["label"].value_counts(dropna=False))
print(train_df["anomaly_type"].value_counts(dropna=False))
```

### 查看单个样本长什么样

```python
row = train_df.iloc[0]
print(row[["sample_id", "label", "anomaly_type", "image_path"]])
```

如果已经完成渲染，可以直接看图像：

```python
from PIL import Image
from IPython.display import display

img = Image.open(row["image_path"])
display(img)
```

## 5. 如何查看 SFT 数据

这部分是本项目自定义桥接产物，但目标格式要对齐 `Unsloth Vision Fine-tuning` 的 conversation / `messages` 样式。

### 查看教师原始标注条数

```python
from pathlib import Path

raw_path = Path("/content/tsad_runtime/sft/sft_raw.jsonl")
print("exists:", raw_path.exists())
print("size_bytes:", raw_path.stat().st_size if raw_path.exists() else 0)
```

### 查看清洗前后数量

```python
def count_jsonl(path):
    with open(path, "r") as f:
        return sum(1 for _ in f)

print("raw:", count_jsonl("/content/tsad_runtime/sft/sft_raw.jsonl"))
print("clean:", count_jsonl("/content/tsad_runtime/sft/sft_clean.jsonl"))
print("reject:", count_jsonl("/content/tsad_runtime/sft/sft_reject.jsonl"))
print("final:", count_jsonl("/content/tsad_runtime/sft/sft_final.jsonl"))
```

### 查看训练格式是否正确

```python
import json

with open("/content/tsad_runtime/sft/train.jsonl", "r") as f:
    first = json.loads(next(f))

print(first.keys())
print(first["messages"][0]["role"])
print(first["messages"][1]["role"])
print(first["messages"][0]["content"])
print(first["messages"][1]["content"])
```

这里重点看两件事：

- 数据是否采用 `messages` / conversation 结构。
- 第一条消息里是否同时包含图像和文本提示，这样才能直接喂给 `Unsloth Vision SFT`。

### 查看一批训练文件是否已生成

```python
from pathlib import Path

for name in ["train.jsonl", "val.jsonl", "dev_test.jsonl", "sft_final.jsonl"]:
    p = Path("/content/tsad_runtime/sft") / name
    print(name, p.exists(), p.stat().st_size if p.exists() else 0)
```

## 6. 如何查看训练与模型产物

### 查看 checkpoint 是否持续写出

```python
from pathlib import Path

ckpt_root = Path("/content/drive/MyDrive/tsad_anomaly/checkpoints/qwen3vl-tsad")
for p in sorted(ckpt_root.glob("checkpoint-*.tar"))[:10]:
    print(p.name)
```

### 查看最终模型是否导出

```python
from pathlib import Path

model_dir = Path("/content/drive/MyDrive/tsad_anomaly/models")
for p in sorted(model_dir.glob("*")):
    print(p.name)
```

通常应能看到：

- `qwen3vl-tsad-merged.tar`
- `qwen3vl-tsad-adapter.tar`

这里的含义要按 `Unsloth` 导出方式理解：

- `merged` 是合并后的可直接部署模型目录打包。
- `adapter` 是保留 LoRA adapter 的轻量导出。

### 查看 GRPO 相关产物

如果启用了 RL 阶段，再检查：

```python
from pathlib import Path

model_dir = Path("/content/drive/MyDrive/tsad_anomaly/models")
for p in sorted(model_dir.glob("*grpo*")):
    print(p.name)
```

这里的 GRPO 产物应理解为：

- 使用 `Unsloth VLM RL` 主干训练得到的模型或 adapter。
- 奖励函数是本项目自定义逻辑，但训练框架本身不是完全自写黑盒。

## 7. 如何查看结果文件

### 官方 baseline 结果

官方 baseline 结果看：

- `/content/tsad_runtime/code/AnomLLM/results`
- `/content/drive/MyDrive/tsad_anomaly/packs/official_baseline.tar`

### `dev_test` 验证结果

验证阶段主要看两个文件：

- `/content/tsad_runtime/results/dev_test_sft_predictions.csv`
- `/content/tsad_runtime/results/dev_test_metrics.csv`

读取方式：

```python
import pandas as pd

pred_df = pd.read_csv("/content/tsad_runtime/results/dev_test_sft_predictions.csv")
metrics_df = pd.read_csv("/content/tsad_runtime/results/dev_test_metrics.csv")

display(pred_df.head(5))
display(metrics_df)
```

### 最终评估结果

最终评估阶段主要看：

- `/content/tsad_runtime/results/final_holdout_<model_name>.csv`
- `/content/tsad_runtime/results/final_summary.csv`

读取方式：

```python
import pandas as pd

summary_df = pd.read_csv("/content/tsad_runtime/results/final_summary.csv")
display(summary_df)
```

注意这里的 `dev_test/final_holdout` 结果属于本项目自定义评估，不等同于 `AnomLLM` 官方 benchmark 输出。

## 8. 如何检查 Drive 同步是否成功

### 检查关键文件是否存在

```python
from pathlib import Path

must_exist = [
    "/content/drive/MyDrive/tsad_anomaly/packs/anomllm_raw_data.tar",
    "/content/drive/MyDrive/tsad_anomaly/packs/images_sft.tar",
    "/content/drive/MyDrive/tsad_anomaly/packs/images_eval.tar",
    "/content/drive/MyDrive/tsad_anomaly/packs/official_baseline.tar",
    "/content/drive/MyDrive/tsad_anomaly/packs/sft_bundle.tar",
    "/content/drive/MyDrive/tsad_anomaly/sft/sft_final.jsonl",
    "/content/drive/MyDrive/tsad_anomaly/models/qwen3vl-tsad-merged.tar",
    "/content/drive/MyDrive/tsad_anomaly/results/final_summary.csv",
    "/content/drive/MyDrive/tsad_anomaly/logs/run_log.csv",
]

for p in must_exist:
    print(p, Path(p).exists())
```

### 检查日志

```python
import pandas as pd

log_df = pd.read_csv("/content/drive/MyDrive/tsad_anomaly/logs/run_log.csv")
display(log_df.tail(10))
```

## 9. 常见查看动作

### 我只想确认官方 baseline 是否已经生成

看这些内容：

- `/content/tsad_runtime/code/AnomLLM/credentials.yml`
- `/content/tsad_runtime/code/AnomLLM/results`
- `/content/drive/MyDrive/tsad_anomaly/packs/official_baseline.tar`

### 我只想确认 split 是否已经生成

看这些文件是否存在：

- `/content/tsad_runtime/data/train.csv`
- `/content/tsad_runtime/data/val.csv`
- `/content/tsad_runtime/data/dev_test.csv`
- `/content/tsad_runtime/data/final_holdout.csv`

### 我只想确认图像是否已经准备好

看这些目录和打包文件：

- `/content/tsad_runtime/images/sft`
- `/content/tsad_runtime/images/eval`
- `/content/drive/MyDrive/tsad_anomaly/packs/images_sft.tar`
- `/content/drive/MyDrive/tsad_anomaly/packs/images_eval.tar`

### 我只想确认训练数据是否准备好

看这些文件：

- `/content/tsad_runtime/sft/train.jsonl`
- `/content/tsad_runtime/sft/val.jsonl`
- `/content/tsad_runtime/sft/dev_test.jsonl`
- `/content/drive/MyDrive/tsad_anomaly/packs/sft_bundle.tar`

### 我只想确认最终评估是否完成

看这些文件：

- `/content/tsad_runtime/results/final_summary.csv`
- `/content/drive/MyDrive/tsad_anomaly/results/final_summary.csv`

## 10. 推荐的检查顺序

每次重新打开 notebook，建议按这个顺序检查：

1. GPU 和依赖是否正常。
2. Drive 是否挂载成功。
3. `AnomLLM` 官方 baseline 是否已恢复。
4. split CSV 是否存在。
5. 图像包和 SFT 数据包是否存在。
6. `Unsloth SFT/GRPO` 模型和 checkpoint 是否存在。
7. `dev_test` 与 `final_holdout` 结果是否存在。
8. 日志是否持续更新。

这样可以快速判断是继续恢复执行，还是需要重跑上游阶段。
