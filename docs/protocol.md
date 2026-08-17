# TSAD v2 协议与数据格式说明

本文档描述 v2 全流程统一使用的异常区间协议、JSONL 数据格式、训练约束和评估/奖励口径。

---

## 1. 异常区间协议

- 使用半开区间 `[start, end)`。
- 单点异常表示为 `[i, i+1)`。
- 多个异常按时间顺序输出。
- 无异常输出 `[]`。
- 所有标签、模型输出、预测和评估均使用同一协议。

解析规则：

- 模型输出允许包含前后解释文字，但必须包含一个合法 JSON 数组。
- 数组元素只接受 `{"start": N, "end": M}`。
- `start` 和 `end` 必须是整数；浮点整数会被接受并转成 int。
- 必须满足 `start < end`。
- 不接受多余字段，例如 `{"start":1,"end":2,"label":"x"}`。
- 越界区间会被 clip 到当前窗口范围；clip 后若 `start >= end` 则判定非法。

---

## 2. 合成数据格式

生成命令：

```bash
tsad-v2 --config configs/base.yaml generate-synthetic
```

输出：

- `data/processed/synthetic/train.jsonl`
- `data/processed/synthetic/val.jsonl`
- `data/processed/synthetic/images/{train,val}/*.png`
- `data/processed/synthetic/series/{train,val}/*.npy`

`train.jsonl` / `val.jsonl` 每条记录：

```json
{
  "sample_id": "syn_train_00000",
  "series_id": "syn_train_00000",
  "source": "synthetic",
  "split": "train",
  "length": 1000,
  "window_start": 0,
  "window_end": 1000,
  "series_path": "/absolute/path/series.npy",
  "image_path": "/absolute/path/image.png",
  "intervals": [{"start": 120, "end": 145}],
  "anomaly_types": ["range"],
  "generation_seed": 3407,
  "generation_parameters": []
}
```

字段说明：

- `intervals` 始终是半开区间列表；无异常为 `[]`。
- `anomaly_types` 与 `intervals` 一一对应。
- `split` 只允许 `train` 或 `val`。
- `source` 固定为 `synthetic`。

---

## 3. UCR 数据格式

### 3.1 原始文件名

```text
001_UCR_Anomaly_DISTORTED_demo_name_5_10_12.txt
```

解析规则：

- `001`：archive id
- `DISTORTED_demo_name`：数据集名称
- `5`：train end
- `10`：anomaly start
- `12`：anomaly end

转换配置：

```yaml
ucr:
  filename_index_base: 0
  anomaly_end_inclusive: true
  train_end_is_count: true
```

默认转换：

- `start = anomaly_start_raw - index_base`
- `end = anomaly_end_raw - index_base + 1`（因为默认 inclusive）
- `test_start = train_end_raw`（因为默认是 count）

### 3.2 `series.jsonl`

每条 UCR 序列一条：

```json
{
  "sample_id": "ucr_001",
  "series_id": "ucr_001",
  "source": "ucr",
  "split": "test",
  "name": "DISTORTED_demo_name",
  "series_path": "/path/to/001_UCR_Anomaly_....txt",
  "length": 1000,
  "eval_start": 200,
  "eval_end": 1000,
  "intervals": [{"start": 500, "end": 520}],
  "ucr_metadata": {}
}
```

`series.jsonl` 是评估单元，不对整条序列做 VLM 推理，只用于汇总窗口预测。

### 3.3 `windows.jsonl`

每个推理窗口一条：

```json
{
  "sample_id": "ucr_001_w0000",
  "series_id": "ucr_001",
  "source": "ucr",
  "split": "test",
  "image_path": "/path/to/image.png",
  "window_start": 200,
  "window_end": 700,
  "length": 500,
  "intervals": [{"start": 500, "end": 520}]
}
```

窗口标签已 clip 到 `[window_start, window_end)`。

---

## 4. 预测文件格式

`infer` 输出 `predictions.jsonl`，每条窗口预测：

```json
{
  "sample_id": "ucr_001_w0000",
  "series_id": "ucr_001",
  "window_start": 200,
  "window_end": 700,
  "raw_output": "[{\"start\":500,\"end\":520}]",
  "intervals": [{"start": 500, "end": 520}],
  "parse_valid": true,
  "parse_error": null,
  "latency_seconds": 1.23,
  "model": "Qwen/Qwen2.5-VL-3B-Instruct"
}
```

- `parse_valid=false` 表示模型输出无法解析成合法区间。
- `evaluate` 会按 `series_id` 合并同一序列的所有窗口预测，再与整条 GT 比较。

---

## 5. 评估输出格式

`evaluate` 输出：

```text
output_dir/
├── summary.json
└── per_sample.jsonl
```

`summary.json` 字段：

```json
{
  "samples": 1,
  "parse_rate": 1.0,
  "point_precision": 1.0,
  "point_recall": 1.0,
  "point_f1": 1.0,
  "event_precision": 1.0,
  "event_recall": 1.0,
  "event_f1": 1.0,
  "mean_matched_iou": 1.0,
  "boundary_mae": 0.0,
  "normal_accuracy": null,
  "point_tp": 0,
  "point_fp": 0,
  "point_fn": 0,
  "event_tp": 0,
  "event_fp": 0,
  "event_fn": 0
}
```

指标定义：

- `point_*`：把区间展开为逐点 0/1 后计算。
- `event_*`：预测区间与 GT 区间按 IoU 做一对一无重复匹配。
- `mean_matched_iou`：所有匹配对的 IoU 均值。
- `boundary_mae`：匹配对的左右边界绝对误差均值。
- `normal_accuracy`：GT 无异常的样本中，预测也为空的占比；无此类样本时为 `null`。
- `parse_rate`：预测文件中 `parse_valid=true` 的窗口占比。

---

## 6. 训练数据约束

`load_training_manifest` 会强制检查：

- `source == "synthetic"`
- `split in {"train", "val"}`
- `sample_id` 非空且不重复
- `image_path` 存在
- `intervals` 可解析且位于 `[0, length)`

因此：

- UCR 数据不能作为 SFT/RL 训练集。
- UCR 不会参与 checkpoint 选择。
- 合成验证集 `val.jsonl` 只用于 SFT 的 early stopping / best model 选择。

---

## 7. RL Reward 口径

GRPO 使用两个 reward：

### 7.1 `format_reward_func`

- 输出能被解析为合法区间列表：`1.0`
- 否则：`0.0`

### 7.2 `quality_reward_func`

对合法输出计算区间质量：

```text
reward = 0.10
       + 0.40 * point_f1
       + 0.25 * event_f1
       + 0.15 * mean_matched_iou
       + 0.05 * count_score
       + 0.05 * boundary_score
```

其中：

- `count_score = 1 - |pred_count - gt_count| / max(1, pred_count, gt_count)`
- `boundary_score` 由匹配区间的 boundary MAE 映射到 `[0,1]`
- 非法输出 reward 为 `0.0`

---

## 8. 来源说明

- UCR Anomaly Archive 与候选数据集资源入口见 [`v2_resources.md`](../v2_resources.md)。
- 本仓库只实现项目协议、数据、reward 和评估；训练主流程基于 Transformers / PEFT / TRL。
