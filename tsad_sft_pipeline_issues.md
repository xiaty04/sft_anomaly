# `tsad_sft_pipeline.ipynb` 修改方案

基于静态代码审查，对照 `AnomLLM/` 上游实现。

---

## 修改 1（必须）：对齐 SFT 输出格式与 AnomLLM 解析协议

**问题**：训练 assistant 输出为 XML 风格（`<label>...<region>[...]</region>...`），但评估时 AnomLLM 的 `parse_output` 只接受纯 JSON `[{"start":..., "end":...}, ...]`，prompt 也要求纯 JSON。训练和评估的输入输出协议完全不一致，导致评估结果不可靠。

**涉及位置**：
- `tsad_sft_pipeline.ipynb` Cell 51 — assistant 输出构造
- `AnomLLM/src/prompt.py:8-13` — 上游 PROMPT 和输出模板
- `AnomLLM/src/utils.py:25` — `parse_output` 截取逻辑

**修改内容**：

Cell 51 中 assistant 输出从：

```python
assistant = (f"<label>{label_str}</label>\n"
             f"<type>{r['anomaly_type']}</type>\n"
             f"<region>{region_str}</region>\n"
             f"<rationale>{r['rationale']}</rationale>")
```

改为与 AnomLLM 一致的纯 JSON 区间列表：

```python
assistant = json.dumps(r.get("intervals", []))
```

如果希望保留 rationale 用于 CoT，可参照 `prompt.py` 中 `COT_ANSWER_TEMPLATE` 的格式：

```python
# CoT 版本（可选）
if r.get("intervals"):
    assistant = (f"Based on the plot, anomalies can be identified:\n"
                 f"```{json.dumps(r.get('intervals', []))}```\n"
                 f"{r['rationale']}")
else:
    assistant = "[]"
```

---

## 修改 2（必须）：对齐 SFT 输入格式与 AnomLLM 推理请求

**问题**：训练时 user message 的 content 顺序（先 image 后 text）和 prompt 文本都与 AnomLLM 推理时不一致。

**涉及位置**：
- `tsad_sft_pipeline.ipynb` Cell 51 — user message 构造
- `AnomLLM/src/prompt.py:203-208` — `create_vision_messages` 中先 text 后 image

**修改内容**：

Cell 51 中 user content 从：

```python
{"role": "user", "content": [
    {"type": "image", "image": r["image_path"]},
    {"type": "text",  "text":  USER_TEXT},
]}
```

改为先 text 后 image，且使用上游 PROMPT：

```python
USER_TEXT = ("Detect ranges of anomalies in this time series, "
             "in terms of the x-axis coordinate.\n"
             "List one by one, in JSON format. \n"
             "If there are no anomalies, answer with an empty list [].\n\n"
             "Output template:\n"
             '[{"start": ..., "end": ...}, {"start": ..., "end": ...}...]')

# ...

{"role": "user", "content": [
    {"type": "text",  "text":  USER_TEXT},
    {"type": "image", "image": r["image_path"]},
]}
```

Cell 53（eval split 转换）如果也含 user message，做同样修改。

---

## 修改 3（建议）：加强蒸馏清洗规则

**问题**：当前清洗只检查 `teacher_anom == ground_truth` 和 `intervals` 是 list，缺少对 intervals 内容的合法性校验。不会频繁触发（教师模型输出通常规范），但属于防御性缺失。

**涉及位置**：
- `tsad_sft_pipeline.ipynb` Cell 48 — 清洗逻辑

**修改内容**：

Cell 48 中清洗条件从：

```python
teacher_anom = bool(r.get("is_anomaly", len(r.get("intervals", [])) > 0))
ground_truth = bool(r["ground_truth"])
intervals_ok = isinstance(r.get("intervals"), list)
if teacher_anom == ground_truth and intervals_ok:
```

改为：

```python
teacher_anom = bool(r.get("is_anomaly", False))
ground_truth = bool(r["ground_truth"])
intervals = r.get("intervals", [])

intervals_ok = isinstance(intervals, list) and all(
    isinstance(iv, dict) and
    isinstance(iv.get("start"), (int, float)) and
    isinstance(iv.get("end"), (int, float)) and
    iv["start"] < iv["end"]
    for iv in intervals
)
# is_anomaly 与 intervals 不能自相矛盾
consistent = (teacher_anom == bool(intervals)) if intervals_ok else False

if teacher_anom == ground_truth and intervals_ok and consistent:
```

---

## 修改 4（建议）：蒸馏输出改为覆盖写 + 去重

**问题**：`sft_raw.jsonl` 使用 append 模式，先跑冒烟再跑全量时前 20 条会重复。

**涉及位置**：
- `tsad_sft_pipeline.ipynb` Cell 46 — `open(out_path, "a")`

**修改内容**（二选一）：

方案 A — 直接改为覆盖写：
```python
with open(out_path, "w") as fout:   # "a" → "w"
```

方案 B — 保留 append 但在清洗阶段去重（更安全，支持断点续跑）：
```python
# Cell 48 清洗开头增加：
seen = set()
# 在循环内：
if r["sample_id"] in seen:
    continue
seen.add(r["sample_id"])
```

---

## 修改 5（可选）：credentials.yml 改为覆盖写

**问题**：Cell 66 用 append 追加 `sft-model`，多次执行会产生重复条目。实际上 YAML 重复 key 只保留最后一个值，不会报错，影响很小。

**涉及位置**：
- `tsad_sft_pipeline.ipynb` Cell 66

**修改内容**：

```python
import yaml

creds_path = ANOMLLM / "credentials.yml"
creds = yaml.safe_load(creds_path.read_text()) or {}
creds["sft-model"] = {"api_key": "dummy", "base_url": "http://127.0.0.1:8001/v1"}
creds_path.write_text(yaml.dump(creds))
```

---

## 不需要修改的点

| 项目 | 原因 |
|------|------|
| `anom_list[0]` / `sensor 0` | 上游生成固定 `number_of_sensors=1`，取 sensor 0 正确 |
| `.head(20)` / `max_steps=5` 冒烟配置 | 已有 `⚠️ 修改此处` 注释标明流程，属正常设计 |
| vLLM 启动后无 health check | `online_api.py` 已有 503 轮询 + 指数退避重试，实际可靠 |

---

## 修改优先级总览

| 优先级 | 编号 | 修改项 | 不修的后果 |
|--------|------|--------|-----------|
| **必须** | 1 | 输出格式对齐纯 JSON | 评估指标不可靠 |
| **必须** | 2 | 输入格式 + prompt 对齐 | 训练/评估分布不一致 |
| 建议 | 3 | 清洗规则加强 | 少量脏样本可能混入 |
| 建议 | 4 | 蒸馏文件去重 | 重跑时数据重复 |
| 可选 | 5 | credentials 覆盖写 | 几乎无影响 |
