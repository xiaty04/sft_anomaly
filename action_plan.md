# 行动执行计划

**节奏**：执行到断点 cell → 停下来看输出 → 把观察结果告诉 Claude Code → 生成下一阶段代码。

---

## 检查点 A：环境 + 数据就绪

**做了什么 / 产物在哪**
挂载 Drive，创建运行目录，安装全部依赖，克隆 AnomLLM 仓库，下载数据到 `data/synthetic/<subset>/eval/`。
每次新 Colab session 都要从这里开始重跑。

**执行**：Cell 0.1 → 0.2 → 0.3 → 1.1 → 1.2 → 1.3 → 1.4（⏸ BPA）

**结果在哪 / 怎么看**
```python
import torch
print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info()[0]/1e9, "GB free")
print("bf16:", torch.cuda.is_bf16_supported())
import unsloth  # 无报错即可
```
Cell 1.3 直接打印数据结构；目视一张 PNG：
```python
from IPython.display import Image
Image("data/synthetic/point/eval/figs/001.png")
```

**检查什么**
- GPU A100 / BF16 True / unsloth 无报错
- 四个子集各 400 条 shape `(1000,1)`，`has_figs: True`
- PNG 目视有折线

**下一步指导**
告诉 Claude Code：GPU 型号、可用 VRAM（GB）、BF16 是否 True、`has_figs` 是否全 True。
→ VRAM 影响 Cell 8.2 的 `per_device_train_batch_size`；`has_figs=False` 则 Cell 5.1 后需插入渲染 cell。

---

## 检查点 B：Baseline 完成

**做了什么 / 产物在哪**
启动本地 vLLM 服务，对全部 4 个子集各 400 条跑 VLM zero-shot 视觉推理，再用 isolation forest 跑统计 baseline，最后用 `result_agg.py` 汇总成 `baseline_compare.csv`。
产物：`results/synthetic/<subset>/qwen-local/0shot-vision.jsonl`、`/isolation-forest/0shot.jsonl`、`results/baseline_compare.csv`

**执行**：Cell 2.1 → 2.2 → 2.3 → 3.1 → 4.1 → 4.2（⏸ BPB）

> **内联冒烟**：Cell 2.3 开始跑后，等 `point` 子集的 jsonl 出现前 3 行，先执行 `head -3 results/synthetic/point/qwen-local/0shot-vision.jsonl`，确认 response 格式为 `[{"start":...,"end":...},...]`，无报错后再让全量继续。

**结果在哪 / 怎么看**
```bash
wc -l results/synthetic/*/qwen-local/0shot-vision.jsonl   # 各应为 400
wc -l results/synthetic/*/isolation-forest/0shot.jsonl
```
```python
import pandas as pd
print(pd.read_csv("/content/tsad_runtime/results/baseline_compare.csv").to_string())
```

**检查什么**
- 所有 jsonl 均 400 行（不足则中断，需续跑）
- `qwen-local (0shot-vision)` 各子集 f1 非零（期望 0.2~0.7）
- f1 全 0 说明 response 解析失败，把 CSV 头几行告诉 Claude Code

**下一步指导**
告诉 Claude Code：`baseline_compare.csv` 全文。
→ 这张表是最终 SFT 对比的基准线，Cell 9.2 会过滤出同一批 eval 150 条做公平对比。

---

## 检查点 C：SFT 数据就绪 + 训练冒烟

**做了什么 / 产物在哪**
依次完成四件事：
1. 生成样本清单（1600 条分层切分为 train/val/eval）→ `sft_manifest.csv`
2. 调用 DashScope 教师模型标注 train+val 共 1350 条，清洗 → `sft_final.jsonl`
3. 转 Unsloth messages 格式 → `train.jsonl`、`val.jsonl`、`eval.jsonl`
4. 加载 4bit 量化模型 + LoRA，`max_steps=5` 冒烟，确认不 OOM、loss 正常

**执行**：Cell 5.1 → 6.1 → 6.2（先 `.head(20)` 跑 20 条确认质量，再去掉限制跑全量） → 6.3 → 7.1 → 7.2 → 8.1 → 8.2（加 `max_steps=5`）（⏸ BPC）

**结果在哪 / 怎么看**
```python
# 蒸馏质量：前 5 条 + 保留率
import json
with open("/content/tsad_runtime/sft/sft_raw.jsonl") as f:
    for _ in range(5): print(json.dumps(json.loads(f.readline()), indent=2, ensure_ascii=False))
# Cell 6.3 自动打印: clean=NNN, reject=NNN, keep_rate=XX%

# 格式验证
import os
with open("/content/tsad_runtime/sft/train.jsonl") as f:
    r = json.loads(f.readline())
print("image exists:", os.path.exists(r["messages"][0]["content"][0]["image"]))
print(r["messages"][1]["content"][:80])  # 确认含 <label>

# 冒烟 loss（Trainer 日志直接打印）
```

**检查什么**
- 蒸馏：`is_anomaly` 与 `ground_truth` 一致率 ≥ 70%，保留率 ≥ 70%
- 格式：`image exists: True`，assistant content 含 `<label>ANOMALY</label>` 或 `<label>NORMAL</label>`
- 冒烟：无 OOM，第 1 步 loss 在 1.5~4（不是 nan 或 0）

**下一步指导**
告诉 Claude Code：蒸馏保留率、冒烟前 5 步 loss、是否有报错。
→ OOM → 调 `per_device_train_batch_size=1`；loss=nan → `learning_rate` 降至 `2e-5`；一致率 < 50% → 修改 User prompt 重跑蒸馏。
冒烟通过后去掉 `max_steps=5`，跑完整训练（Cell 8.2 正式版）→ Cell 8.3 导出。

---

## 检查点 D：最终对比

**做了什么 / 产物在哪**
用与 baseline 完全相同的推理方式（vLLM + `online_api.py`）对全部 4 个子集跑 SFT 模型推理，从三种方法的 jsonl 中各自过滤出相同的 eval 150 条，计算并对比 F1。
产物：`results/sft_eval_metrics.csv`

**执行**：Cell 9.1 → 9.2（⏸ BPD）

**结果在哪 / 怎么看**
Cell 9.2 直接打印：

| 方法 | avg F1 | avg affi F1 |
|------|--------|-------------|
| isolation-forest | ? | ? |
| qwen-local 0shot-vision | ? | ? |
| sft-0shot | ? | ? |

**检查什么**
- SFT 是否比 VLM zero-shot 有提升（F1 高 ≥ 0.03 为可见提升）
- 有无某个子集明显退步

**下一步指导**
告诉 Claude Code：对比表全文，以及是否考虑 GRPO。
→ 进入 GRPO 需满足：SFT F1 高 ≥ 0.05、蒸馏保留率 ≥ 70%、还有剩余 session 时间。告知奖励函数偏好（label 正确 / region 精确 / 两者结合），Claude Code 届时生成 GRPO cell。
