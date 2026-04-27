# AGENTS.md

## 项目概览

TSAD-SFT：基于视觉 SFT 微调的时间序列异常检测项目。该项目基于 AnomLLM 框架，使用教师蒸馏 + SFT 微调 Qwen3-VL，以从时间序列折线图中检测异常区间。

## 流水线

```
合成数据（4 类 x 400）→ VLM 零样本基线 → 教师蒸馏（qwen3.5-plus）
→ 自动清洗 → SFT 训练（Qwen3-VL-8B + LoRA + Unsloth）→ SFT 推理（vLLM）→ 评估（F1 / Affi-F1）
→ GRPO 强化学习（F1 task reward + Unsloth）→ GRPO 评估（F1 / Affi-F1）
```

## 关键约束

- **不要尝试直接运行 `.ipynb`** —— Notebook 永远只在 Google Colab（A100 GPU）上运行，不要在本地环境执行
- **不要编写过度冗余的代码** —— 保持实现简洁，避免不必要的抽象、重复逻辑和样板代码
- **禁止复杂化与过度设计** —— 对简单问题优先采用最小可行方案；能用直接的 notebook cell、shell 命令或局部修改解决，就不要额外引入 helper、封装层、状态机或大段控制逻辑
- **Part 1-4 已运行完成** —— 后续诊断、后处理、消融和新实验默认从 Google Drive 已保存的 `packs/` 归档恢复数据，并创建新的 notebook；不要改动 Part 1-4 的主线阶段结构
- 运行环境：Google Colab；教师蒸馏阶段需要 DashScope API Key

## 文件结构

```
notebooks/part1.ipynb         # Part 1：环境初始化 → Baseline 评估 → 检查点 B
notebooks/part2.ipynb         # Part 2：恢复检查点 B → 教师蒸馏 → 训练数据准备 → before_stage8 打包
notebooks/part3.ipynb         # Part 3：恢复 before_stage8 → SFT 训练 → 导出 → 评估
notebooks/part4.ipynb         # Part 4：恢复 Part 3 模型 → GRPO 训练 → 评估
notebooks/part5_drive_experiments.ipynb
                              # Part 5：从 Drive 保存归档恢复数据，做诊断、后处理和后续实验
docs/
  notebook_guide.md           # 分阶段 Notebook 使用指南
  project_report.md           # 当前结果、流程与证据边界
  optimization_plan.md        # 后续诊断与优化方案
  references/                 # Unsloth / TRL 离线参考资料
archive/
  code/                       # 旧 AnomLLM 本地副本归档
  old_plans/                  # 早期计划文档
  notebook_exports/           # notebook 文本导出
  figures/                    # 历史图片输出
  papers/                     # 参考论文
```

## 技术栈

- 模型：Qwen3-VL-8B（student），qwen3.5-plus（teacher）
- 训练：Unsloth + TRL SFTTrainer，4-bit quantization，LoRA（rank=16，alpha=16）
- 推理：vLLM（OpenAI-compatible API）
- 指标：point-wise F1，affiliation F1

## 约定

- VLM 输出格式：`[{"start": N, "end": M}, ...]`
- 语言：文档使用中文，代码使用英文
- 数据：1000 步单变量时间序列，带异常区间标签

## Notebook 固定细节速查

下面是当前四本 notebook 中已经写死的关键细节。后续 agent 默认以这里为准，除非用户明确要求改 notebook。

### 运行目录与持久化路径

```python
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
ANOMLLM    = RT_CODE / "AnomLLM"
```

- Colab 每次新 session 都要从 `Cell 0.1` 重新开始：挂载 Drive、建目录、重装依赖
- `Cell 1.2` 的数据恢复逻辑：
  - 优先从 `/content/drive/MyDrive/tsad_anomaly/packs/anomllm_data.tar` 恢复
  - 若 Drive 没缓存，则安装 `s5cmd` 并从 `https://s3-west.nrp-nautilus.io` 的 `s3://anomllm/data/*` 拉取到 `AnomLLM/data/`
  - S3 不可用时才考虑取消注释 `bash synthesize.sh`
  - 首次下载后会再打包回 Drive
- `PYTHONPATH` 在 notebook 中被设置为 `/content/tsad_runtime/code/AnomLLM/src`
- 固定子集顺序：`DATASETS = ["flat-trend", "range", "point", "freq"]`

### Notebook 划分

- `notebooks/part1.ipynb`：阶段 `0-4`，并在检查点 B 后把 baseline/data 中间产物打包到 Drive 的 `before_ckptB_*.tar.gz`
- `notebooks/part2.ipynb`：阶段 `0-B`、`5-7`，并在阶段 7 末尾把蒸馏产物和后续评估所需文件打包到 Drive 的 `before_stage8_*.tar.gz`
- `notebooks/part3.ipynb`：阶段 `0-C`、`8-10`，恢复 `before_stage8_*.tar.gz` 后继续训练、导出和评估；结果保存到 `part3_results_only_*.tar.gz`，模型保存到 `DRV_SFT/part3_models_*`
- `notebooks/part4.ipynb`：阶段 `0-D`、`11-13`，从 `DRV_SFT/part3_models_*` 恢复 SFT 合并模型，从 `part3_results_only_*.tar.gz` 恢复数据文件，执行 GRPO 训练和评估
- `notebooks/part5_drive_experiments.ipynb`：Part 1-4 完成后的实验入口，从 `DRV_PACK` 恢复 `part3_results_only_*.tar.gz` 和 `part4_results_only_*.tar.gz`，生成诊断表和后处理实验结果

### Notebook 阶段与实际动作

#### `notebooks/part1.ipynb`

- `0.1`：`drive.mount("/content/drive")`
- `0.2`：创建运行目录和 Drive 持久化目录
- `0.3`：安装 `unsloth[colab-new]`、`vllm`、`openai`、`accelerate`、`bitsandbytes`、`scikit-learn`、`pandas`、`pyyaml`、`datasets`、`matplotlib`、`pillow`、`trl`、`loguru`、`google-generativeai`、`requests`、`affiliation-metrics-py`
- `1.1`：若 `/content/tsad_runtime/code/AnomLLM` 不存在则 `git clone https://github.com/Rose-STL-Lab/AnomLLM.git`
- `1.2`：恢复或下载合成数据
- `1.3`：逐个检查 `point/range/freq/flat-trend` 的 `eval/data.pkl`，打印样本数、shape、`has_figs`
- `1.4`：固定本次实验数据子集，并画异常/正常分布柱状图
- `检查点 A`：确认 GPU 型号、剩余显存、`torch.cuda.is_bf16_supported()`，并目视 `point/eval/figs/001.png`
- `2.1`：先 `pkill` 8000 端口旧服务，再后台启动本地 vLLM
  - 模型：`Qwen/Qwen3-VL-8B-Instruct`
  - served model name：`qwen-local`
  - 地址：`127.0.0.1:8000`
  - 额外参数：`--max-model-len 8192 --gpu-memory-utilization 0.95`
  - 日志：`/tmp/vllm.log`
  - 必须用 `curl http://127.0.0.1:8000/health` 返回 `200` 再继续
- `2.2`：写 `AnomLLM/credentials.yml`
  ```yaml
  qwen-local:
    api_key: dummy
    base_url: "http://127.0.0.1:8000/v1"
  ```
- `2.3`：对四个子集循环执行 `python src/online_api.py --data "$datum" --model qwen-local --variant 0shot-vision`
- `3.1`：对四个子集循环执行 `python src/baselines/isoforest.py --data "$datum" --model isolation-forest`
- `4.1`：对每个子集跑 `src/result_agg.py`，产出 `results/agg/{subset}.pkl`
- `4.2`：把四个 `pkl` 合并成 `/content/tsad_runtime/results/baseline_compare.csv`
- `检查点 B`：`qwen-local/0shot-vision.jsonl` 和 `isolation-forest/0shot.jsonl` 都应各 400 行；Baseline F1 经验期望在 `0.2~0.7`
- 检查点 B 后的备份 cell：把 baseline JSONL、agg PKL、`baseline_compare.csv`、4 个子集的 `eval/data.pkl` 与 `figs/` 打包到 `before_ckptB_*.tar.gz`

#### `notebooks/part2.ipynb`

- `0-B.1`：挂载 Drive、定义运行目录、安装依赖
- `0-B.2`：恢复最新 `before_ckptB_*.tar.gz`，并校验 `baseline_compare.csv`、4 个 `data.pkl`、各子集 baseline JSONL / agg PKL
- `5.1`：从 4 个子集的 `eval/data.pkl` 提取 1600 条样本，构建 `sft_manifest.csv`
- `6.1`：设置 DashScope 教师模型，notebook 中唯一需要手工填写的是 `DASHSCOPE_API_KEY`
- `6.2a`：只蒸馏前 20 条，输出 `sft_raw_smoke.jsonl`
- `6.2b`：对 `train+val` 全量蒸馏，输出 `sft_raw.jsonl`
- `6.3`：清洗蒸馏结果，输出 `sft_final.jsonl`
- `7.1`：把 `sft_final.jsonl` 转成 Unsloth `messages` 训练格式，输出 `train.jsonl` 和 `val.jsonl`
- `7.2`：把 `eval` split 单独导出成 `eval.jsonl`
- `7.3`：把 `sft_manifest.csv`、`sft_raw_smoke.jsonl`、`sft_raw.jsonl`、`sft_final.jsonl`、`train.jsonl`、`val.jsonl`、`eval.jsonl`，以及阶段 9 仍需用到的 baseline/data 文件打包到 `before_stage8_*.tar.gz`

#### `notebooks/part3.ipynb`

- `0-C.1`：挂载 Drive、定义运行目录、安装依赖
- `0-C.2`：恢复最新 `before_stage8_*.tar.gz`，并校验：
  - `sft_manifest.csv` 的 split 分布必须是 `train=1280 / val=160 / eval=160`
  - `train.jsonl`、`val.jsonl`、`eval.jsonl`、`sft_final.jsonl` 必须存在且非空
  - 抽样解析一条 `train.jsonl`，确认 image 路径存在、assistant 可解析为区间列表
  - 四个子集的 baseline JSONL 与 `data.pkl` 仍可读取
- `8.1`：加载 `unsloth/Qwen3-VL-8B-Instruct-bnb-4bit` 并挂 LoRA
- `8.2a`：冒烟训练 5 步，只验证流程、OOM 和 nan
- `8.2b`：清显存后重新加载干净模型，跑完整训练
- `检查点 C`：先查看 loss 曲线、蒸馏前 5 条样本、`train.jsonl` 中 image/assistant JSON 是否正常，再执行导出
- `8.3`：导出 merged model 和 adapter
- `9.1`：后台启动 merged model 的 vLLM 服务，地址 `127.0.0.1:8001`，served model name 为 `sft-model`，日志在 `/tmp/sft-vllm.log`
- `9.2`：只在 `eval` 的 160 条样本上，对 `isolation-forest`、`qwen-local-0shot`、`sft-0shot` 做公平对比，输出 `sft_eval_metrics.csv`
- `检查点 D`：看 SFT 是否相对 VLM zero-shot 提升至少 `0.03`；若要进入 GRPO，经验门槛是 `F1 提升 >= 0.05` 且蒸馏保留率 `>= 70%`
- `10`：（已移至 Part 4）检查点 D 后的 GRPO 占位说明，指引用户切换到 `notebooks/part4.ipynb`

#### `notebooks/part4.ipynb`

- `0-D.1`：挂载 Drive、定义运行目录（与 Part 3 相同的路径常量）、安装依赖（同 Part 3 + 确保 `trl>=0.26.2`）
- `0-D.2`：克隆 AnomLLM 仓库；从 `DRV_SFT/part3_models_*` 恢复 SFT 合并模型到 `RT_SFT/qwen3vl-tsad-merged`；从 `DRV_PACK/part3_results_only_*.tar.gz` 恢复数据文件（`sft_manifest.csv`、`train.jsonl`、`val.jsonl`、`eval.jsonl`、baseline JSONL、`data.pkl`、`sft_eval_metrics.csv`）；校验模型目录和数据文件完整性
- `11.1`：加载 SFT 合并模型（`RT_SFT/qwen3vl-tsad-merged`），4bit 量化，添加 LoRA（`finetune_vision_layers=False`，`finetune_language_layers=True`，`r=16`）；定义 `format_reward_func` 和 `f1_reward_func`
- `11.2`：将 `train.jsonl` 转成 GRPO prompt 格式：每条记录包含 `prompt`（user message with image）、`image`（PIL Image）、`ground_truth`（GT intervals JSON string）；仅保留 `user` 部分，不含 `assistant`
- `11.3a`：GRPO 冒烟训练（`max_steps=5`），验证 reward 函数返回值合理、无 OOM、无 nan
- `11.3b`：完整 GRPO 训练
- `检查点 E`：查看 reward 曲线趋势、生成样本质量（前 5 条 completion）；确认 mean reward 上升
- `11.4`：导出 GRPO merged model 和 adapter
- `12.1`：后台启动 GRPO 模型的 vLLM 服务，端口 `8002`，served model name `grpo-model`，日志 `/tmp/grpo-vllm.log`
- `12.2`：在相同 eval 160 条上评估四种方法：`isolation-forest`、`qwen-local-0shot`、`sft-0shot`、`grpo-0shot`
- `检查点 F`：查看 GRPO 是否相对 SFT 有提升；四种方法 F1 对比表
- `13`：结果和模型备份到 Drive（`part4_results_only_*.tar.gz` 和 `DRV_SFT/part4_models_*`）

### SFT 清单与数据格式

- `sft_manifest.csv` 的关键列：
  - `sample_id`：例如 `flat-trend_000`
  - `subset`
  - `pkl_path`
  - `pkl_idx`
  - `image_path`
  - `label`
  - `anomaly_type`
  - `intervals`
  - `split`
- `anomaly_type` 映射：
  - `flat-trend -> trend`
  - `range -> range`
  - `point -> point`
  - `freq -> freq`
  - 正常样本统一为 `none`
- 分层切分逻辑：
  - 第一次 `train_test_split(test_size=0.20, random_state=3407, stratify=subset + "_" + label)` 得到 `train=1280`、`tmp=320`
  - 第二次把 `tmp` 均分成 `val=160`、`eval=160`
- `train.jsonl` / `val.jsonl` 中每条记录格式固定为：
  ```json
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "<VISION_USER_TEXT>"},
          {"type": "image", "image": "/content/.../001.png"}
        ]
      },
      {
        "role": "assistant",
        "content": "[{\"start\": 123, \"end\": 145}]"
      }
    ]
  }
  ```
- `eval.jsonl` 不是训练格式，只保留 `sample_id/image_path/label/anomaly_type/intervals`

### 教师蒸馏的固定 prompt 与清洗规则

- `Cell 6.1` 中固定参数：
  - `TEACHER_MODEL = "qwen3.5-plus"`
  - `TEACHER_TEMPERATURE = 0.2`
  - DashScope OpenAI-compatible base URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- notebook 中的用户提示本质上是：
  - 根据时序图检测异常区间
  - 用 x 轴坐标表示
  - 若无异常则返回空列表 `[]`
- 教师被要求返回严格 JSON object，而不是直接返回区间列表；固定 schema 为：
  ```json
  {
    "is_anomaly": true,
    "anomaly_type": "point|range|freq|trend|none",
    "intervals": [{"start": 1, "end": 2}],
    "rationale": "..."
  }
  ```
- API 调用固定设置：
  - `response_format={"type": "json_object"}`
  - `extra_body={"enable_thinking": False}`
  - 输入图片通过 base64 data URL 传入
- 冒烟通过标准：
  - `is_anomaly` 与 `ground_truth` 一致率 `>= 70%`
  - `intervals` 格式合法率 `>= 70%`
- `intervals` 合法性的判断：
  - 必须是 list
  - 每个元素都是 dict
  - `start/end` 都是数值
  - `start < end`
- `sft_final.jsonl` 的保留规则：
  - `teacher_anom == ground_truth`
  - `intervals` 结构合法
  - `teacher_anom == bool(intervals)`

### SFT 训练的精确配置

- 基础模型：`unsloth/Qwen3-VL-8B-Instruct-bnb-4bit`
- 加载方式：
  - `load_in_4bit=True`
  - `use_gradient_checkpointing="unsloth"`
- LoRA 配置：
  - `finetune_vision_layers=True`
  - `finetune_language_layers=True`
  - `finetune_attention_modules=True`
  - `finetune_mlp_modules=True`
  - `target_modules="all-linear"`
  - `r=16`
  - `lora_alpha=16`
  - `lora_dropout=0.0`
- 冒烟训练 `Cell 8.2a`：
  - `num_train_epochs=3`
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=8`
  - `learning_rate=2e-5`
  - `bf16=True`
  - `logging_steps=1`
  - `eval_strategy="no"`
  - `save_strategy="no"`
  - `optim="adamw_8bit"`
  - `report_to="none"`
  - `remove_unused_columns=False`
  - `max_steps=5`
- 冒烟后的判断口径：
  - 若出现 `nan`，不要继续，先降 `learning_rate`，然后从 `Cell 8.1` 重跑
  - 若初始 loss 不在 `[0.5, 10]`，视为异常
  - markdown 检查点里给出的经验值是 loss 大致在 `1.5~4` 且非 nan
- 完整训练 `Cell 8.2b`：
  - 先 `del trainer, model`，`gc.collect()`，`torch.cuda.empty_cache()`
  - 然后重新加载一遍干净模型和 tokenizer
  - `num_train_epochs=3`
  - `per_device_train_batch_size=2`
  - `gradient_accumulation_steps=8`
  - `learning_rate=1e-4`
  - `bf16=True`
  - `logging_steps=10`
  - `eval_strategy="steps"`
  - `eval_steps=100`
  - `save_strategy="steps"`
  - `save_steps=100`
  - `save_total_limit=2`
  - `optim="adamw_8bit"`
  - `report_to="none"`
  - `remove_unused_columns=False`
- 导出产物：
  - merged model：`/content/tsad_runtime/sft/qwen3vl-tsad-merged`
  - LoRA adapter：`/content/tsad_runtime/sft/qwen3vl-tsad-adapter`

### 最终评估的固定口径

- `Cell 9.1` 会把 `credentials.yml` 追加一项：
  ```yaml
  sft-model:
    api_key: dummy
    base_url: "http://127.0.0.1:8001/v1"
  ```
- `Cell 9.2` 不是直接复用阶段 4 的 baseline 总表，因为阶段 4 是每个子集 400 条全量指标
- 公平比较时，必须：
  - 从 `sft_manifest.csv` 里取 `split == "eval"` 的 160 条样本
  - 按 `subset -> pkl_idx` 建索引
  - 回到原始 `data.pkl` 中读取这些样本的 GT intervals
  - 只在这 160 条上评估三种方法：`isolation-forest`、`qwen-local-0shot`、`sft-0shot`
- 评估实现依赖 `AnomLLM/src/utils.py` 中的：
  - `compute_metrics`
  - `interval_to_vector`
  - `load_results`
- 若某条预测为 `None`，notebook 会把它视为全零向量再算指标
- 最终 summary 是按 `method` 分组后，对 `f1` 和 `affi f1` 取平均，保存到 `/content/tsad_runtime/results/sft_eval_metrics.csv`

### GRPO 训练的精确配置

- 基础模型：从 `RT_SFT/qwen3vl-tsad-merged`（Part 3 导出的 SFT 合并模型）加载
- 加载方式：
  - `load_in_4bit=True`
  - `use_gradient_checkpointing="unsloth"`
- LoRA 配置：
  - `finetune_vision_layers=False`（GRPO 阶段冻结 vision encoder）
  - `finetune_language_layers=True`
  - `finetune_attention_modules=True`
  - `finetune_mlp_modules=True`
  - `r=16`
  - `lora_alpha=16`
  - `lora_dropout=0.0`
- Reward 函数（两个，加权求和）：
  - `format_reward_func`（权重 1.0）：解析 completion 为 JSON 区间列表；格式合法（list of dict，每个含 `start`/`end`，`start < end`）得 1.0 分，否则 0.0
  - `f1_reward_func`（权重 2.0）：解析 completion 为区间列表，调用 `interval_to_vector` + 点级 F1 计算；F1 值即为 reward（范围 0.0~1.0）；解析失败时 reward 为 0.0
  - reward 函数签名使用 `completions` + `ground_truth` + `**kwargs`
- 冒烟训练配置：
  - `max_steps=5`
  - `per_device_train_batch_size=1`
  - `num_generations=2`
  - `learning_rate=5e-6`
  - `max_prompt_length=2048`
  - `max_completion_length=256`
  - `optim="adamw_8bit"`
  - `report_to="none"`
  - `loss_type="dr_grpo"`
- 完整训练配置：
  - `num_train_epochs=1`（GRPO 通常不需要多 epoch）
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=4`
  - `num_generations=4`
  - `learning_rate=5e-6`
  - `warmup_ratio=0.1`
  - `lr_scheduler_type="cosine"`
  - `max_prompt_length=2048`
  - `max_completion_length=256`
  - `max_grad_norm=0.1`
  - `logging_steps=1`
  - `save_steps=50`
  - `save_total_limit=2`
  - `optim="adamw_8bit"`
  - `report_to="none"`
  - `loss_type="dr_grpo"`
- 导出产物：
  - merged model：`/content/tsad_runtime/sft/qwen3vl-tsad-grpo-merged`
  - LoRA adapter：`/content/tsad_runtime/sft/qwen3vl-tsad-grpo-adapter`

### GRPO 评估的固定口径

- `Cell 12.1` 会把 `credentials.yml` 追加一项：
  ```yaml
  grpo-model:
    api_key: dummy
    base_url: "http://127.0.0.1:8002/v1"
  ```
- `Cell 12.2` 在 `eval` 的 160 条样本上评估四种方法：`isolation-forest`、`qwen-local-0shot`、`sft-0shot`、`grpo-0shot`
- SFT 结果复用 Part 3 已有的推理输出（从 `part3_results_only_*.tar.gz` 恢复），不重新推理
- GRPO 推理使用 `online_api.py --model grpo-model --variant 0shot-vision`
- 评估逻辑和指标计算方式与 Part 3 Cell 9.2 完全一致（复用 `compute_metrics`、`interval_to_vector`）
- 最终 summary 保存到 `RT_RESULTS / "grpo_eval_metrics.csv"`

### 默认操作约束

- 默认不要改动 notebook 的 cell 顺序和阶段划分
- 默认不要把 notebook 逻辑再拆成新的本地脚本，除非用户明确要求工程化
- 默认只有 `Cell 6.1` 里的 `DASHSCOPE_API_KEY` 需要人工填写；其它参数按 notebook 默认值先跑通
- 默认先跑冒烟，再跑全量或完整训练；不要跳过检查点
- 默认先完成 Part 3 全部评估并通过检查点 D，再进入 Part 4 GRPO；不要跳过 SFT 阶段
