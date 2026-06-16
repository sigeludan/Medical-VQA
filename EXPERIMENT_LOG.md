# Medical VQA 实验记录

基于 **InternVL2-8B** 在 **VQA-RAD** 数据集上的医学视觉问答实验。  
主方法：冻结视觉编码器 + LoRA 微调语言模型（SFT）。  
消融实验：联合微调 **mlp1 projector**（§4.7）、LoRA **rank=16 + early stopping**（§4.8），均为负向结果。

---

## 1. 实验目标

| 目标                                     | 状态                                            |
| ---------------------------------------- | ----------------------------------------------- |
| 环境搭建、数据预处理、zero-shot baseline | 完成                                            |
| LoRA SFT 训练、val 中间评估              | 完成                                            |
| test 完整评估、Before/After 对比         | 完成                                            |
| LoRA + mlp1 projector 联合微调消融       | 完成（val 负向，未跑 test）                     |
| LoRA rank=16 + early stopping 消融       | 完成（val 负向，未跑 test；训练因磁盘不足中断） |



---

## 2. 环境与硬件

| 项目            | 配置                                       |
| --------------- | ------------------------------------------ |
| GPU             | NVIDIA A800 80GB PCIe                      |
| Python          | 3.12                                       |
| PyTorch         | 2.5.1+cu124                                |
| CUDA            | 12.4                                       |
| transformers    | 4.41.x                                     |
| peft            | 已安装                                     |
| Flash Attention | 未安装（使用 eager attention，不影响训练） |

**基础模型**：`models/OpenGVLab/InternVL2-8B`（本地，约 16GB，4 个 safetensors 分片）

**项目根目录**：`/root/autodl-tmp/VQA`

---

## 3. 数据集：VQA-RAD

### 3.1 数据来源

- HuggingFace：`flaviagiammarino/vqa-rad`（去重后版本）
- 国内下载：使用 `hf-mirror.com` + `aria2` 多线程（`scripts/download_vqa_rad.sh`）
- 原始格式：parquet（图像嵌入其中），解压后 2244 张图像

### 3.2 划分方式

按 **image 级别** 随机划分，比例 **7 : 1 : 2**，`seed=42`：

| 划分     | 样本数   | 说明                   |
| -------- | -------- | ---------------------- |
| train    | 1570     | closed 1362 / open 208 |
| val      | 224      | 训练过程监控           |
| test     | 450      | 最终评估（不参与训练） |
| **合计** | **2244** | 2244 张唯一图像        |

### 3.3 数据格式

JSONL 每条样本：

```json
{
  "image": "data/images/train_0000.jpg",
  "question": "are regions of the brain infarcted?",
  "answer": "yes",
  "answer_type": "closed",
  "split": "train"
}
```

- `answer_type`：`closed`（yes/no 或短答）/ `open`（描述类）
- 训练用 InternVL 格式：`data/train_internvl.json`、`data/val_internvl.json`（`scripts/build_train_json.py` 生成）

---

## 4. 实验流程

### Baseline 推理

#### 4.1 预处理

```bash
python3 scripts/preprocess_vqa_rad.py
python3 scripts/build_train_json.py   # Day 2 前执行
```

#### 4.2 Prompt 设计

| 版本   | 策略                                                | 脚本                           |
| ------ | --------------------------------------------------- | ------------------------------ |
| **v1** | 统一要求 yes/no 短答                                | `scripts/baseline_infer.py`    |
| **v2** | 按题型分流：yes/no 题只答 yes/no；其他题 1–6 词短答 | `scripts/baseline_infer_v2.py` |

v2 显著减少「what/where 题误答 yes/no」的问题，**后续训练与评估统一使用 v2 prompt**。

#### 4.3 Zero-shot Baseline 结果（test 450）

| 实验        | Prompt | Closed EM | Open BLEU-1 | Open ROUGE-L | 输出文件                                |
| ----------- | ------ | --------- | ----------- | ------------ | --------------------------------------- |
| Baseline v1 | v1     | **44.9%** | 0.036       | 0.052        | `outputs/baseline_predictions.jsonl`    |
| Baseline v2 | v2     | **52.1%** | 0.274       | 0.324        | `outputs/baseline_v2_predictions.jsonl` |

**v1 → v2 变化（test）**：

| 子项                        | v1    | v2    |
| --------------------------- | ----- | ----- |
| yes/no EM (229题)           | 71.6% | 73.8% |
| 短答 EM (145题)             | 2.8%  | 17.9% |
| 误用 yes/no（GT 非 yes/no） | 38.0% | 0.7%  |

结论：v2 主要修复了**答题格式**；yes/no 医学判断能力变化不大。正式对比以 **v2 为 Before**。

---

### LoRA 微调

#### 4.4 方法设计

```
InternVL2-8B
├── vision_model (InternViT)     ← 完全冻结
├── mlp1 (projector)             ← 冻结（train_projector=False）
└── language_model (InternLM2)   ← LoRA 微调
    └── target_modules:
        attention.wqkv, attention.wo,
        feed_forward.w1, w2, w3
```

| 超参数          | 值                                      |
| --------------- | --------------------------------------- |
| LoRA rank       | 8                                       |
| LoRA alpha      | 16                                      |
| LoRA dropout    | 0.05                                    |
| 可训练参数      | 18,874,368 / 7,756,656,640（**0.24%**） |
| Epochs          | 3                                       |
| Batch size      | 2 × grad_accum 8 = **有效 16**          |
| Learning rate   | 2e-4，cosine + warmup 3%                |
| 精度            | bf16                                    |
| 对话模板        | internlm2-chat                          |
| num_image_token | 256 / patch                             |
| max_num_patches | 6                                       |
| Loss            | 仅对 answer token 计算 CE               |

#### 4.5 训练过程

| 项目                | 数值                     |
| ------------------- | ------------------------ |
| 训练样本            | 1570                     |
| 总 step             | 294                      |
| 训练时长            | 约 1 h 23 min（~5002 s） |
| 最终 train loss     | 0.507                    |
| Step 200 train loss | 0.397                    |
| Step 200 val loss   | 0.740                    |

训练 notebook：`train_lora.ipynb`  
LoRA 权重：`checkpoints/internvl2-vqa-lora/adapter/`（约 37MB）

**备注**：

- 保存 adapter 时曾尝试连接 `huggingface.co` 拉取 `internlm2_5-7b-chat` config，国内超时；权重已成功保存，已将 `adapter_config.json` 的 `base_model_name_or_path` 改为本地路径。
- notebook 保存单元格已改为离线保存。

#### 4.6 Val 评估（224 条）

```bash
python3 scripts/eval_lora.py \
  --lora-path checkpoints/internvl2-vqa-lora/adapter \
  --test-file data/vqa_rad_val.jsonl
```

| 指标         | Baseline v2（test 参考） | LoRA（val） |
| ------------ | ------------------------ | ----------- |
| Closed EM    | 52.1%                    | **59.4%**   |
| Open BLEU-1  | 0.274                    | 0.313       |
| Open ROUGE-L | 0.324                    | 0.356       |

Val 分项：yes/no **76.3%**，短答 **34.2%**。

---

#### 4.7 消融：LoRA + mlp1 Projector 联合微调

**动机**：主实验冻结 `mlp1` projector（§4.4），视觉特征经固定映射进入 LLM。尝试同时微调 LoRA 与 projector，看能否改善视觉-语言对齐。

**方法设计**

```
InternVL2-8B
├── vision_model (InternViT)     ← 完全冻结
├── mlp1 (projector)             ← 联合微调（train_projector=True）
└── language_model (InternLM2)   ← LoRA 微调（同 §4.4 target_modules）
```

| 超参数            | LoRA-only（§4.4） | LoRA + mlp1                              |
| ----------------- | ----------------- | ---------------------------------------- |
| LoRA rank / alpha | 8 / 16            | 8 / 16                                   |
| 可训练参数        | 18.9M（0.24%）    | **52.4M（0.65%）**（+ mlp1 33.6M）       |
| Learning rate     | 2e-4              | **1e-4**                                 |
| Epochs / batch    | 3 / 有效 16       | 3 / 有效 16                              |
| projector 保存    | —                 | `mlp1_projector.pt`（约 64MB，单独保存） |

训练 notebook：`train_lora_mlp1.ipynb`  
权重目录：`checkpoints/internvl2-vqa-lora-mlp1/`（adapter + `mlp1_projector.pt`）

**训练过程**

| 项目                | LoRA-only | LoRA + mlp1 |
| ------------------- | --------- | ----------- |
| 总 step             | 294       | 294         |
| 训练时长            | ~83 min   | ~81 min     |
| 最终 train loss     | **0.507** | 0.560       |
| Step 200 train loss | **0.397** | 0.497       |
| Step 200 val loss   | **0.740** | 0.770       |

训练 loss 与 val loss 均高于 LoRA-only，提示联合微调在当前超参下未带来更好泛化。

**Val 评估（224 条）**

```bash
python3 scripts/eval_lora.py \
  --lora-path checkpoints/internvl2-vqa-lora-mlp1/adapter \
  --projector-path checkpoints/internvl2-vqa-lora-mlp1/mlp1_projector.pt \
  --test-file data/vqa_rad_val.jsonl \
  --metrics-file lora_mlp1_val_metrics.json \
  --predictions-file lora_mlp1_val_predictions.jsonl
```

| 指标         | LoRA-only（val） | LoRA + mlp1（val） | Δ           |
| ------------ | ---------------- | ------------------ | ----------- |
| Closed EM    | **59.4%**        | 55.8%              | **-3.6 pp** |
| Open BLEU-1  | **0.313**        | 0.272              | -0.041      |
| Open ROUGE-L | **0.356**        | 0.301              | -0.055      |

**Val 分项**

| 子项                 | LoRA-only  | LoRA + mlp1 | Δ           |
| -------------------- | ---------- | ----------- | ----------- |
| yes/no（118 题）     | **76.3%**  | 69.5%       | **-6.8 pp** |
| 短答 closed（79 题） | 34.2%      | **35.4%**   | +1.2 pp     |
| 开放题（27 题）      | BLEU 0.313 | 0.272       | 更差        |

**与 LoRA-only 逐题配对（val）**

|                    | 数量      |
| ------------------ | --------- |
| 两模型都对         | 104       |
| 两模型都错         | 93        |
| 仅 mlp1 对         | 9         |
| 仅 LoRA 对         | 18        |
| **净损失（mlp1）** | **-9 题** |

**消融结论**：联合微调 mlp1 在 val 上全面退步，主要损失在 yes/no 判断；短答略有提升但不足以弥补。**未跑 test**（val 已明确落后，正式模型仍用 LoRA-only）。推理须同时加载 adapter 与 `mlp1_projector.pt`（`eval_lora.py --projector-path`）。

---

#### 4.8 消融：LoRA rank=16 + Early Stopping

**动机**：主实验 rank=8、固定 3 epoch，存在 train loss 低于 val loss 的过拟合迹象（39 题 test 退步）。尝试增大 LoRA 表达能力（rank 16）并用 early stopping 在 val loss 最低点停止。

**方法设计**

| 超参数                  | LoRA-only（§4.4） | LoRA rank=16         |
| ----------------------- | ----------------- | -------------------- |
| LoRA rank / alpha       | 8 / 16            | **16 / 32**          |
| 可训练参数              | 18.9M（0.24%）    | **37.7M（0.47%）**   |
| Learning rate           | 2e-4              | 2e-4                 |
| Max epochs              | 3                 | **5**（上限）        |
| eval / save steps       | 200               | **98**（约每 epoch） |
| early_stopping_patience | —                 | **2**                |
| load_best_model_at_end  | False             | **True**             |

训练 notebook：`train_lora_r16.ipynb`  
权重目录：`checkpoints/internvl2-vqa-lora-r16/`  
工具：`scripts/export_lora_from_checkpoint.py`（从 Trainer 全量 checkpoint 导出 LoRA adapter）

**训练过程**

| Step    | Train Loss | Val Loss  | 备注                                       |
| ------- | ---------- | --------- | ------------------------------------------ |
| 99      | 0.689      | 0.756     | epoch 1 结束                               |
| **198** | **0.463**  | **0.705** | **val loss 最佳**                          |
| 297     | 0.167      | 0.878     | 过拟合；保存 checkpoint 时**磁盘不足**中断 |

- 训练未跑完 Cell 5 手动保存 adapter；从 `checkpoint-198` 导出 adapter（`best_eval_loss=0.705`，约 2 epoch）。

**Val 评估（224 条）**

```bash
python3 scripts/eval_lora.py \
  --lora-path checkpoints/internvl2-vqa-lora-r16/adapter \
  --test-file data/vqa_rad_val.jsonl \
  --metrics-file lora_r16_val_metrics.json \
  --predictions-file lora_r16_val_predictions.jsonl
```

| 指标                   | LoRA rank=8（val） | LoRA rank=16（val） | Δ           |
| ---------------------- | ------------------ | ------------------- | ----------- |
| Closed EM              | **59.4%**          | 52.3%               | **-7.1 pp** |
| Open BLEU-1            | 0.313              | **0.320**           | +0.007      |
| Open ROUGE-L           | 0.356              | **0.381**           | +0.025      |
| Step 200 左右 val loss | 0.740              | **0.705**           | 更低        |

**Val 分项**

| 子项                  | rank=8    | rank=16 | Δ       |
| --------------------- | --------- | ------- | ------- |
| yes/no（118 题）      | **76.3%** | 70.3%   | -6.0 pp |
| 短答 closed（106 题） | **30.2%** | 20.8%   | -9.4 pp |

**与 rank=8 逐题配对（val）**：仅 rank16 对 22 / 仅 rank8 对 39，**净损失 -17 题**。

**消融结论**：rank=16 在 val closed EM 上明显差于 rank=8，尽管 val loss 更低，说明 **loss 与 EM 不完全一致**。开放题 BLEU/ROUGE 略好但样本仅 27 条。Early stopping 最佳点（step 198）仍不及 rank=8 训满 3 epoch。**未跑 test**。正式模型保持 **rank=8**。

---

### Test 最终评估

```bash
python3 scripts/eval_lora.py \
  --lora-path checkpoints/internvl2-vqa-lora/adapter \
  --test-file data/vqa_rad_test.jsonl \
  --predictions-file lora_test_predictions.jsonl \
  --metrics-file lora_test_metrics.json
```

---

## 5. 最终结果（test 450，同一 prompt v2）

### 5.1 总表

**正式模型：LoRA-only**（test 450）。mlp1 消融仅 val，见 §4.7。

| 模型                        | 划分     | Closed EM | Open BLEU-1 | Open ROUGE-L |
| --------------------------- | -------- | --------- | ----------- | ------------ |
| InternVL2-8B zero-shot (v2) | test     | 52.1%     | 0.274       | 0.324        |
| **+ LoRA SFT（正式）**      | **test** | **63.1%** | **0.320**   | **0.386**    |
| + LoRA + mlp1（消融）       | val      | 55.8%     | 0.272       | 0.301        |
| + LoRA rank=16（消融）      | val      | 52.3%     | 0.320       | 0.381        |
| LoRA rank=8（对照）         | val      | 59.4%     | 0.313       | 0.356        |

**LoRA vs Baseline（test）**

| **Δ（绝对值）** | +11.0 pp   | +0.046     | +0.062     |
| --------------- | ---------- | ---------- | ---------- |
| **Δ（相对）**   | **+21.1%** | **+17.0%** | **+19.0%** |

### 5.2 分项对比（test，LoRA-only）

| 子项                  | Baseline v2 | LoRA      | Δ            |
| --------------------- | ----------- | --------- | ------------ |
| yes/no（229 题）      | 73.8%       | **77.3%** | +3.5 pp      |
| 短答 closed（145 题） | 17.9%       | **40.7%** | **+22.8 pp** |
| 开放题（76 题）       | BLEU 0.274  | **0.320** | +17%         |

### 5.3 配对分析（test 450，LoRA-only）

|                               | 数量       |
| ----------------------------- | ---------- |
| 由错变对                      | **+86**    |
| 由对变错                      | **-39**    |
| **净增正确**                  | **+47 题** |
| yes/no 被修复                 | 32 题      |
| 短答被修复                    | 45 题      |
| 开放题 BLEU 明显提升（>0.05） | 20 / 76 题 |

### 5.4 宽松匹配评估（Relaxed EM，test 450）

**动机**：Strict EM 对字面完全一致要求过严（如 `axial` vs `axial plane`、`brain` vs `the brain`），部分预测语义正确仍计错（见 §6.2）。

**实现**（`scripts/vqa_common.py`）：

```
relaxed_match = exact_match（yes/no 仍 strict）
              ∨ synonyms_match（18 组 VQA-RAD 同义词）
              ∨ word_containment_match（词边界包含，避免 normal⊂abnormal）
```

`eval_lora.py` / `baseline_infer_v2.py` 支持 `--relaxed-metrics`，输出 `closed_relaxed_match`。

```bash
# 在已有预测上重算（无需重新推理）
python3 scripts/eval_lora.py \
  --lora-path checkpoints/internvl2-vqa-lora/adapter \
  --test-file data/vqa_rad_test.jsonl \
  --predictions-file lora_test_predictions.jsonl \
  --metrics-file lora_test_metrics_relaxed.json \
  --relaxed-metrics
```

**LoRA rank=8 test 结果**

| 指标               | Strict    | Relaxed      | Δ           |
| ------------------ | --------- | ------------ | ----------- |
| Closed EM          | **63.1%** | **67.9%**    | **+4.8 pp** |
| 新增算对（closed） | —         | **18 / 374** | —           |

开放题 BLEU/ROUGE 不变（宽松规则仅作用于 closed）。

**典型由 strict→relaxed 算对的案例**

| 类型   | GT              | 预测                               |
| ------ | --------------- | ---------------------------------- |
| 同义词 | `axial plane`   | `axial`                            |
| 同义词 | `4th ventricle` | `fourth ventricle`                 |
| 同义词 | `csf`           | `cerebrospinal fluid`              |
| 同义词 | `the brain`     | `brain`                            |
| 同义词 | `x ray`         | `chest x ray`                      |
| 包含   | `enlarged`      | `enlarged with nodular thickening` |
| 包含   | `fluid`         | `cerebrospinal fluid`              |

**说明**：正式对比与论文主表仍用 **Strict EM 63.1%**；Relaxed **67.9%** 作为辅助参考。输出：`outputs/lora_test_metrics_relaxed.json`。

---

## 6. 典型案例

### 6.1 成功案例（Baseline 错 → LoRA 对）

**解剖结构识别**

```
Q: what are the bright white structures almost forming an x?
GT: lateral ventricles
Baseline: cerebral hemispheres  →  LoRA: lateral ventricles ✓
```

**yes/no 判断修正**

```
Q: is this consistent with an acute infarction?
GT: yes | Baseline: no  →  LoRA: yes ✓
```

**答法对齐数据集**

```
Q: what plane is this?
GT: axial | Baseline: axial plane  →  LoRA: axial ✓
```

### 6.2 失败案例（仍错或退步）

**格式不匹配（语义对、EM 错）**

```
Q: in what plane was this image taken?
GT: axial plane | Baseline: axial plane ✓  →  LoRA: axial ✗
```

**微调后退步**

```
Q: are there calcifications present on the abdominal aorta?
GT: yes | Baseline: yes  →  LoRA: no ✗

Q: what is the condition?
GT: diverticulitis | Baseline: diverticulitis  →  LoRA: appendicitis ✗
```

**左右侧混淆**

```
Q: is the lesion on the left or right?
GT: right | Baseline: right  →  LoRA: left ✗
```

---

## 7. 结论与分析

### 7.1 主要结论

1. **LoRA 微调有效**：test closed EM 从 52.1% 提升至 63.1%（+11 pp），达到 Day 2 目标（+5%）的两倍。
2. **最大收益在短答题**：145 条短答 EM 从 17.9% → 40.7%（约 2.3 倍），说明 SFT 成功对齐 VQA-RAD 短答风格。
3. **yes/no 小幅提升**：73.8% → 77.3%，不是主战场但仍有净收益。
4. **开放题改善**：BLEU/ROUGE 提升约 17–19%，但 76 题中仍有大量字面不匹配。
5. **Prompt 很重要**：v1 → v2 带来 +7 pp，说明训推 prompt 一致至关重要。
6. **mlp1 联合微调为负向消融**（§4.7）：val Closed EM 59.4% → 55.8%，yes/no 76.3% → 69.5%。
7. **LoRA rank=16 为负向消融**（§4.8）：val Closed EM 59.4% → 52.3%，短答 30.2% → 20.8%；val loss 更低但 EM 更差。正式方案保持 **rank=8、LoRA-only、3 epoch**。
8. **宽松匹配（§5.4）**：Strict EM 63.1% 下有 18 题 closed 属同义/包含关系误判；Relaxed EM **67.9%**（+4.8 pp），不改变开放题指标。

### 7.2 局限性

| 问题                          | 说明                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| 数据量小                      | 仅 1570 条训练样本，易过拟合（train loss 0.40 vs val loss 0.74） |
| Exact Match 偏严              | `axial` vs `axial plane` 等；已实现 Relaxed EM（§5.4），test +4.8 pp |
| 39 题退步                     | 约 8.7% 样本微调后变差，存在灾难性遗忘个案                   |
| 复杂病理混淆                  | diverticulitis / appendicitis 等仍易混                       |
| projector 联合微调无效        | §4.7：val EM -3.6 pp                                         |
| rank=16 + early stopping 无效 | §4.8：val EM -7.1 pp；loss↓ 但 EM↓                           |
| Trainer checkpoint 占磁盘     | 每个 `checkpoint-*` 约 16GB（全量模型），adapter 仅 ~37–73MB |

### 7.3 与 Grounding DINO LoRA 的异同

|           | Grounding DINO        | 本实验 InternVL2 VQA            |
| --------- | --------------------- | ------------------------------- |
| 任务      | 目标检测 / 文本定位   | 看图生成答案                    |
| LoRA 位置 | Transformer + bbox 头 | InternLM2 注意力 + FFN          |
| 冻结部分  | backbone + BERT       | InternViT                       |
| rank      | 32                    | **8**（正式；rank 16 消融更差） |
| 损失      | cls + bbox + GIoU     | 因果 LM CE（仅 answer）         |

共同点：冻结视觉主干，PEFT 高效适配。

---

## 8. 项目文件结构

```
VQA/
├── EXPERIMENT_LOG.md              # 本文件
├── train_lora.ipynb               # LoRA rank=8 训练（正式）
├── train_lora_mlp1.ipynb          # LoRA + mlp1 消融
├── train_lora_r16.ipynb           # LoRA rank=16 + early stopping 消融
├── vqa_3day_plan.html             # 三天实验计划
├── data/
│   ├── vqa_rad_{train,val,test}.jsonl
│   ├── train_internvl.json
│   ├── val_internvl.json
│   └── images/                    # 2244 张图像
├── models/OpenGVLab/InternVL2-8B/
├── scripts/
│   ├── preprocess_vqa_rad.py
│   ├── build_train_json.py
│   ├── baseline_infer.py          # prompt v1
│   ├── baseline_infer_v2.py       # prompt v2
│   ├── eval_lora.py
│   ├── export_lora_from_checkpoint.py  # 从 checkpoint-* 导出 adapter
│   ├── internvl_sft_utils.py
│   └── vqa_common.py
├── checkpoints/internvl2-vqa-lora/
│   └── adapter/                   # 正式 LoRA 权重（rank=8）
├── checkpoints/internvl2-vqa-lora-mlp1/
│   ├── adapter/
│   └── mlp1_projector.pt
├── checkpoints/internvl2-vqa-lora-r16/
│   └── adapter/                   # rank=16 消融（自 checkpoint-198 导出）
└── outputs/
    ├── baseline_predictions.jsonl
    ├── baseline_v2_predictions.jsonl
    ├── lora_val_predictions.jsonl
    ├── lora_test_predictions.jsonl
    ├── lora_test_metrics_relaxed.json   # strict + relaxed EM
    ├── lora_mlp1_val_predictions.jsonl
    ├── lora_r16_val_predictions.jsonl
    └── *_metrics.json
```

---

## 9. 复现命令

```bash
cd /root/autodl-tmp/VQA

# 1. 数据
bash scripts/download_vqa_rad.sh
python3 scripts/preprocess_vqa_rad.py
python3 scripts/build_train_json.py

# 2. Baseline（v2 推荐）
python3 scripts/baseline_infer_v2.py --limit 0

# 3. 训练（打开 train_lora.ipynb 逐格运行）
# 或复用已保存 adapter

# 4. 评估（LoRA-only，正式模型，strict）
python3 scripts/eval_lora.py \
  --lora-path checkpoints/internvl2-vqa-lora/adapter \
  --test-file data/vqa_rad_test.jsonl \
  --predictions-file lora_test_predictions.jsonl \
  --metrics-file lora_test_metrics.json

# 4b. 同上 + relaxed EM（可基于已有 predictions 重算）
python3 scripts/eval_lora.py \
  --lora-path checkpoints/internvl2-vqa-lora/adapter \
  --test-file data/vqa_rad_test.jsonl \
  --predictions-file lora_test_predictions.jsonl \
  --metrics-file lora_test_metrics_relaxed.json \
  --relaxed-metrics

# 5. 评估（LoRA + mlp1 消融）
python3 scripts/eval_lora.py \
  --lora-path checkpoints/internvl2-vqa-lora-mlp1/adapter \
  --projector-path checkpoints/internvl2-vqa-lora-mlp1/mlp1_projector.pt \
  --test-file data/vqa_rad_val.jsonl \
  --metrics-file lora_mlp1_val_metrics.json \
  --predictions-file lora_mlp1_val_predictions.jsonl

# 6. 评估（LoRA rank=16 消融）
python3 scripts/eval_lora.py \
  --lora-path checkpoints/internvl2-vqa-lora-r16/adapter \
  --test-file data/vqa_rad_val.jsonl \
  --metrics-file lora_r16_val_metrics.json \
  --predictions-file lora_r16_val_predictions.jsonl

# 7. 从 Trainer checkpoint 抢救 adapter（训练中断时）
python3 scripts/export_lora_from_checkpoint.py \
  --checkpoint checkpoints/internvl2-vqa-lora-r16/checkpoint-198 \
  --output-dir checkpoints/internvl2-vqa-lora-r16 \
  --lora-rank 16 --lora-alpha 32
```

---

## 10. 后续可改进方向

- [x] 解冻 `mlp1` projector 联合微调 → **val 负向**（§4.7）
- [x] LoRA rank 16、early stopping → **val 负向**（§4.8）；正式模型仍 rank=8
- [ ] mlp1 分组学习率（可选）
- [x] 评估时增加宽松匹配（同义词、包含关系）→ **§5.4**，test Relaxed EM 67.9%

---

## 11. 时间线

| 日期       | 事项                                                         |
| ---------- | ------------------------------------------------------------ |
| 2026-06-14 | 数据下载（hf-mirror）、预处理、baseline v1/v2                |
| 2026-06-14 | 训练数据构建、LoRA 训练脚本/notebook                         |
| 2026-06-15 | 3 epoch 训练结束，adapter 保存                               |
| 2026-06-15 | val / test 评估，整理实验记录                                |
| 2026-06-15 | LoRA + mlp1 训练（`train_lora_mlp1.ipynb`），val 评估，确认负向消融 |
| 2026-06-15 | LoRA rank=16 + early stopping（`train_lora_r16.ipynb`）；val EM 52.3%（负向） |
| 2026-06-15 | 实现 Relaxed EM（`vqa_common.py` + `--relaxed-metrics`）；rank=8 test strict 63.1% → relaxed 67.9% |

---

*最后更新：2026-06-15（含 Relaxed EM）*

