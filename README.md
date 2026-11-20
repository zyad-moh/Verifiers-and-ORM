# Verifiers and Outcome Reward Model (ORM) on GSM8K with Phi-3-mini

A small, end-to-end, step-by-step walkthrough of building an **Outcome Reward Model (ORM)** for math reasoning:

1. Measure how well a base LLM solves GSM8K and try common **verification techniques** (pass@n, majority voting, LLM-as-a-judge).
2. Use the model's own samples to **build a labeled ORM dataset** (question + candidate solution + correct/incorrect).
3. **Train an ORM** (frozen Phi-3 backbone + linear head), diagnose class imbalance, fix it, and evaluate both as a classifier and as a **best-of-4 ranker**.

Everything uses [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) and the first 100 questions of the GSM8K train split.

> This is a learning/experimentation project. The dataset is tiny (100 questions, 400 solutions), so treat the numbers as a demonstration of the method, not as a benchmark.

---

## Repository contents

| File | Purpose |
|------|---------|
| `verifiers_and_build_orm_data.ipynb` | Notebook 1: baseline accuracy, verification methods, and ORM dataset construction |
| `ORM.ipynb` | Notebook 2: ORM architecture, training, imbalance fix, threshold tuning, evaluation |

Data files produced/consumed along the way (not included unless you add them):

| File / folder | Created by | Used by |
|---------------|------------|---------|
| `data_orm.jsonl` | Notebook 1 | Notebook 2 (full dataset, 400 rows) |
| `train_orm.jsonl`, `val_orm.jsonl`, `test_orm.jsonl` | Notebook 1 (sequential 300 / 52 / 48 split) | Notebook 2 (first, imbalanced experiment) |
| `orm_train/`, `orm_val/`, `orm_test/` | Notebook 2 (`save_to_disk`) | Notebook 2 (balanced split used for final results) |
| `orm_checkpoint.pt`, `orm_weighted_bce_checkpoint.pt` | Notebook 2 | Saved ORM head/optimizer state |

**Run order:** `verifiers_and_build_orm_data.ipynb` → `ORM.ipynb`.

---

## Notebook 1 — `verifiers_and_build_orm_data.ipynb`

### Baseline: how good is Phi-3-mini on GSM8K?
- Prompts the model to solve each problem step by step and end with `Final Answer: <number>`; the final answer is pulled out with a regex, and the ground truth is parsed from the `####` marker in GSM8K.
- Two practical issues found and fixed along the way:
  - **Truncated outputs:** `max_new_tokens=300` cut answers off before the `Final Answer:` line (many `None` extractions), so it was raised to 1000.
  - **Missing chat template:** feeding a raw prompt made the model ramble until the token limit (~61 s/sample). Using `tokenizer.apply_chat_template(..., add_generation_prompt=True)` fixed both the speed and the quality.
- With greedy decoding (`do_sample=False`) on the first 50 questions, the model got **4 wrong** (indices 8, 11, 22, 34).

### Verification techniques explored
| Technique | Setup | Observation |
|-----------|-------|-------------|
| **pass@n** | 4 samples per question (`temperature=0.7`) on the first 10 questions | A question the greedy run got wrong (Q9) was answered correctly in 2 of 4 samples |
| **Majority voting** | 8 samples per question on the first 10 questions, most common final answer wins (`collections.Counter`) | All 10 final answers matched the ground truth |
| **LLM-as-a-judge** | Phi-3 itself asked to answer YES/NO about each candidate | Outputs weren't clean YES/NO, since the small local model doesn't follow the format reliably; a stronger API model would be a better judge |

### Building the ORM dataset
1. Generate **4 sampled solutions** (`temperature=0.7`, `max_new_tokens=1000`) for each of the first **100** GSM8K train questions.
2. Extract each solution's final number and compare it to the ground truth: **label 1 = correct, 0 = incorrect**.
3. Flatten into rows of the form:
   ```json
   {"question": "...", "solution": "...", "label": 1}
   ```
4. Write to `data_orm.jsonl` (400 rows: 323 correct, 77 incorrect) and make an initial sequential split into `train_orm.jsonl` / `val_orm.jsonl` / `test_orm.jsonl` (300 / 52 / 48 rows).

---

## Notebook 2 — `ORM.ipynb`

### Architecture: frozen backbone + linear probe
```
Question + Candidate solution
        │  (chat template + tokenizer)
        ▼
Phi-3-mini (frozen)
        ▼
Last-layer hidden states  [batch, seq, 3072]
        ▼
Masked mean pooling (ignores padding via attention_mask)
        ▼
Linear(3072 → 1)   ← only trainable part (3,073 params)
        ▼
Logit → sigmoid → P(solution is correct)
        ▼
BCEWithLogitsLoss
```
Input format fed to the model:
```
Problem:
{question}

Candidate Solution:
{solution}

Determine whether the candidate solution is correct.
```
The notebook first verifies each component on a single example (tokenization, forward pass, pooling, loss, one optimizer step) before wrapping them into the `ORMModel` class and training on the full dataset (`AdamW`, `lr=1e-3`, batch size 4, `max_length=2048`).

### Experiment 1: plain BCE on the sequential split — a misleading result
- Validation accuracy looked great (**96.15%**), but the validation set had **51 correct vs 1 incorrect** solutions, so the model could just predict "correct" for everything.
- The class balance across splits was badly skewed, so the data was re-split.

### Fix 1: question-level stratified split
- All 4 solutions of a question stay in the same split (**no question leakage**, verified with set intersections).
- Questions are stratified by how many of their 4 solutions are incorrect (0–4), with `random.seed(42)`.

| Split | Questions | Rows | Correct | Incorrect |
|-------|-----------|------|---------|-----------|
| Train | 75 | 300 | 244 | 56 |
| Val   | 12 | 48  | 39  | 9  |
| Test  | 13 | 52  | 40  | 12 |

### Experiment 2: plain BCE on the balanced-by-question split
Still ignored the minority class: **all 9 incorrect validation solutions were classified as correct** (incorrect-class recall = 0).

### Fix 2: class-weighted BCE
Per-sample loss weights: **label 0 (incorrect) → 244/56 ≈ 4.36**, label 1 → 1.0 (per-sample weights rather than `pos_weight`, since the goal is to up-weight the *minority negative* class). The training loss number goes up simply because the loss definition changed.

### Fix 3: decision-threshold tuning on validation

| Setup | Accuracy | Incorrect F1 | Macro F1 |
|-------|----------|--------------|----------|
| Plain BCE | 81.2% | 0.0% | 44.8% |
| Weighted BCE, threshold 0.5 | 85.4% | 58.8% | 75.0% |
| Weighted BCE, threshold 0.4 | 87.5% | 62.5% | 77.5% |

Threshold **0.4** was selected on the validation set. With weighted BCE at 0.5, incorrect-class recall on validation went from 0/9 to 5/9.

### Final test evaluation
| Metric | Result |
|--------|--------|
| Classification accuracy (threshold 0.4) | **78.8%** |
| Best-of-4 ranking accuracy | **100%** (13/13 questions) |

**How to read these two numbers:** they measure different things. Classification asks whether each probability is on the right side of the threshold; ranking asks whether the correct candidate gets a *higher* score than the wrong ones. An ORM can be a useful ranker even when its calibration/thresholding is imperfect.

**Caveats:**
- The test set is only 13 questions. Several of them have all four solutions correct, where best-of-4 is trivially satisfied, so the 100% figure is optimistic. A larger test set, or reporting only questions that mix correct and incorrect candidates, would be a stronger check.
- The same 100 questions provide both the training and evaluation data, and the base model is also the generator, so the results say little about generalization to other models or problem distributions.

