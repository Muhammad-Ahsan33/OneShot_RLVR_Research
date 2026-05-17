# OneShot_RLVR_Research

# One-Shot RLVR: Reproducing "RL for Reasoning in LLMs with One Training Example"

A resource-constrained reproduction of the NeurIPS 2025 paper:
> **Reinforcement Learning for Reasoning in Large Language Models with One Training Example**  
> Yiping Wang et al. — [arXiv:2504.20571](https://arxiv.org/abs/2504.20571)

This repository implements 1-shot RLVR on a single **Tesla T4 (16 GB)** using 4-bit QLoRA + TRL,
instead of the paper's 8x A100 setup.

---

## What is 1-Shot RLVR?

The paper shows that training a language model using **just one math problem** via Reinforcement
Learning with Verifiable Rewards (RLVR) can improve its reasoning performance almost as much as
training on 1,200+ examples.

- Base model (Qwen2.5-Math-1.5B) starts at **36.0%** on MATH500
- After 1-shot RLVR: **73.6%** on MATH500
- Training example used: a single wind pressure algebra problem (π₁)

---

---

## Notebooks Overview

### Notebook 1 — Core 1-Shot RLVR (`rlvr_assignment.ipynb`)
- Trains Qwen2.5-Math-1.5B on π₁ (wind pressure problem) using binary outcome reward
- Implements 2-shot experiment using π₁ + π₁₃ (geometry problem)
- Evaluates on a small custom problem set and 10-problem MATH500 subset
- **Key result:** Demonstrates training loop works on T4 with 4-bit QLoRA

### Notebook 2 — Format Reward Baseline (`rlvr_format_reward.ipynb`)
- Reproduces the paper's Appendix C.2.3 format reward experiment
- Reward is given for any output containing `\boxed{}` regardless of correctness
- Tracks both format compliance and answer accuracy separately
- **Key result:** Shows format correction is a significant component of total improvement

### Notebook 3 — MATH500 Evaluation (`rlvr_one_shot_math500.ipynb`)
- Loads the official HuggingFaceH4/MATH-500 benchmark
- Evaluates baseline model on 200 problems across all 7 subject categories
- Runs 1-shot RLVR training and post-training evaluation
- Reports per-subject accuracy to match paper Table 3
- **Key result:** Baseline accuracy 10.5% on 200-problem subset (paper: 36.0% on 500)

---

## Our Setup vs Paper

| Setting | Paper | This Repo |
|---|---|---|
| Model | Qwen2.5-Math-1.5B | Qwen2.5-Math-1.5B |
| Precision | fp16 (full) | 4-bit NF4 + fp16 compute |
| GPU | 8x A100 (640 GB) | 1x Tesla T4 (16 GB) |
| Framework | verl | TRL v0.15.2 |
| Trainable params | 1.5B (100%) | 18.5M (1.18% via LoRA) |
| Batch size | 128 | 2 |
| Rollouts per prompt | 8 | 2 |
| Max response tokens | 3072 | 128–512 |
| Max prompt tokens | 1024 | 256 |
| Training steps | 2000 | 30–400 |
| MATH500 problems | 500 | 10–200 (subset) |

---

## Key Hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| `lora_r` | 16 | LoRA rank — controls adapter capacity |
| `lora_alpha` | 32 | LoRA scaling factor (alpha/r = 2) |
| `lora_dropout` | 0.05 | Regularization for LoRA layers |
| `load_in_4bit` | True | 4-bit NF4 quantization |
| `double_quant` | True | Quantize quantization constants (saves ~0.4 GB) |
| `per_device_batch` | 2 | Examples per GPU per step |
| `num_generations` | 2 | Rollouts per prompt for GRPO |
| `temperature` | 0.6 | Rollout generation temperature (same as paper) |
| `beta` (KL coeff) | 0.001 | KL divergence penalty (same as paper) |
| `learning_rate` | 5e-6 | AdamW learning rate |
| `dataset_repeat` | 64 | Copies of π₁ to fill training batch |

---

## Training Examples Used

**π₁ — Wind Pressure Algebra Problem (Table 2 in paper)**

**π₁₃ — Circle Geometry Problem (Table 21 in paper)**

---

## Results

### Baseline vs Post-Training (Notebook 3, 200-problem MATH500 subset)

| Subject | Our Baseline | Paper Baseline | Paper after π₁ |
|---|---|---|---|
| Algebra | 13.5% | 37.1% | 88.7% |
| Counting & Probability | 6.7% | 31.6% | 63.2% |
| Geometry | 12.5% | 39.0% | 56.1% |
| Intermediate Algebra | 2.4% | 43.3% | 62.9% |
| Number Theory | 4.5% | 24.2% | 79.0% |
| Prealgebra | 28.1% | 36.6% | 81.7% |
| Precalculus | 0.0% | 33.9% | 64.3% |
| **Overall** | **10.5%** | **36.0%** | **74.0%** |

### Why Our Results Differ

- **Training steps:** The paper's improvement happens between steps 100–1540. Our 30–400 step runs stop before this window
- **Token limits:** MATH500 needs 500–1500 tokens of reasoning. Our 128–512 token cap cuts off answers mid-solution
- **Quantization:** 4-bit quantization + 1.18% trainable parameters limits how strongly the policy can shift
- **Evaluation subset:** 10–200 problems vs paper's 500 introduces high variance
- **Prompt mismatch:** We did not use the official Qwen2.5-Math evaluation template in all notebooks

---

## Installation

```bash
pip install peft==0.14.0
pip install bitsandbytes==0.45.0
pip install accelerate==1.2.1
pip install datasets==3.2.0
pip install trl==0.15.2
pip install transformers==4.46.3
pip install sentencepiece
```

Or run directly on **Google Colab** — each notebook includes installation cells.

---

## How to Run

1. Open any notebook in Google Colab
2. Set runtime to **GPU (T4)**
3. Run all cells in order — installation cells are included
4. Notebook 3 is recommended as the starting point for MATH500 evaluation

> **Note:** Full MATH500 evaluation (500 problems) takes ~40–45 minutes on T4.
> Each notebook uses `math500.select(range(200))` to limit to 200 problems for time.
> Change to `math500` for full evaluation.

---

## Memory Budget (T4 = 16 GB)
4-bit model (Qwen2.5-Math-1.5B) : ~1.5 GB
LoRA adapters                    : ~0.5 GB
Reference model (4-bit)          : ~1.5 GB
Optimizer states (CPU offload)   : ~2.0 GB
Rollouts (2 per prompt)          : ~4.0 GB
Activations + overhead           : ~4.0 GB
─────────────────────────────────────────
Total estimate                   : ~13.5 GB

---

## Original Paper

```bibtex
@inproceedings{wang2025oneshot,
  title     = {Reinforcement Learning for Reasoning in Large Language Models
               with One Training Example},
  author    = {Yiping Wang and Qing Yang and Zhiyuan Zeng and Liliang Ren and
               Liyuan Liu and Baolin Peng and Hao Cheng and Xuehai He and
               Kuan Wang and Jianfeng Gao and Weizhu Chen and Shuohang Wang and
               Simon Shaolei Du and Yelong Shen},
  booktitle = {NeurIPS},
  year      = {2025},
  url       = {https://arxiv.org/abs/2504.20571}
}
```

Official code: [https://github.com/ypwang61/One-Shot-RLVR](https://github.com/ypwang61/One-Shot-RLVR)

---

## Acknowledgements

This implementation was created as part of a critical analysis assignment on the One-Shot RLVR paper.
All experiments were run on free-tier Google Colab T4 GPUs.
