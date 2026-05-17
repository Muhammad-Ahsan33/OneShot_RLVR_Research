# 1-Shot RLVR: Reinforcement Learning with One Training Example

**Official Reproduction** of the paper:  
**"Reinforcement Learning for Reasoning in Large Language Models with One Training Example"** (NeurIPS 2025)

> One example (π₁) boosts Qwen2.5-Math-1.5B from **36.0% → 73.6%** on MATH500.

---

## 📋 Overview

This repository contains **Colab-ready notebooks** implementing **1-shot RLVR** (and few-shot variants) as described in the paper. It reproduces the surprising result that a **single training example** can dramatically improve mathematical reasoning.

### Key Highlights
- Runs on **Tesla T4** (15.6GB) using **4-bit quantization + LoRA**
- Exact π₁ (wind pressure) and π₁₃ examples from the paper
- Binary outcome reward, format reward, and ablation experiments
- Strong generalization across math benchmarks

---

## 📁 Files

| Notebook | Purpose |
|---------|--------|
| `rlvr_output_binary_reward.ipynb` | Binary reward experiment |
| `rlvr_format_reward.ipynb` | Format reward baseline |
| `rlvr-math500_baseline_evaluation.ipynb` | Baseline evaluation |
| `rlvr_threshold_change.ipynb` | Threshold experiments |

---

## 🚀 Quick Start

1. Open **`rlvr_output_binary_reward.ipynb`** or  **`rlvr_format_reward.ipynb`** in Google Colab
2. Run cells sequentially (GPU → Install → Config → Train → Evaluate)
3. Change `num_train_steps` and `dataset_repeat` as needed

---

## ✨ Features

- **4-bit NF4 quantization** + Double Quantization
- LoRA (rank 16) on Qwen2.5-Math-1.5B
- GRPO Trainer from TRL
- Paper-reproducing π₁ & π₁₃ examples
- 1-shot and 2-shot support
- Binary + Format reward options
- Easy configuration in one cell


---

## 🛠️ Main Configuration

```python
CFG = {
    "model_name": "Qwen/Qwen2.5-Math-1.5B",
    "load_in_4bit": True,
    "quant_type": "nf4",
    "lora_r": 16,
    "num_generations": 2,
    "max_new_tokens": 512,
    "per_device_batch": 2,
    "dataset_repeat": 64,      # Paper duplication trick
    "num_train_steps": 50,     
    "learning_rate": 5e-6,
    "temperature": 0.6,
}
