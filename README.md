# ELM: Continual Self-Improvement with Lightweight Experiential Latent Memories

[![Paper](https://img.shields.io/badge/arXiv-2606.17803-b31b1b.svg)](https://arxiv.org/abs/2606.17803)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Replication of the paper **"Continual Self-Improvement with Lightweight Experiential Latent Memories"** (*Dorovatas, Kalaj, Aljundi — Toyota Motor Europe & University of Trento, 2026*).

---

## 🌟 Key Insights & Replicated Pipelines

Large language models are fundamentally stateless at test time, discarding the reasoning compute spent on solving complex problems. **ELM (Experiential Latent Memory)** distills inference-time computation into compact, modular soft-prompt memories ($\sim 0.001\%$ of model parameters) trained with Group Relative Policy Optimization (GRPO) using self-supervised internal rewards (majority voting), completely avoiding catastrophic forgetting.

```
Incoming Problem x_i (Test-time Stream)
       │
       ├──────────────────────────────────┐
       │ (1) Memory Retrieval             │ (2) Memory Creation (after response)
       ▼                                  ▼
Retrieve m* from Pool M            Generate K=8 completions from π_θ
(via Input or KV similarity)       Extract majority vote consensus y*
       │                           Assign reward R(y_j, y*) ∈ {0, 1}
       ▼                           Compute GRPO advantages A_j
Generate a_zs and a_mem            Train 20-token soft prompt m^(i) (10 steps)
       │                           Add m^(i) and key k^(i) to Memory Pool M
       ▼
If a_zs == a_mem ──> Output answer
Else ──> Verifier(a_zs, a_mem) ──> Output higher-scoring response
```

### 🔬 Replicated Pipelines & Baselines:
1. **Online Continual Self-Improvement (Algorithm 1 / Table 3)**:
   - Streaming evaluation without prior dataset access.
   - Dual retrieval: **$ELM_{input}$** (input embedding mean-pooling) and **$ELM_{kv}$** (layer-wise QKV activations).
   - Response routing + Process Reward Model (**PRM**) verifier (`Qwen2.5-Math-PRM-7B`).
2. **Offline Per-Sample Analysis (Table 1 & Table 2)**:
   - Evaluates memory specialization, **Oracle ZS vs Mem**, **Negative Flip Rate** ($1 \to 0$), and **IoU Overlap**.
3. **Cross-Dataset Transfer (Figure 3)**:
   - Evaluates memories learned on AMC23 transferred directly to AIME24 (and vice versa).
4. **All Comparison Baselines**:
   - Zero-Shot Chain-of-Thought
   - Online In-Context Learning (ICL)
   - Reasoning Bank (Reflection + Strategy summarization)
   - Log-Augmented Generation (LAG & LAG-TTS)
   - Verifier@2
   - Full Model Offline TTRL & GRPO

---

## 📂 Project Structure

```
harness-memory/
├── elm/
│   ├── config.py                 # Hyperparameters (K=8, S=10, lr=1e-4, len=20, beta=0.01)
│   ├── memory/
│   │   ├── soft_prompt.py        # Soft prompt memory unit (m_x in R^{20 x d})
│   │   ├── lora_memory.py        # LoRA memory ablation (r=16)
│   │   ├── prefix_tuning.py      # Prefix tuning ablation
│   │   ├── pool.py               # Memory pool M storage & persistence
│   │   └── indexing.py           # Input and KV cosine similarity retrieval
│   ├── training/
│   │   ├── grpo.py               # DR-GRPO policy trainer with advantage normalization
│   │   ├── internal_reward.py    # Majority voting consensus extractor and reward assigner
│   │   └── sft_trainer.py        # SFT memory trainer ablation
│   ├── verifier/
│   │   └── prm.py                # PRM verifier (Qwen2.5-Math-PRM-7B) & response router
│   ├── pipelines/
│   │   ├── online_elm.py         # Algorithm 1: Streaming Online Continual Self-Improvement
│   │   ├── offline_elm.py        # Per-sample offline training & flip analysis
│   │   ├── cross_transfer.py     # Cross-dataset memory transfer
│   │   └── baselines/            # Zero-shot, ICL, Reasoning Bank, LAG, Verifier@2, TTRL
│   ├── data/
│   │   ├── benchmarks.py         # Loaders for AMC23, AIME24, MATH500
│   │   └── math_eval.py          # \\boxed{} parsing and mathematical grading
│   ├── evaluator.py              # Pass@1, flip rates, IoU, and markdown table formatter
│   └── cli.py                    # Unified CLI entrypoint
├── colab_runner.ipynb            # Google Colab notebook for running experiments on GPU
├── run_experiments.py            # Unified benchmark execution script
├── pyproject.toml                # Project configuration, dependencies, and CLI entrypoints
└── tests/
    └── test_elm_pipeline.py      # Test suite
```

---

## 🚀 Quick Start

### 1. Installation
```bash
# Install package in editable mode with dependencies
pip install -e .

# Or with dev & viz optional dependencies
pip install -e ".[dev,viz]"
```

### 2. Running Online ELM via CLI
You can use either the installed `elm` command directly or `python -m elm.cli`:

```bash
# Run online streaming ELM on AMC23 with Qwen2.5-Math-7B
elm online --benchmark AMC23 --model Qwen/Qwen2.5-Math-7B --retrieval input

# Run with layer-wise KV retrieval (ELM_kv)
elm online --benchmark AMC23 --retrieval kv

# Run in mock mode (fast CPU test without downloading 7B weights)
elm online --benchmark mock --mock_run
```

### 3. Running Offline & Transfer Pipelines
```bash
# Offline evaluation on sample memories
python -m elm.cli offline --benchmark AMC23 --mock_run

# Cross-dataset transfer: AMC23 -> AIME24
python -m elm.cli transfer --source AMC23 --target AIME24 --mock_run

# Run Baselines
python -m elm.cli baseline --type zero_shot --benchmark AMC23 --mock_run
python -m elm.cli baseline --type reasoning_bank --benchmark AMC23 --mock_run
python -m elm.cli baseline --type lag --benchmark AMC23 --mock_run
python -m elm.cli baseline --type verifier@2 --benchmark AMC23 --mock_run
```

### 4. Running Full Reproduction Suite
```bash
python run_experiments.py
```

---

## ☁️ Running on Google Colab (GPU / Remote CLI)

You can run experiments either via the interactive notebook **`colab_runner.ipynb`** or through the **Colab SSH / CLI** configured in your terminal:

```bash
# Connect to Colab GPU via terminal
ssh colab

# Run the training inside persistent tmux
tmux new -s elm_run
python run_experiments.py
```

