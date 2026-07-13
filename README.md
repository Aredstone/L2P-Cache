<div align="center">

# [CVPR 2026] L2P: Learnable Linear Predictor for Efficient Diffusion Models

**Beyond Fixed Formulas: Data-Driven Linear Predictor for Efficient Diffusion Models**

[Zhirong Shen](https://arxiv.org/abs/2604.26365)<sup>1</sup>, Rui Huang<sup>1</sup>, Jiacheng Liu<sup>1</sup>, Chang Zou<sup>1</sup>, Peiliang Cai<sup>1</sup>, Shikang Zheng<sup>1</sup>, Zhengyi Shi<sup>1</sup>, Liang Feng<sup>1</sup>, [Linfeng Zhang](https://arxiv.org/abs/2604.26365)<sup>1</sup>

[[📄 Paper](https://arxiv.org/abs/2604.26365)] · [[🤗 FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)] · [[💻 Code](https://github.com/Aredstone/L2P-Cache)]

[![arXiv](https://img.shields.io/badge/arXiv-2604.26365-b31b1b.svg)](https://arxiv.org/abs/2604.26365)
[![Conference](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://arxiv.org/abs/2604.26365)
[![License](https://img.shields.io/badge/License-Flux--dev%20NC-lightgrey.svg)](model_licenses/LICENSE-FLUX1-dev)

</div>

---

## 🔥 News

- **`2026/04/29`** 🎉 L2P is accepted by **CVPR 2026**! Paper is available on [arXiv](https://arxiv.org/abs/2604.26365).
- **`2026/07/13`** 🚀 **L2P-FLUX** code is released — FLUX.1-dev inference, weight training, and evaluation scripts.
- **Coming soon** 🔜 L2P for **Qwen-Image** and **HunyuanVideo**, along with full experiment reproduction.

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Highlights](#-highlights)
- [Method](#-method)
- [Main Results](#-main-results)
- [Roadmap](#%EF%B8%8F-roadmap)
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Acknowledgements](#-acknowledgements)
- [Citation](#-citation)

---

## 📖 Abstract

> To address the high sampling cost of Diffusion Transformers (DiTs), feature caching offers a training-free acceleration method. However, existing methods rely on **hand-crafted forecasting formulas** that fail under aggressive skipping.
>
> We propose **L2P (Learnable Linear Predictor)**, a simple data-driven caching framework that replaces fixed coefficients with **learnable per-timestep weights**. Rapidly trained in **~20 seconds** on a single GPU, L2P accurately reconstructs current features from past trajectories.
>
> L2P significantly outperforms existing baselines: it achieves a **4.55× FLOPs reduction** and **4.15× latency speedup** on FLUX.1-dev, and maintains high visual fidelity under up to **7.18× acceleration** on Qwen-Image models, where prior methods show noticeable quality degradation.

---

## ✨ Highlights

| | |
| :--- | :--- |
| 🧠 **Data-driven** | Replace fixed Taylor / reuse formulas with learnable linear weights |
| ⚡ **Ultra-fast training** | ~20 seconds on a single GPU, no DiT fine-tuning required |
| 🎯 **Aggressive skipping** | Stable quality where hand-crafted predictors break down |
| 🔌 **Plug-and-play** | Drop-in acceleration on top of existing DiT inference pipelines |

---

## 🔬 Method

At each denoising step, L2P maintains a history of past output features. On **cache steps**, instead of running the full transformer, L2P predicts the current feature as a causal linear combination of history — weights are **per-timestep** and learned from data.

```mermaid
flowchart LR
    A[Past Features<br/>x₀, x₁, …, xₜ₋₁] --> B[Learnable Weights<br/>Wₜ · lower-triangular]
    B --> C[Predicted Feature x̂ₜ]
    C --> D[Skip Transformer]

    E[Full Step] --> F[Run Transformer]
    F --> G[Update History]
    G --> A
```

**Training objective:** minimize MSE between predicted and ground-truth features over denoising trajectories. The weight matrix is lower-triangular (causal) and shared across spatial tokens after FFT decomposition.

---

## 📊 Main Results

### FLUX.1-dev (this repo)

| Method | FLOPs ↓ | Latency ↓ | Training Cost |
| :--- | :---: | :---: | :---: |
| Full compute | 1.00× | 1.00× | — |
| **L2P (Ours)** | **4.55×** | **4.15×** | **~20 s / 1 GPU** |

### Qwen-Image *(coming soon)*

| Method | Max Acceleration | Quality under Aggressive Skipping |
| :--- | :---: | :--- |
| Prior caching methods | — | Noticeable degradation |
| **L2P (Ours)** | **7.18×** | High visual fidelity |

> Full benchmark tables, Geneval / ImageReward / DrawBench numbers, and comparison with TaylorSeer, ToCa, FORA will be released with the complete experiment suite.

---

## 🗺️ Roadmap

| Module | Status | Description |
| :--- | :---: | :--- |
| **L2P-FLUX** | ✅ | Inference, weight training, DrawBench200 eval |
| **L2P-Qwen-Image** | 🔜 | Image generation acceleration |
| **L2P-HunyuanVideo** | 🔜 | Video generation acceleration |
| **Full Experiments** | 🔜 | All main paper results & reproduction scripts |

---

## 🛠️ Installation

### Requirements

```
Python >= 3.10
CUDA >= 11.8
PyTorch >= 2.5
```

### Setup

```bash
git clone https://github.com/Aredstone/L2P-Cache.git
cd L2P-Cache

python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .

# Optional: PSNR / SSIM / LPIPS evaluation
pip install -e ".[eval]"
```

### Model Weights

Download [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and set environment variables:

```bash
export FLUX_DEV=/path/to/FLUX.1-dev/flux1-dev.safetensors
export AE=/path/to/FLUX.1-dev/ae.safetensors
export FLUX_MODEL_DIR=/path/to/FLUX.1-dev   # optional: local diffusers layout
export HF_HOME=/path/to/huggingface/cache   # optional
```

> **Tip:** If `FLUX_MODEL_DIR` points to a diffusers-style folder, T5 (`text_encoder_2/`) and CLIP (`text_encoder/`) are loaded locally with their tokenizers. Otherwise they are fetched from HuggingFace.

---

## ⚡ Quick Start

Generate images with L2P acceleration on DrawBench200:

```bash
export PYTHONPATH=src

python src/sample.py \
  --prompt_file predictor/DrawBench200.txt \
  --output_dir outputs/samples \
  --interval 7 \
  --width 1024 --height 1024 \
  --num_steps 50 \
  --device 0
```

Compare against full-compute baseline (`--interval 1`) and evaluate with `evaluate.py`.

---

## 📖 Usage

### Inference

```bash
export PYTHONPATH=src

python src/sample.py \
  --prompt_file predictor/DrawBench200.txt \
  --output_dir outputs/samples \
  --interval 7 \
  --width 1024 \
  --height 1024 \
  --num_steps 50 \
  --seed 0 \
  --device 0
```

| Argument | Description |
| :--- | :--- |
| `--interval 1` | Full compute baseline (transformer at every step) |
| `--interval 7` | L2P accelerated (~5× faster in practice) |
| `--test_FLOPs` | FLOPs profiling mode (no image output) |

**Supported intervals** with tuned step schedules: `5, 6, 7, 8, 10, 16` — see `FULL_STEP_SCHEDULES` in [`src/flux/model.py`](src/flux/model.py).

<details>
<summary><b>GenEval evaluation</b></summary>

```bash
python src/geneval_flux.py \
  /path/to/evaluation_metadata.jsonl \
  --model_name flux-dev \
  --steps 50 --width 1024 --height 1024 \
  --output_dir outputs/geneval
```

</details>

### Train Predictor Weights

L2P learns a **50×50 lower-triangular** weight matrix via MSE on feature trajectories:

```bash
python predictor/train.py \
  --weight_path predictor/weight.txt \
  --device cuda:0
```

**Prerequisites:** pre-extracted feature tensors at `./train/features.pt` and `./valid/features.pt`.

**Training time:** ~20 seconds on a single GPU.

### Evaluation

Compare L2P outputs against a full-compute reference:

```bash
# 1. Generate L2P samples
python src/sample.py --interval 7 --output_dir outputs/l2p ...

# 2. Generate baseline samples
python src/sample.py --interval 1 --output_dir outputs/baseline ...

# 3. Compute metrics
python evaluate.py \
  --test_folder outputs/l2p \
  --reference_folder outputs/baseline \
  --prompt_file predictor/DrawBench200.txt
```

| Metric | Description |
| :--- | :--- |
| PSNR / SSIM / LPIPS | Pixel-level quality vs. baseline |
| CLIP Score | Text-image alignment *(optional)* |
| ImageReward / PickScore | Human preference proxies *(optional)* |

---

## 🏗️ Project Structure

```
L2P-Cache/
├── src/
│   ├── sample.py                  # Batch inference entry point
│   ├── geneval_flux.py            # GenEval benchmark
│   └── flux/
│       ├── model.py               # L2P integration in FLUX forward
│       ├── learnable_utils/       # Linear predictor (FFT + weight apply)
│       ├── l2p_cache.py           # Step scheduling & cache state
│       └── ideas/cache_denoise.py # Cache-aware denoising loop
├── predictor/
│   ├── train.py                   # Weight training (~20s)
│   ├── weight.txt                 # Pre-trained FLUX weights
│   └── DrawBench200.txt           # Benchmark prompts
├── evaluate.py                    # Quality & alignment metrics
└── README.md
```

---

## 👍 Acknowledgements

This project builds upon the excellent open-source work of:

- [FLUX](https://github.com/black-forest-labs/flux) — inference framework
- [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) · [ToCa](https://github.com/Shenyi-Z/ToCa) — prior feature caching methods that inspired this line of research

We thank the authors for their contributions to the community.

---

## 📌 Citation

If you find L2P useful for your research, please cite:

```bibtex
@inproceedings{shen2026l2p,
  title     = {Beyond Fixed Formulas: Data-Driven Linear Predictor for Efficient Diffusion Models},
  author    = {Shen, Zhirong and Huang, Rui and Liu, Jiacheng and Zou, Chang and Cai, Peiliang and Zheng, Shikang and Shi, Zhengyi and Feng, Liang and Zhang, Linfeng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

<div align="center">

**[⬆ Back to Top](#cvpr-2026-l2p-learnable-linear-predictor-for-efficient-diffusion-models)**

</div>
