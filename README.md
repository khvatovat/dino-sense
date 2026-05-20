# DINO-Sense

Self-supervised spatial understanding from satellite imagery — bridging perception and world models.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Work in progress — figures and results added as experiments complete.

## Quick Start

```bash
pip install -e .
python scripts/extract_features.py --dataset eurosat
python scripts/generate_figures.py
```

## Project Structure

```
src/dino_sense/   DINOv2 wrapper, data loaders, eval, segmentation, transition model
scripts/          CLI: feature extraction, transition training, figure generation
notebooks/        Colab-ready demo
configs/          Hyperparameters
app.py            Gradio interactive demo (HuggingFace Spaces)
```

## Data Setup

**EuroSAT** — downloaded automatically on first run.

**LoveDA** — download from [Zenodo](https://doi.org/10.5281/zenodo.5706578) and extract:
```
data/loveda/
  Train/Urban/{images_png,masks_png}/
  Train/Rural/{images_png,masks_png}/
  Val/Urban/{images_png,masks_png}/
```

## Acknowledgments

DINOv2 (Meta FAIR), EuroSAT (Sentinel-2), LoveDA (Zenodo).
Inspired by [DINO-WM](https://arxiv.org/abs/2411.04983) (Zhou, Pan, LeCun, Pinto — ICML 2025).
