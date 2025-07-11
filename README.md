# UCSD🔱 ECE228: Machine Learning for Physical Applications (Spring 2025)

[English Version](README.md) | [中文版](README_CN.md)

Welcome to the official course repository for **ECE228: Machine Learning for Physical Applications** at UCSD, Spring 2025. This repo contains all the completed assignments and labs demonstrating fundamental concepts of ML in physical sciences, such as regression, sparsity, and deep learning for PDEs and dynamical systems.

Explore each folder for detailed implementation, report writeups, and source code.


## 📐 Assignment 1: Sparse Linear Regression
- Implemented linear regression and LASSO from scratch
- Developed ISTA algorithm to solve the L1-regularized problem
- Compared results with `scikit-learn` implementation
- Conducted experiments on synthetic and real-world datasets

## 🧠 Assignment 2: Transformers & Attention
- Built Transformer modules from scratch using PyTorch
- Implemented scaled dot-product attention, multi-head attention, and positional encoding
- Applied Transformer encoders to classification tasks
- Trained and evaluated models on real-world data

## 🌊 Assignment 3: Neural Operators
- Introduced the concept of learning mappings between function spaces
- Implemented FNO (Fourier Neural Operator) using spectral convolution
- Modeled physical systems like fluid dynamics with operator learning
- Compared generalization across resolutions and domains

## 🚦 Final Project – Adaptive Spatial-Temporal Transformer Network&nbsp;(**ASTTN-Weather**)
**Goal:** Accurate short-term traffic-flow forecasting by capturing road-network topology *and* exogenous weather effects.

| Aspect | Implementation Highlights |
| ------ | ------------------------- |
| **Model** | Extended **ASTTN** [[paper](https://arxiv.org/abs/2207.05064)] with:<br>• *Static* adjacency from GIS road graph<br>• Joint spatial & temporal self-attention in a single block<br>• Multi-modal fusion of weather features (temperature, precipitation, wind)<br>• Lightweight output head with weighted multi-horizon loss |
| **Variants** | Baseline ASTTN, ASTTN-Weather (ours), and ablations w/-wo weather or adaptive edges |
| **Dataset** | 1-min loop-detector volume & speed + NOAA hourly weather, aligned to 15-min intervals |
| **Results** | AGSTTN-Weather ↓MAE by 9 % and ↑R² by 6 % over vanilla AGSTTN on 15-min horizon |
| **Reproducibility** | `train.py` supports CPU-only training (PyTorch 2.7.1), configurable via YAML; logs, checkpoints, and tensorboard summaries auto-saved |
| **Visualization** | Scripts for attention heat-maps and per-edge importance under different weather scenarios |


Enjoy your journey into scientific machine learning! 🚀
