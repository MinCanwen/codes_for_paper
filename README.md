# Robust Physics-Informed Neural Networks for Equation Discovery

This repository provides a simplified and self-contained implementation accompanying my prior research on
improving the robustness and generalization of physics-informed neural networks (PINNs) for equation discovery
from noisy and limited observations.


---

## Research Motivation

Physics-informed neural networks are widely used to infer governing equations of physical systems by enforcing
differential operators in the loss function. However, empirical studies show that this direct enforcement can
introduce stiff training dynamics, leading to optimization instability and degraded generalization, especially
in the presence of Gaussian noise or higher-order derivatives.

The central research question explored in this work is:

> **How can physical structure be discovered more robustly without increasing computational cost?**

---

## Method Overview

The key idea is to reformulate how physical constraints are incorporated during training:

- Differential operators are replaced with finite-difference approximations to reduce numerical stiffness.
- Additional network outputs are introduced to represent latent physical quantities explicitly.
- The framework remains compatible with standard PINN training pipelines and does not increase computational
  complexity.

This reformulation preserves physical interpretability while improving optimization stability.

---

## Experimental Setup

All benchmark experiments are conducted under controlled conditions:

- Same datasets, network architectures, hyperparameters, and number of training iterations
- Gaussian noise applied to observations to test robustness
- Evaluation across multiple benchmark PDEs

Performance is measured in terms of:
- Successful equation discovery
- Training stability
- Computational runtime

---

## Results Summary

On wave equation benchmarks with Gaussian noise:
- The modified framework achieved **100% convergence**, compared to **50% for standard PINNs**
- Runtime was reduced by approximately **one minute**
- Performance gains were more pronounced for problems involving higher-order derivatives

These results indicate that numerical reformulation plays a critical role in improving robustness and
generalization in physics-informed learning.

---


## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

(Exact versions are not critical for reproducing the qualitative behavior.)

---

## Usage

Example scripts for running benchmark experiments are provided:

```bash
python run_wave_equation.py
