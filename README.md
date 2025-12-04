# Temp-SCONE: A Novel Out-of-Distribution Detection and Domain Generalization Framework for Wild Data with Temporal Shift

Temp-SCONE is a temporally-consistent extension of SCONE designed for dynamic domains with temporal shifts. It addresses the limitation of SCONE in static environments by introducing temporal regularization to improve robustness under evolving data distributions.

## Overview

Temp-SCONE extends SCONE's energy margin-based framework by introducing a temporal regularization loss that stabilizes model confidence across evolving distributions. The method leverages Average Thresholded Confidence (ATC) and Average Confidence (AC) metrics to monitor prediction stability on both in-distribution and covariate-shifted samples. When confidence drift between consecutive timesteps exceeds a tolerance threshold, Temp-SCONE applies a differentiable temporal loss with adaptive weighting, penalizing instability while preserving flexibility for gradual shifts.

This temporal regularization is jointly optimized with cross-entropy and energy-based OOD objectives, enabling Temp-SCONE to maintain strong in-distribution performance while improving robustness to covariate shifts and enhancing semantic OOD detection under dynamic open-world conditions.

## Key Contributions

- Temporal regularization loss using ATC and AC metrics to stabilize confidence across time
- Improved robustness to temporal shifts in dynamic environments
- Theoretical insights linking temporal consistency to generalization error bounds

## Datasets

Temp-SCONE has been evaluated on:
- **Dynamic datasets**: CLEAR, YearBook
- **Static datasets**: CIFAR-10, CINIC-10, Imagenette, STL-10

## Code Structure

The main training code is located in:
- **Training**: `CIFAR/train.py`
- **Dataset Loading**: `CIFAR/load_any_dataset.py`

### Key Functions

**Temporal Regularization:**
- `compute_temporal_loss()` (line 533 in `train.py`): Computes temporal loss based on changes in average confidence between consecutive cities
- `compute_average_confidence()` (line 853 in `train.py`): Calculates Average Confidence (AC) metric for monitoring prediction stability

**Dataset Loading:**
- `load_cifar()` (line 170 in `load_any_dataset.py`): Loads CIFAR-10 dataset with corrupted variants
- `load_Imagenette()` (line 345 in `load_any_dataset.py`): Loads Imagenette dataset for multi-city training
- `load_cinic10()` (line 576 in `load_any_dataset.py`): Loads CINIC-10 dataset

**Training Pipeline:**
- Main training loop (line 1262 in `train.py`): Implements sequential training across multiple cities with temporal constraints
- Loss computation (line 700 in `train.py`): Combines cross-entropy, energy-based OOD, and temporal regularization losses with weighted components
- `mix_batches()` (line 433 in `train.py`): Creates mixed batches with in-distribution, corrupted, and OOD samples based on pi_1 and pi_2 proportions

**Multi-City Training:**
- City-wise confidence tracking (line 1313 in `train.py`): Stores average confidence metrics per city for temporal comparison
- Temporal loss integration (line 645 in `train.py`): Computes temporal loss at each training epoch comparing current metrics with previous city averages

## Implementation Details

The temporal regularization is integrated into the SCONE framework as follows:

1. **Confidence Monitoring**: During training, Average Confidence (AC) is computed on both clean and corrupted test sets using `compute_average_confidence()`.

2. **Temporal Loss Computation**: `compute_temporal_loss()` compares current AC values against the average AC from the previous city. If the drift exceeds a tolerance threshold (`temporal_delta_ac = 0.05`), a penalty is applied.

3. **Joint Optimization**: The temporal loss is combined with SCONE's energy-based losses in the final objective function (line 700), where the temporal component contributes 40% of the total loss weight.

4. **Multi-Domain Training**: The code supports sequential training across different domains (cities) where each city represents a distinct data distribution with temporal evolution.

## Usage

The training script supports multiple datasets and can be configured through command-line arguments. See `CIFAR/train.py` for available options and hyperparameters. The temporal regularization is integrated into the SCONE loss function and is automatically applied during training across sequential domains (cities).

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{naiknaware2025tempscone,
  title={Temp-{SCONE}: A Novel Out-of-Distribution Detection and Domain Generalization Framework for Wild Data with Temporal Shift},
  author={Aditi Naiknaware and Sanchit Singh and Hajar Homayouni and Salimeh Sekeh},
  booktitle={NeurIPS 2025 Workshop: Reliable ML from Unreliable Data},
  year={2025},
  url={https://openreview.net/forum?id=9We0ZjdP6u}
}
```
