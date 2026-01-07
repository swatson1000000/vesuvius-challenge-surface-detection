# Vesuvius Challenge - Surface Detection

[![Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

> 3D papyrus surface segmentation from CT scans of carbonized Herculaneum scrolls

## Overview

This repository contains the implementation for the Vesuvius Challenge Surface Detection competition. The goal is to segment papyrus surfaces in 3D CT scans of carbonized scrolls from the Villa dei Papiri library, buried by Mount Vesuvius in AD 79.

**Competition Details:**
- **Deadline:** February 13, 2026
- **Prize Pool:** $200,000 (Top 10)
- **Inference Time Limit:** 9 hours (CPU/GPU)

## Problem Statement

Build a 3D segmentation model to trace scroll surfaces through complex:
- Tightly wound, compressed papyrus layers
- Noise and artifacts from carbonization
- Complex folds and gaps
- Variable scroll conditions

The challenge requires maintaining topological correctness while handling these difficulties.

## Project Structure

```
.
├── bin/                          # Main implementation (Option C)
│   ├── train.py                  # Training script
│   ├── inference.py              # Inference pipeline
│   ├── nnunet_topo_wrapper.py    # nnU-Net + topology wrapper
│   ├── topology_losses.py        # clDice, connectivity losses
│   ├── morphology_postprocess.py # Topology correction
│   ├── config.yaml               # Training configuration
│   └── README.md                 # Implementation details
├── option_c_implementation/      # Alternative implementation
├── topological-metrics-kaggle/   # Competition metrics library
├── train_images/                 # Training CT scans (25GB)
├── train_labels/                 # Training segmentation masks
├── test_images/                  # Test data
├── COMPETITION_INFO.md           # Official competition details
├── PLAN.md                       # Comprehensive strategy document
└── README.md                     # This file
```

## Approach

### Architecture: nnU-Net + Topology-Aware Training

1. **Base Model:** nnU-Net (state-of-the-art medical segmentation)
2. **Topology Losses:** clDice, connectivity, boundary losses
3. **Post-Processing:** Morphological operations for topology correction
4. **Metrics:** Direct optimization for TopoScore, SurfaceDice@τ=2.0, VOI

### Loss Function
```
total_loss = 0.4 * dice_loss + 
             0.2 * focal_loss + 
             0.2 * boundary_loss + 
             0.1 * clDice_loss + 
             0.1 * connectivity_loss
```

## Data

- **Training:** 806 labeled 3D volumes across 6 scrolls
- **Test:** 1 sample from scroll 26002
- **Volume Size:** Typical 320×320×320 voxels
- **Format:** 8-bit grayscale TIFF
- **Label Sparsity:** <1% foreground voxels

### Scroll Distribution
- Scroll 34117: 382 samples (47.4%)
- Scroll 35360: 176 samples (21.8%)
- Scroll 26010: 130 samples (16.1%)
- Scroll 26002: 88 samples (10.9%) - test scroll
- Scroll 44430: 17 samples (2.1%)
- Scroll 53997: 13 samples (1.6%)

## Installation

### Prerequisites
```bash
# Python 3.8+
python --version

# CUDA 11.8+ (for GPU training)
nvidia-smi
```

### Setup
```bash
# Clone repository
git clone <repository-url>
cd vesuvius-challenge-surface-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
cd bin
python train.py --config config.yaml --fold 0
```

### Inference

```bash
cd bin
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input ../test_images/ \
    --output ../predictions/
```

### Quick Start

```bash
cd bin
bash quick_start.sh
```

## Evaluation Metrics

The competition uses a composite metric combining:

1. **Topological Score (TopoScore):** Betti number correctness
   - k=0: Connected components (β₀)
   - k=1: Loops/tunnels (β₁)
   - k=2: Cavities (β₂)

2. **SurfaceDice@τ=2.0:** Surface distance-aware Dice
   
3. **Variation of Information (VOI):** Instance segmentation quality

**Final Score:**
```
score = TopoScore × (1 + SurfaceDice) / (1 + VOI)
```

## Expected Performance

| Stage | Expected CV Score |
|-------|------------------|
| Baseline nnU-Net | 0.60-0.70 |
| + Topology losses | 0.75-0.80 |
| + Post-processing | 0.80-0.85 |

## Documentation

- [`COMPETITION_INFO.md`](COMPETITION_INFO.md) - Official competition details
- [`PLAN.md`](PLAN.md) - Comprehensive implementation strategy (827 lines)
- [`bin/README.md`](bin/README.md) - Implementation-specific documentation

## Development Setup

### Git Hooks

Install Claude code review hooks:
```bash
~/.claude/install-hooks.sh
```

This enables automatic code review before each push, blocking commits with critical/high severity issues.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Competition Links

- **Competition:** https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection
- **Previous Challenge:** https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection
- **Official Website:** https://scrollprize.org/

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Vesuvius Challenge organizers
- nnU-Net framework authors
- Topological metrics library contributors

---

**Status:** Active development | Competition ends February 13, 2026

