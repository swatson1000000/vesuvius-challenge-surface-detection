# Option C Implementation: nnU-Net + Topology-Aware Loss

This directory contains the implementation of Option C from the PLAN.md (section 2.1).

## Overview

This implementation combines:
1. **nnU-Net** - State-of-the-art medical segmentation framework
2. **Topology-Aware Losses** - clDice, connectivity losses for topological correctness
3. **Morphological Post-Processing** - Topology correction and instance separation
4. **Competition Metrics** - Direct optimization for TopoScore, SurfaceDice, and VOI

## Files

- `nnunet_topo_wrapper.py` - Main nnU-Net wrapper with topology-aware training
- `topology_losses.py` - clDice, connectivity, and boundary losses
- `morphology_postprocess.py` - Post-processing pipeline for topology correction
- `train.py` - Training script with topology optimization
- `inference.py` - Inference pipeline with topology fixes
- `config.yaml` - Configuration file for training

## Key Features

### 1. Topology-Aware Loss Function
```
total_loss = 0.4 * dice_loss + 
             0.2 * focal_loss + 
             0.2 * boundary_loss + 
             0.1 * clDice_loss + 
             0.1 * connectivity_loss
```

### 2. Post-Processing Pipeline
- Remove small components (reduce false positives in k=0)
- Fill small holes (reduce cavities in k=2)
- Morphological operations (remove spurious bridges)
- Instance separation via watershed (prevent mergers)

### 3. Competition Metric Integration
- Direct validation using TopoScore, SurfaceDice@τ=2.0, and VOI
- Stratified cross-validation by scroll ID
- Metric-aligned hyperparameter tuning

## Usage

### Training
```bash
python train.py --config config.yaml --fold 0
```

### Inference
```bash
python inference.py --checkpoint best_model.pth --input test_images/ --output predictions/
```

## Expected Performance

- Baseline nnU-Net: ~0.60-0.70 CV score
- With topology losses: ~0.75-0.80 CV score
- With post-processing: ~0.80-0.85 CV score
