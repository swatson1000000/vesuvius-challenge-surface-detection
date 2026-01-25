#!/usr/bin/env python
"""Compare new predictions with ensemble predictions."""

import tifffile
import numpy as np
from scipy import ndimage

# Load predictions
pred_new = tifffile.imread('predictions/1407735.tif')
pred_ensemble = tifffile.imread('predictions/1407735_ensemble.tif')
pred_soft_ensemble = tifffile.imread('predictions/1407735_soft_ensemble.tif')

print("=" * 70)
print("PREDICTION COMPARISON ANALYSIS")
print("=" * 70)

# Convert soft ensemble to binary
pred_soft_binary = (pred_soft_ensemble > 0.5).astype(np.uint8)

print("\n1. NEW SINGLE FOLD PREDICTION (Jan 23)")
print("-" * 70)
print(f"Shape: {pred_new.shape}")
print(f"Unique values: {np.unique(pred_new)}")
fg_new = (pred_new > 0).mean()
print(f"Foreground: {fg_new*100:.2f}%")
labeled_new, nc_new = ndimage.label(pred_new)
print(f"Components: {nc_new}")

print("\n2. ENSEMBLE PREDICTION (Jan 13)")
print("-" * 70)
print(f"Shape: {pred_ensemble.shape}")
print(f"Value range: {pred_ensemble.min()}-{pred_ensemble.max()}")
print(f"Unique values: {len(np.unique(pred_ensemble))} unique")
fg_ensemble = (pred_ensemble > 0).mean()
print(f"Foreground: {fg_ensemble*100:.2f}%")

print("\n3. SOFT ENSEMBLE (binary threshold 0.5)")
print("-" * 70)
fg_soft = (pred_soft_binary > 0).mean()
print(f"Foreground: {fg_soft*100:.2f}%")

print("\n4. CROSS-PREDICTION AGREEMENT")
print("-" * 70)

# Agreement metrics
agreement_new_ensemble = np.mean(pred_new == (pred_ensemble > 0).astype(np.uint8))
agreement_new_soft = np.mean(pred_new == pred_soft_binary)
agreement_ensemble_soft = np.mean((pred_ensemble > 0).astype(np.uint8) == pred_soft_binary)

print(f"New vs Ensemble agreement: {agreement_new_ensemble*100:.2f}%")
print(f"New vs Soft Ensemble agreement: {agreement_new_soft*100:.2f}%")
print(f"Ensemble vs Soft Ensemble agreement: {agreement_ensemble_soft*100:.2f}%")

# Intersection over Union (IoU)
intersection = np.sum((pred_new > 0) & (pred_ensemble > 0))
union = np.sum((pred_new > 0) | (pred_ensemble > 0))
iou_ensemble = intersection / union if union > 0 else 0
print(f"\nIoU (New vs Ensemble): {iou_ensemble:.4f}")

intersection_soft = np.sum((pred_new > 0) & (pred_soft_binary > 0))
union_soft = np.sum((pred_new > 0) | (pred_soft_binary > 0))
iou_soft = intersection_soft / union_soft if union_soft > 0 else 0
print(f"IoU (New vs Soft Ensemble): {iou_soft:.4f}")

# Dice coefficient
dice_ensemble = 2 * intersection / (np.sum(pred_new > 0) + np.sum(pred_ensemble > 0)) if np.sum(pred_new > 0) + np.sum(pred_ensemble > 0) > 0 else 0
print(f"Dice (New vs Ensemble): {dice_ensemble:.4f}")

print("\n5. CONSISTENCY ASSESSMENT")
print("-" * 70)
if agreement_new_ensemble > 0.95:
    print("✅ EXCELLENT CONSISTENCY")
    print("   All three predictions agree on the surface location")
elif agreement_new_ensemble > 0.85:
    print("✅ GOOD CONSISTENCY")
    print("   Predictions are highly correlated")
elif agreement_new_ensemble > 0.75:
    print("⚠️  MODERATE CONSISTENCY")
    print("   Some variation in predictions")
else:
    print("❌ LOW CONSISTENCY")
    print("   Predictions diverge significantly")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"New model (Jan 23):      {fg_new*100:.2f}% foreground, {nc_new} components")
print(f"Ensemble (Jan 13):       {fg_ensemble*100:.2f}% foreground")
print(f"Average agreement:       {np.mean([agreement_new_ensemble, agreement_new_soft])*100:.2f}%")
print(f"Average IoU:             {np.mean([iou_ensemble, iou_soft]):.4f}")
print("=" * 70)
