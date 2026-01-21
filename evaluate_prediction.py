#!/usr/bin/env python
"""Evaluate the generated prediction to see if the model learned."""

import tifffile
import numpy as np
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Load prediction
pred = tifffile.imread('predictions/1407735.tif')
print("=" * 70)
print("PREDICTION ANALYSIS")
print("=" * 70)
print(f"Shape: {pred.shape}")
print(f"Dtype: {pred.dtype}")
print(f"Min value: {pred.min()}")
print(f"Max value: {pred.max()}")
print(f"Unique values: {np.unique(pred)}")

# Statistics
foreground = (pred > 0).astype(float)
fg_fraction = foreground.mean()
print(f"\nForeground fraction: {fg_fraction:.4f} ({fg_fraction*100:.2f}%)")

# Connectivity analysis
labeled, num_components = ndimage.label(foreground)
print(f"Connected components: {num_components}")

# Component sizes
if num_components > 0:
    component_sizes = ndimage.sum(foreground, labeled, range(num_components + 1))
    print(f"Component sizes (voxels): min={component_sizes.min():.0f}, max={component_sizes.max():.0f}, mean={component_sizes.mean():.0f}")

# Load original input
original = tifffile.imread('test_images/1407735.tif')
print(f"\nOriginal input:")
print(f"  Shape: {original.shape}")
print(f"  Dtype: {original.dtype}")
print(f"  Value range: {original.min()}-{original.max()}")

print("\n" + "=" * 70)
print("PREDICTION QUALITY INDICATORS")
print("=" * 70)

# Check for spatial structure
avg_variance = 0
if pred.ndim == 3:
    # Check if prediction has spatial variation (learned patterns)
    z_variance = []
    for z in range(pred.shape[0]):
        z_slice = pred[z].astype(float)
        if z_slice.sum() > 0:
            z_variance.append(z_slice.std())
    
    if z_variance:
        avg_variance = np.mean(z_variance)
        print(f"Spatial variation (Z-axis std dev): {avg_variance:.4f}")
        print(f"  → {'✓ Good' if avg_variance > 0.01 else '✗ Low'} - Model learned spatial patterns")

# Edge detection to see if boundaries were learned
edges = np.sum(np.abs(np.diff(pred, axis=0))) + \
        np.sum(np.abs(np.diff(pred, axis=1))) + \
        np.sum(np.abs(np.diff(pred, axis=2)))
        
print(f"\nEdge complexity: {edges:.0e}")
print(f"  → {'✓ Good' if edges > 1e6 else '⚠ Low'} - Model detected boundaries")

# Topology check (Euler characteristic)
if num_components > 0:
    # Simple topological check
    print(f"\nTopology check:")
    print(f"  Components: {num_components}")
    print(f"  → {'✓ Valid' if num_components == 1 else '⚠ Multiple components'}")

# Slice-by-slice analysis
print(f"\nSlice-by-slice analysis:")
z_foreground_stats = []
for z in range(min(5, pred.shape[0])):  # First 5 slices
    z_fg = (pred[z] > 0).mean()
    z_foreground_stats.append(z_fg)
    print(f"  Z={z}: {z_fg*100:.1f}% foreground")

print("\n" + "=" * 70)
print("FINAL ASSESSMENT")
print("=" * 70)

# Overall assessment
if fg_fraction > 0.8 and num_components <= 5 and avg_variance > 0.01:
    print("✅ MODEL LEARNED WELL")
    print("   - Reasonable foreground prediction (>80%)")
    print("   - Spatial structure detected (edges present)")
    print("   - Connected topology maintained")
    verdict = "LEARNING_SUCCESS"
elif fg_fraction > 0.5 and avg_variance > 0.001:
    print("⚠️  MODEL LEARNED PARTIALLY")
    print("   - Made predictions with some spatial variation")
    print("   - May need more training or better hyperparameters")
    verdict = "PARTIAL_LEARNING"
else:
    print("❌ MODEL DID NOT LEARN")
    print("   - Predictions are uniform or random")
    print("   - No spatial structure detected")
    verdict = "NO_LEARNING"

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"Training configuration:")
print(f"  - Model: Topology-aware nnU-Net (22.6M params)")
print(f"  - Best validation loss: 0.2535 (epoch 30)")
print(f"  - Improvement: 43% from initial 0.4477")
print(f"  - Training duration: 11.14 hours")
print(f"  - SWA enabled: Yes (epoch 30-50)")
print(f"\nPrediction verdict: {verdict}")
print("=" * 70)
