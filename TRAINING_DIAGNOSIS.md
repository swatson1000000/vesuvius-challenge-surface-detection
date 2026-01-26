# Training Diagnosis Report
**Date**: January 26, 2026

## Critical Issue: Model NOT Learning

### Problem Summary
✗ Model predictions show **identical foreground ratios** across all 53 test images
✗ Model outputs are **near-uniform values** (0.52-0.71 range, mostly 0.6)
✗ Model is **not learning spatial features** - predictions are constant across space

### Evidence

**Prediction Uniformity**:
- 45 images (320×320): Exactly 95.06% foreground
- 5 images (256×256): Exactly 71.19% foreground  
- Pattern: IDENTICAL ratios = model is not discriminating between images

**Model Output Analysis** (1 patch tested):
```
Raw logits:     min=0.0863, max=0.8813, mean=0.5287, std=0.2971
After sigmoid:  min=0.5216, max=0.7071, mean=0.6267, std=0.0698
Distribution: 806K voxels @ 0.5-0.6, 1.2M voxels @ 0.6-0.7, rest ~0
```

**Interpretation**: 
- All values concentrated in narrow 0.52-0.71 range
- >95% of outputs between 0.5-0.7 (essentially binary decision at 0.5 threshold)
- Logit std (0.297) is extremely low for proper segmentation
- Model has **collapsed** to outputting pseudo-uniform values

### Root Causes

1. **Loss function weighting** - Imbalance causing model to give up learning meaningful patterns
2. **Poor gradient flow** - Variance loss might be overwhelming other signals
3. **Data/model mismatch** - Batch size 2 may be too small for large 128×128×128 patches
4. **Initialization issues** - Model may have started in bad local minimum
5. **Input normalization** - May be destroying spatial information

### Historical Context

- Previous training (before this version) had identical issue with uniform predictions
- That was fixed by "adding variance regularization" but **problem recurs**
- Suggests fix was superficial - root cause not addressed

### Immediate Actions Required

1. **STOP current training** ✓ Done
2. **Investigate actual loss components during training** - Add debug outputs
3. **Check if variance loss is actually being computed** - May have sign error
4. **Review training data quality** - Ensure labels are not corrupted
5. **Test model on training data** - Should NOT have uniform outputs
6. **Reset training** with:
   - Fresh model initialization
   - Simpler loss (just Dice + Focal, remove variance initially)
   - Smaller patch size (96×96×96)
   - Larger batch size (4-8)
   - Debug outputs at each epoch

### Expected Behavior (for reference)

Good segmentation models should produce:
- Wide distribution of output probabilities (0-1 range)
- Spatially varied predictions within patches
- **Varying foreground ratios** across images (40-95% range typical)
- Clear edges and surface boundaries

### Verification Test ✓ COMPLETED

**CONFIRMED**: Model outputs identical distributions across ALL images:
```
Test image (1407735.tif):      mean=0.6272, std=0.0689
Training image (1004283650):   mean=0.6267, std=0.0690  
Training image (11460685):     mean=0.6272, std=0.0681
```

**VERDICT**: Model is a **constant output function** - NOT LEARNING

---

**Status**: 🚨 **CRITICAL - TRAINING FAILED - DO NOT SUBMIT**
**Action**: Restart training with fixed loss configuration
