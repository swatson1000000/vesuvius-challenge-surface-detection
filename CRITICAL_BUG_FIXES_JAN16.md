# CRITICAL BUG FIXES - January 16, 2026

## Problem Summary

The training script had two interconnected critical bugs preventing proper model selection and catastrophic degradation recovery:

### Bug #1: Inverted Loss Comparison (PRIMARY)

**File**: `bin/nnunet_topo_wrapper.py` Line 391

**Original Code**:
```python
val_score = -val_losses['total']  # Negate the loss
...
if val_score < self.best_val_score:  # Save if "better"
    self.save_checkpoint('best_model.pth')
```

**Problem**: 
- Loss values are always positive (e.g., 0.3226, 0.6720)
- Negating them makes them negative (e.g., -0.3226, -0.6720)
- **More negative = worse loss value appears "better"**
- So: Loss 0.6720 → -0.6720 (more negative than -0.3226) → saved as "best"

**Evidence from Log**:
```
Epoch 11: Val Loss: 0.3226 ← ACTUAL BEST
         New best score: -0.3226

Epoch 14: Val Loss: 0.6720 ← DEGRADATION (108% worse!)
         New best score: -0.6720 ← SAVED AS BEST (wrong!)
         New best model saved! Score: -0.6720

Epoch 28: Val Loss: 0.6783 ← WORSE STILL
         New best score: -0.6783 ← SAVED AGAIN (wrong!)
```

### Bug #2: Degradation Detection Loads Wrong Model

**File**: `bin/nnunet_topo_wrapper.py` Lines 415-454

**Problem**:
- Degradation detection at Epoch 14 should trigger:
  - Loss: 0.6720 vs best: 0.3226 = **108% degradation** > 15% threshold
  - Should rollback...
- **BUT** it tries to load `best_model.pth` which was just saved as the degraded 0.6720 model
- So rollback "restores" to the bad model anyway

**Result**: Endless loop in degraded state (Epochs 14-49 all stuck at ~0.65-0.68 loss)

## Fixes Applied

### Fix #1: Remove Loss Negation

**File**: `bin/nnunet_topo_wrapper.py` Lines 139 and 385-400

```python
# Line 139: Comment fix
self.best_val_score = float('inf')  # Will be set to loss values (lower is better)

# Lines 385-400: Comparison fix
val_score = val_losses['total']  # DO NOT negate!
if 'competition_score' in val_losses:
    val_score = val_losses['competition_score']

# Lower loss (lower value) is better
if val_score < self.best_val_score:  # Now correctly saves lower losses as better
    self.best_val_score = val_score
    self.logger.info(f"✅ New best score: {val_score:.4f} (previous: {prev_score:.4f})")
    self.save_checkpoint('best_model.pth')
    best_loss_ever = current_val_loss  # Track actual best loss value
```

**Impact**: 
- Epoch 11 loss 0.3226 → saves as best ✅
- Epoch 14 loss 0.6720 → does NOT save (0.6720 > 0.3226) ✅
- Only truly better models are saved

### Fix #2: Degradation Detection Now Triggers

With Fix #1 applied:
- When degradation happens (Epoch 14: loss jumps to 0.6720)
- Degradation check: `0.6720 > 0.3226` = true → **would rollback**
- Loading `best_model.pth` which still contains the Epoch 11 model (0.3226) ✅
- Training resumes from actually good state

## Training Timeline - BEFORE and AFTER

### BEFORE (Current Run)
```
Epoch 11: Loss 0.3226 ✅ Excellent    [Saved as "best": -0.3226]
Epoch 13: Loss 0.3227 ✅ Still good
Epoch 14: Loss 0.6720 ❌ DEGRADATION! [Saved as "best": -0.6720]
Epoch 15: Loss 0.6618 ❌ Stuck
...
Epoch 45: Loss 0.6788 ❌ Stuck
Epoch 49: Loss 0.6384 ❌ Never escapes
```

### AFTER (Fixed)
```
Epoch 11: Loss 0.3226 ✅ Excellent    [Saved as "best"]
Epoch 13: Loss 0.3227 ✅ Still good
Epoch 14: Loss 0.6720 ❌ DEGRADATION! [NOT saved - stays 0.3226]
          🚨 DEGRADATION DETECTED (108% > 15%)
          ✓ Loading best model from Epoch 11
          ✓ Resuming training from good state
Epoch 15: Loss resumes from 0.3226 base ✅
```

## Why This Happened

1. **Negation Logic Error**: Whoever wrote the loss comparison forgot that negative numbers are flipped
   - -0.6720 < -0.3226 is TRUE (more negative = smaller)
   - But loss 0.6720 is WORSE than 0.3226!

2. **Degradation Detection Was Cosmetic**: 
   - The code looked correct in theory
   - But relied on `best_model.pth` containing the actual best model
   - Which it didn't due to Bug #1
   - So rollback "fixed" nothing

## How to Verify Fix Works

1. **Restart training from scratch**:
```bash
rm -rf bin/checkpoints/fold_0/
python bin/train.py --config config.yaml --fold 0
```

2. **Monitor logs for correct saves**:
```bash
tail -f log/train_*.log | grep -E "(New best|DEGRADATION)"
```

3. **Expected behavior**:
   - Early epochs save increasingly better models (loss decreases)
   - If degradation occurs, it will trigger rollback
   - Loss never gets stuck in bad state

4. **Check checkpoint file sizes** (rough indicator):
```bash
ls -lh bin/checkpoints/fold_0/best_model.pth
ls -lh bin/checkpoints/fold_0/checkpoint_epoch_*.pth
# All should be ~258MB (same architecture)
```

## Files Modified

- `bin/nnunet_topo_wrapper.py`: Lines 139, 385-400
  - Removed negation of loss values
  - Fixed comparison logic
  - Updated comments for clarity

## Status

✅ **Critical bugs identified and fixed**
✅ **Ready for retraining**
⚠️ **Previous run's "best" checkpoint is incorrect** - delete and retrain

---

**Timeline**:
- **Epoch 11**: Achieved good loss of 0.3226
- **Bug manifested Epoch 14**: Loss jumped to 0.6720 but saved as "best"
- **Degradation detection never worked**: Because best_model.pth contained bad model
- **Stuck in bad state**: Epochs 14-49 trapped at ~0.65-0.68 loss
- **Fixed**: January 16, 2026, 16:00 UTC
