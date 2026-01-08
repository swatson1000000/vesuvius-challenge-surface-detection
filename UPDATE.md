# Training Updates - January 7, 2026

## Problem Identified

Training was showing extremely slow learning progress:
- Training loss stagnated at ~0.33, improving only ~2% over 5 epochs
- Validation loss plateaued at ~0.32, improving only ~4% over 5 epochs
- Individual batch losses oscillating wildly (0.27 to 0.47)
- Model appeared stuck in local minimum

## Root Causes

1. **Learning rate too high** (0.005) - causing gradient overshooting
2. **No warmup period** - immediate high learning rate caused instability
3. **Gradient clipping too aggressive** (1.0) - limiting effective learning
4. **Scheduler too aggressive** - reducing LR too quickly on minor plateaus

## Changes Implemented

### 1. Learning Rate Reduction
**File:** `bin/config.yaml`
- **Before:** `learning_rate: 0.005`
- **After:** `learning_rate: 0.0005`
- **Reason:** 10x reduction prevents optimizer overshooting and reduces loss oscillation

### 2. Learning Rate Warmup Scheduler
**Files:** `bin/nnunet_topo_wrapper.py`

Added warmup functionality:
- Gradually increases LR from 0 to base LR over first 10 epochs
- Formula: `lr = base_lr * (epoch + 1) / warmup_epochs`
- Prevents early training instability
- Main scheduler only activates after warmup completes

**Implementation:**
```python
# In TopologyAwareTrainer.__init__:
self.warmup_epochs = warmup_epochs
self.base_lr = optimizer.param_groups[0]['lr']

# In train() method:
if epoch < self.warmup_epochs:
    warmup_factor = (epoch + 1) / self.warmup_epochs
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.base_lr * warmup_factor
```

### 3. Gradient Clipping Adjustment
**File:** `bin/nnunet_topo_wrapper.py`
- **Before:** `torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)`
- **After:** `torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)`
- **Reason:** Allows slightly larger gradients while preventing explosions; better for complex multi-component loss

### 4. Scheduler Parameter Tuning
**File:** `bin/config.yaml`

**Before:**
```yaml
scheduler_patience: 5  # More aggressive - reduce LR faster if stuck
scheduler_factor: 0.3  # Bigger reduction when plateau detected
```

**After:**
```yaml
scheduler_patience: 8  # Wait longer before reducing
scheduler_factor: 0.5  # Moderate reduction when plateau detected
grad_clip_norm: 2.0   # Increased from 1.0 for better stability
```

**Reason:** Less aggressive scheduler gives model more time to explore before reducing LR

### 5. Scheduler Configuration in Training Script
**File:** `bin/train.py`
- **Before:** Hardcoded `factor=0.5`
- **After:** `factor=config.get('scheduler_factor', 0.5)`
- **Reason:** Makes scheduler factor configurable via config.yaml

## Expected Improvements

With these changes, training should exhibit:

1. **Smoother convergence** - Less batch-to-batch loss oscillation
2. **Steadier progress** - More consistent epoch-to-epoch improvements
3. **Better exploration** - Model can find better minima with careful steps
4. **Faster learning** - Paradoxically, lower LR with warmup often learns faster than high unstable LR
5. **Better final performance** - More careful optimization finds better solutions

## Training Trajectory Comparison

### Before Changes (Epochs 0-5):
```
Epoch 0: Train=0.3350, Val=0.3330
Epoch 1: Train=0.3316, Val=0.3288
Epoch 2: Train=0.3296, Val=0.3276
Epoch 3: Train=0.3305, Val=0.3246
Epoch 4: Train=0.3296, Val=0.3316
Epoch 5: Train=0.3285, Val=0.3209
```
- **Total improvement:** Train -1.9%, Val -3.6%
- **Trend:** Minimal, inconsistent progress

### Expected After Changes (Epochs 0-10):
```
Epoch 0-9:  Warmup phase - gradual learning rate ramp
Epoch 10+:  Stable learning with full LR
```
- **Expected:** Consistent 5-10% improvement per warmup epoch
- **Target:** Val loss < 0.25 by epoch 20

## Files Modified

1. **bin/config.yaml**
   - Reduced learning_rate from 0.005 to 0.0005
   - Updated scheduler_patience from 5 to 8
   - Updated scheduler_factor from 0.3 to 0.5
   - Added grad_clip_norm: 2.0

2. **bin/nnunet_topo_wrapper.py**
   - Added warmup_epochs parameter to TopologyAwareTrainer.__init__
   - Added base_lr tracking
   - Implemented warmup schedule in train() method
   - Modified gradient clipping from 1.0 to 2.0
   - Updated scheduler to only activate after warmup

3. **bin/train.py**
   - Made scheduler_factor configurable from config.yaml

## Next Steps

1. **Restart training** with new configuration
2. **Monitor first 10 epochs** to verify warmup is working correctly
3. **Check epoch 15-20** for steady improvement trend
4. **Adjust if needed:**
   - If still too slow: increase LR to 0.001
   - If unstable: reduce LR to 0.0001 or extend warmup to 15 epochs
   - If converging too fast: may be overfitting, check validation metrics

## Restart Instructions

```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin
nohup python train.py --fold 0 --data_dir .. > ../log/train_balanced_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Monitor with:
```bash
tail -f ../log/train_balanced_*.log
```

---

## Update #2: Critical Fixes (January 7, 2026 - 21:20)

### Problem Re-assessment

After implementing the first round of changes, training **still showed minimal learning**:
- Epoch 0-8: Train loss stuck at ~0.328, Val loss ~0.323-0.336
- **Total improvement after 8 epochs: < 0.5%** (essentially no learning)
- Batch losses still oscillating wildly (0.26 to 0.43)
- **Warmup bug discovered**: warmup_epochs wasn't being passed to trainer, so warmup executed immediately

### Root Causes Identified

1. **Learning rate STILL too high** - 0.0005 still causing instability
2. **Warmup configuration bug** - `warmup_epochs` not passed through `model_config` to trainer
3. **Loss complexity** - 5 different loss components potentially fighting each other
4. **No gradient monitoring** - Can't tell if gradients are flowing properly

### Critical Fixes Implemented

#### 1. Further Learning Rate Reduction
**File:** `bin/config.yaml`
- **Before:** `learning_rate: 0.0005`
- **After:** `learning_rate: 0.0001` (5x reduction)
- **Reason:** Even 0.0005 was causing large oscillations. 0.0001 should allow steady convergence.

#### 2. Fixed Warmup Bug
**File:** `bin/train.py`
- **Problem:** `warmup_epochs` wasn't being passed through `model_config`
- **Fix:** Added `'warmup_epochs': config.get('warmup_epochs', 0)` to model_config dict
- **Impact:** Warmup will now actually execute over 10 epochs instead of completing immediately

**Before:**
```python
model_config = {
    'in_channels': 1,
    'out_channels': 1,
    'initial_filters': config.get('initial_filters', 32),
    'depth': config.get('depth', 5),
    'learning_rate': config.get('learning_rate', 1e-3),
    'weight_decay': config.get('weight_decay', 1e-5),
    'checkpoint_dir': f"checkpoints/fold_{args.fold}",
    **config.get('loss_weights', {})
}
```

**After:**
```python
model_config = {
    'in_channels': 1,
    'out_channels': 1,
    'initial_filters': config.get('initial_filters', 32),
    'depth': config.get('depth', 5),
    'learning_rate': config.get('learning_rate', 1e-3),
    'weight_decay': config.get('weight_decay', 1e-5),
    'warmup_epochs': config.get('warmup_epochs', 0),  # ADDED THIS LINE
    'checkpoint_dir': f"checkpoints/fold_{args.fold}",
    **config.get('loss_weights', {})
}
```

#### 3. Simplified Loss Function
**File:** `bin/config.yaml`
- **Problem:** 5 loss components (Dice, Focal, Boundary, clDice, Connectivity) may be conflicting
- **Fix:** Temporarily disabled 3 complex topology losses, focusing on core segmentation

**Before:**
```yaml
loss_weights:
  dice_weight: 0.3
  focal_weight: 0.4
  boundary_weight: 0.15
  cldice_weight: 0.10
  connectivity_weight: 0.05
```

**After:**
```yaml
loss_weights:
  dice_weight: 0.5      # Primary - overlap measure
  focal_weight: 0.5     # Primary - handles imbalance
  boundary_weight: 0.0  # Disabled temporarily
  cldice_weight: 0.0    # Disabled temporarily
  connectivity_weight: 0.0  # Disabled temporarily
```

**Rationale:**
- Get basic segmentation working first with Dice + Focal
- Add topology losses back later once model learns basic task
- Simpler loss landscape = easier optimization

#### 4. Added Gradient Norm Logging
**File:** `bin/nnunet_topo_wrapper.py`
- **Added:** Gradient norm tracking every 50 batches
- **Purpose:** Monitor gradient flow to diagnose vanishing/exploding gradients

**Implementation:**
```python
# Backward pass
loss.backward()

# Gradient clipping (more aggressive for stability)
grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

self.optimizer.step()

# Log gradient norm periodically
if batch_idx % 50 == 0:
    self.logger.info(f"Gradient norm: {grad_norm:.4f}")
```

### Expected Impact

With these critical fixes, training should now show:

1. **Proper warmup progression:**
   - Epoch 0: LR = 0.00001 (10% of base)
   - Epoch 5: LR = 0.00006 (60% of base)
   - Epoch 9: LR = 0.0001 (100% of base)

2. **Stable learning:**
   - Less oscillation in batch losses
   - Consistent epoch-to-epoch improvement
   - Target: 5-10% loss reduction per epoch during warmup

3. **Visible progress:**
   - Training loss should drop from 0.33 to < 0.25 by epoch 10
   - Validation loss should follow similar trend
   - Gradient norms should be in reasonable range (0.1 - 10.0)

4. **Diagnostic information:**
   - Warmup LR messages every epoch
   - Gradient norms every 50 batches
   - Can identify if problem is vanishing/exploding gradients vs. other issues

### Files Modified (Update #2)

1. **bin/config.yaml**
   - Learning rate: 0.0005 → 0.0001
   - Simplified loss weights (disabled 3 topology losses)

2. **bin/train.py**
   - Fixed warmup bug by adding warmup_epochs to model_config

3. **bin/nnunet_topo_wrapper.py**
   - Added gradient norm logging

### Next Steps

1. **Monitor first 10 epochs** closely for warmup progression
2. **Check gradient norms** - should be 0.5-5.0 range
3. **If still not learning:**
   - Check data loader (maybe seeing same samples repeatedly)
   - Verify model architecture (90M params is large for 128³ patches)
   - Try even lower LR (0.00005)
   - Add learning curve visualization
4. **If learning well:**
   - Gradually re-enable topology losses one at a time
   - Monitor impact of each loss component

### Comparison Summary

| Parameter | Original | Update #1 | Update #2 |
|-----------|----------|-----------|-----------|
| Learning Rate | 0.005 | 0.0005 | **0.0001** |
| Warmup Working | No | Bug (immediate) | **Fixed** |
| Loss Components | 5 | 5 | **2 (simplified)** |
| Gradient Logging | No | No | **Yes** |
| Grad Clip Norm | 1.0 | 2.0 | 2.0 |
| Scheduler Patience | 5 | 8 | 8 |
| Scheduler Factor | 0.3 | 0.5 | 0.5 |
