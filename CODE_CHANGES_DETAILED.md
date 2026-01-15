# Code Changes: Exact Modifications

## File: bin/nnunet_topo_wrapper.py

### Change 1: Method Signature (Lines 249-269)

**ADDED PARAMETER:**
```python
catastrophic_degradation_threshold: float = 0.15,
```

This parameter (default 15%) controls when to trigger rollback.

---

### Change 2: Initialization Variables (Lines ~306-321)

**ADDED VARIABLE INITIALIZATION:**
```python
best_loss_ever = float('inf')  # Track absolute best loss across all epochs
```

Then at line ~320:
```python
# Store best hyperparameters for potential rollback
best_lr = self.base_lr
best_loss_weights = None
if hasattr(self.criterion, 'dice_weight'):
    best_loss_weights = {
        'dice_weight': self.criterion.dice_weight,
        'focal_weight': self.criterion.focal_weight,
        'boundary_weight': getattr(self.criterion, 'boundary_weight', 0.0),
        'cldice_weight': getattr(self.criterion, 'cldice_weight', 0.0),
        'connectivity_weight': getattr(self.criterion, 'connectivity_weight', 0.0),
    }
```

These variables store the best state for recovery.

---

### Change 3: Best Model Saving (Lines ~380-395)

**MODIFIED SECTION - Now Updates Best State:**
```python
if val_score > self.best_val_score:
    self.best_val_score = val_score
    self.save_checkpoint('best_model.pth')
    # Update best loss tracking
    best_loss_ever = current_val_loss
    best_lr = self.base_lr
    if hasattr(self.criterion, 'dice_weight'):
        best_loss_weights = {
            'dice_weight': self.criterion.dice_weight,
            'focal_weight': self.criterion.focal_weight,
            'boundary_weight': getattr(self.criterion, 'boundary_weight', 0.0),
            'cldice_weight': getattr(self.criterion, 'cldice_weight', 0.0),
            'connectivity_weight': getattr(self.criterion, 'connectivity_weight', 0.0),
        }
    self.logger.info(f"New best model saved! Score: {val_score:.4f}")
    patience_counter = 0
```

**KEY ADDITION:** Lines capturing `best_loss_ever`, `best_lr`, and `best_loss_weights`

---

### Change 4: Catastrophic Degradation Detection & Recovery (Lines ~400-440)

**COMPLETELY NEW SECTION - This is the Core Fix:**

```python
# ⚠️ CATASTROPHIC DEGRADATION DETECTION
# If loss increased by more than threshold from best, rollback
if best_loss_ever != float('inf'):
    degradation_fraction = (current_val_loss - best_loss_ever) / best_loss_ever
    if degradation_fraction > catastrophic_degradation_threshold:
        self.logger.error(f"🚨 CATASTROPHIC DEGRADATION DETECTED!")
        self.logger.error(f"   Best loss: {best_loss_ever:.4f}")
        self.logger.error(f"   Current loss: {current_val_loss:.4f}")
        self.logger.error(f"   Degradation: {degradation_fraction*100:.1f}% (threshold: {catastrophic_degradation_threshold*100:.1f}%)")
        self.logger.error(f"   Rolling back to best checkpoint and restoring hyperparameters...")
        
        # Rollback model
        self.load_checkpoint('best_model.pth')
        
        # Restore best learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = best_lr
        self.base_lr = best_lr
        self.logger.info(f"   ✓ Restored LR: {best_lr:.6f}")
        
        # Restore best loss weights
        if best_loss_weights is not None and hasattr(self.criterion, 'dice_weight'):
            self.criterion.dice_weight = best_loss_weights['dice_weight']
            self.criterion.focal_weight = best_loss_weights['focal_weight']
            if hasattr(self.criterion, 'boundary_weight'):
                self.criterion.boundary_weight = best_loss_weights['boundary_weight']
            if hasattr(self.criterion, 'cldice_weight'):
                self.criterion.cldice_weight = best_loss_weights['cldice_weight']
            if hasattr(self.criterion, 'connectivity_weight'):
                self.criterion.connectivity_weight = best_loss_weights['connectivity_weight']
            self.logger.info(f"   ✓ Restored loss weights (Dice={best_loss_weights['dice_weight']:.2f}, "
                           f"Focal={best_loss_weights['focal_weight']:.2f})")
        
        # Reset intervention counter - start fresh from good state
        intervention_count = 0
        self.logger.info(f"   ✓ Reset intervention counter")
        self.logger.info(f"   ✓ Continuing training from best state...")
```

This is the main new logic that:
1. Detects degradation
2. Loads best checkpoint
3. Restores hyperparameters
4. Resets interventions

---

## Total Lines Changed/Added

- **Method signature:** 1 new parameter
- **Initialization:** ~15 new lines
- **Best model tracking:** ~9 modified lines
- **Detection & recovery:** ~40 new lines
- **Total:** ~65 lines added/modified

## Code Impact

**Before:**
- 609 lines in file
- No degradation detection
- No recovery mechanism

**After:**
- 673 lines in file (+64 lines)
- Automatic degradation detection
- Automatic recovery to best state

## Backward Compatibility

✅ **Fully backward compatible:**
- New parameter has default value (0.15)
- Existing code works without changes
- Only activates on catastrophic degradation
- No changes to existing APIs

## Dependencies

No new dependencies added:
- Uses existing checkpoint loading
- Uses existing optimizer/scheduler
- Uses existing logger
- All features use standard PyTorch

---

## Testing the Changes

### Minimal Test
```python
# No changes needed - runs automatically
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    scheduler=scheduler
)
```

### With Custom Threshold
```python
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    scheduler=scheduler,
    catastrophic_degradation_threshold=0.20  # 20% instead of 15%
)
```

---

## Log Output Changes

**New error log when degradation detected:**
```
🚨 CATASTROPHIC DEGRADATION DETECTED!
   Best loss: 0.3226
   Current loss: 0.5000
   Degradation: 54.8% (threshold: 15.0%)
   Rolling back to best checkpoint and restoring hyperparameters...
   ✓ Restored LR: 0.001000
   ✓ Restored loss weights (Dice=0.40, Focal=0.20)
   ✓ Reset intervention counter
   ✓ Continuing training from best state...
```

---

## Performance Impact

**Minimal computational overhead:**
- One division operation per epoch: `(current_val_loss - best_loss_ever) / best_loss_ever`
- Simple comparison: `if degradation_fraction > threshold`
- Only checkpoint loading if triggered (rare event)

**Estimated overhead:** < 0.1% per epoch

---

## Summary of Changes

| Aspect | Details |
|--------|---------|
| **Lines Added** | ~64 |
| **Complexity** | Simple (mostly tracking and comparison) |
| **Overhead** | Negligible |
| **Backward Compatible** | ✅ Yes |
| **New Dependencies** | ❌ None |
| **Breaking Changes** | ❌ None |
| **Default Behavior** | ✅ Automatic (15% threshold) |
| **Customizable** | ✅ Yes (threshold parameter) |
