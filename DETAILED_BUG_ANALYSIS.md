# Training Evaluation & Bug Analysis

## Log Analysis: Epochs 0-49

### Phase 1: Initial Training (Epochs 0-13)

```
Epoch  0: Loss = 0.4469  → Saved as best (first model)
Epoch  1: Loss = 0.4445  (improved, but NOT saved due to bug*)
Epoch  2: Loss = 0.4411  (improved, but NOT saved)
Epoch  3: Loss = 0.4468  (worse, NOT saved)
Epoch  4: Loss = 0.4456  (worse, NOT saved)
Epoch  5: Loss = 0.4458  (worse, NOT saved)
Epoch  6: Loss = 0.3858  ✅ Saved as best (WRONG! Bug activates)
Epoch  7: Loss = 0.3905  
Epoch  8: Loss = 0.3878  
Epoch  9: Loss = 0.3897  
Epoch 10: Loss = 0.3235  ✅ HUGE improvement - should be saved
Epoch 11: Loss = 0.3226  ✅ ACTUAL BEST - should be saved
Epoch 12: Loss = 0.3229  (slightly worse)
Epoch 13: Loss = 0.3227  (very close to best)

*Note: Epochs 1-5 were improvements but NOT saved because:
- Epoch 1: 0.4445 → -0.4445 (less negative than -0.4469, "worse")
- So comparison "val_score < best_val_score" is (-0.4445 < -0.4469) = FALSE
```

### Phase 2: Catastrophic Degradation (Epochs 14-49)

```
Epoch 14: Loss = 0.6720  ⚠️ DEGRADATION SPIKE! (108% worse than 0.3226)
          BUT SAVED AS BEST (-0.6720 < -0.3226 in negative logic)
          
Epoch 15: Loss = 0.6618  (degradation continues)
Epoch 16: Loss = 0.6600  (stuck)
...
Epoch 27: Loss = 0.6643  (still degraded)
Epoch 28: Loss = 0.6783  (WORSE!) Saved as "new best" 
Epoch 29: Loss = 0.6487  
...
Epoch 45: Loss = 0.6788  (EVEN WORSE!) Saved as "new best"
...
Epoch 49: Loss = 0.6384  (stuck in bad state)
```

## The Bug Explained Step-by-Step

### How the Negation Bug Works

**Step 1: Calculate loss**
```python
val_losses['total'] = 0.3226  # Good model
val_losses['total'] = 0.6720  # Bad model
```

**Step 2: Apply buggy negation**
```python
val_score = -val_losses['total']
# Good model: val_score = -0.3226
# Bad model:  val_score = -0.6720
```

**Step 3: Compare with "best"** 
```python
self.best_val_score = -0.3226  # Started with good model
if val_score < self.best_val_score:  # -0.6720 < -0.3226?
    # YES! Because -0.6720 is MORE negative (smaller number)
    self.save_checkpoint('best_model.pth')  # SAVES BAD MODEL
```

### Mathematical Proof

On the number line:
```
-1.0  -0.6720  -0.3226  0  
 |      |        |      |
 ◄─────►◄─────────►   More negative = smaller value
```

- `-0.6720 < -0.3226` is TRUE (mathematically correct)
- But **loss 0.6720 is WORSE than 0.3226** (logic backwards)

## Why Degradation Detection Failed

### Degradation Logic (Lines 415-454)
```python
if best_loss_ever != float('inf'):
    degradation_fraction = (current_val_loss - best_loss_ever) / best_loss_ever
    
    if degradation_fraction > catastrophic_degradation_threshold:
        # Should rollback...
        self.load_checkpoint('best_model.pth')
```

### What Should Have Happened (Epoch 14)
```python
best_loss_ever = 0.3226  # From epoch 11
current_val_loss = 0.6720  # Epoch 14

degradation_fraction = (0.6720 - 0.3226) / 0.3226 = 1.083 = 108.3%

if 108.3% > 15%:  # TRUE!
    load_checkpoint('best_model.pth')  # Should load epoch 11 model
```

### What Actually Happened
```python
best_loss_ever = 0.3226  # Still correct
current_val_loss = 0.6720  # Correct

# BUT WAIT: At line 398, when checking to save:
if val_score < self.best_val_score:  # -0.6720 < -0.3226 = TRUE
    self.save_checkpoint('best_model.pth')  # OVERWRITES with 0.6720 model!
    best_loss_ever = current_val_loss  # Updates to 0.6720 (WRONG!)
```

**So**:
- Degradation IS detected (108% > 15%) ✓
- Rollback IS triggered ✓
- **BUT** `best_model.pth` now contains the BAD 0.6720 model ✗
- Loading it "recovers" to... the bad state ✗

## Cascade of Disasters

```
Epoch 11: Good model (0.3226) saved
Epoch 14: Bad model (0.6720) → overwrites best_model.pth
         Degradation detected → load best_model.pth
         Loads 0.6720 (the bad model)
         Continue from bad state
Epoch 15-49: Trapped in bad state (~0.65-0.68 loss)
```

## The Fix

### Before
```python
val_score = -val_losses['total']  # Negate (WRONG)
if val_score < self.best_val_score:  # Compare inverted values
    self.save_checkpoint('best_model.pth')
```

### After
```python
val_score = val_losses['total']  # DO NOT negate (CORRECT)
if val_score < self.best_val_score:  # Compare actual values
    self.save_checkpoint('best_model.pth')
```

### With Fix Applied (Epoch 14)
```python
val_score = 0.6720  # Actual loss value
self.best_val_score = 0.3226  # Best achieved

if 0.6720 < 0.3226:  # FALSE! Correctly recognized as worse
    # Skip saving
```

## Expected Training Curve - After Fix

```
Epoch  0: Loss = 0.4469  → Save as best (first)
Epoch  1: Loss = 0.4445  ✅ Save as best (improved!)
Epoch  2: Loss = 0.4411  ✅ Save as best (improved!)
Epoch  6: Loss = 0.3858  ✅ Save as best (big jump)
Epoch 10: Loss = 0.3235  ✅ Save as best
Epoch 11: Loss = 0.3226  ✅ Save as best (ACTUAL BEST)
Epoch 13: Loss = 0.3227  (skip - not better than 0.3226)
Epoch 14: Loss = 0.6720  (skip - much worse)
         🚨 DEGRADATION DETECTED (108% > 15%)
         ✓ Rollback to epoch 11 (0.3226 model)
Epoch 15: Loss = 0.3232  ✅ Resume from good state
...
(Continue improving or plateau, but never get stuck at 0.66)
```

## What to Watch For in Retrained Run

1. **Early epochs**: Loss should decrease progressively
2. **Best model saves**: Should be decreasing loss values
   - Example log: `✅ New best score: 0.4469 (previous: inf)`
   - Then: `✅ New best score: 0.4445 (previous: 0.4469)`
   - Then: `✅ New best score: 0.3226 (previous: 0.3858)`
   - NOT: `✅ New best score: 0.6720 (previous: 0.3226)` ❌

3. **Degradation recovery**: If loss jumps, should see:
   ```
   🚨 CATASTROPHIC DEGRADATION DETECTED!
      Degradation: 108.3% (threshold: 15%)
      Rolling back to best checkpoint...
   ```

4. **No more bad states**: Should never get stuck at high loss values

## Conclusion

| Aspect | Impact |
|--------|--------|
| **Bug Severity** | CRITICAL - Completely breaks model selection |
| **Affected Epochs** | All (but manifested Epoch 14+) |
| **Data Loss** | No (just wrong checkpoint saved) |
| **Training Time Wasted** | ~12 hours (Epochs 11-49) stuck in bad state |
| **Fix Complexity** | SIMPLE - Remove 1 negation operator |
| **Fix Applied** | YES - Line 387 |
| **Recovery Action** | Retrain from scratch (delete old checkpoints) |

---

**Recommendation**: Delete `bin/checkpoints/fold_0/` and retrain immediately.
The previous training is compromised and cannot be salvaged.
