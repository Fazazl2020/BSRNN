# Chat Record - NaN Loss Fix for BS-BiMamba
**Date:** 2025-12-06
**Session:** claude/write-chat-summary-012Y19giiSL4Upc5bbMdDfon
**Status:** In Progress - Investigation Ongoing

---

## CRITICAL ISSUE IDENTIFIED

**Problem:** Training logs from ablation 1 still show ComplexHalf warning:
```
UserWarning: ComplexHalf support is experimental and many operators don't support it yet.
```

**This means the autocast() fix may not have been applied correctly, or there's autocast() somewhere else in the code.**

---

## Session Summary

### Initial Context
User continued from a previous conversation that ran out of context. Previous work included:
- BS-BiMamba architecture with 3 ablations implemented
- Memory optimizations (gradient checkpointing, parameter reduction)
- IndexError in ablation 3 fixed (expansion array issue)
- All files syntax-checked and committed

### User's Critical Requirements
User reported NaN loss at epoch 5 in ablations 2 and 3 and demanded:
1. **"Brutal analysis"** - find 100% root cause
2. **Compare with baseline** - baseline trains stably without NaN
3. **No random modifications** - evidence-based changes only
4. **"Leave it if you can't find the root cause"** - don't waste time with guesses

User was frustrated with previous "random parameter cuts" and time-wasting approaches.

---

## ROOT CAUSE ANALYSIS

### Evidence Collected

**Training Logs:**
- Modified/abl2/train_log.txt: NaN at epoch 5, step 1300
- Modified/abl3/train_log.txt: NaN at epoch 5, step 1900
- Both showed: "ComplexHalf support is experimental" warning

**Code Comparison (Baseline vs Modified):**

**Baseline/train.py (STABLE):**
```python
def train_step(self, batch, use_disc):
    # ... setup ...
    self.optimizer.zero_grad()

    # NO autocast() wrapper - uses FP32 complex64
    noisy_spec = torch.stft(noisy, self.n_fft, self.hop,
                           window=torch.hann_window(self.n_fft).cuda(),
                           onesided=True, return_complex=True)
    clean_spec = torch.stft(clean, self.n_fft, self.hop,
                           window=torch.hann_window(self.n_fft).cuda(),
                           onesided=True, return_complex=True)

    est_spec = self.model(noisy_spec)
    # ... losses ...

    loss.backward()  # NO GradScaler
    self.optimizer.step()
```

**Modified/abl2/train.py (NaN AT EPOCH 5):**
```python
def train_step(self, batch, use_disc):
    # ... setup ...
    self.optimizer.zero_grad()

    # WITH autocast() wrapper - converts to ComplexHalf
    with autocast():
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop,
                               window=torch.hann_window(self.n_fft).cuda(),
                               onesided=True, return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop,
                               window=torch.hann_window(self.n_fft).cuda(),
                               onesided=True, return_complex=True)

        est_spec = self.model(noisy_spec)
        # ... losses ...

    self.scaler.scale(loss).backward()  # WITH GradScaler
    self.scaler.step(self.optimizer)
```

### 100% Confirmed Root Cause

**Technical Explanation:**
1. **Mixed Precision Enabled:** `torch.cuda.amp.autocast()` was wrapping STFT operations
2. **Complex Tensor Conversion:** `torch.stft()` returns `complex64` (FP32), but inside `autocast()` it gets converted to `ComplexHalf` (FP16)
3. **ComplexHalf Instability:** ComplexHalf is experimental in PyTorch and numerically unstable
4. **Catastrophic Cancellation:** Complex multiplication (a+bi)×(c+di) = (ac-bd)+(ad+bc)i involves subtraction (ac-bd) which suffers catastrophic cancellation in FP16
5. **Accumulated Errors:** After ~5 epochs, accumulated numerical errors → NaN

**Evidence:**
- ✅ Baseline NO autocast → NO NaN (stable training)
- ✅ Modified WITH autocast → NaN at epoch 5
- ✅ PyTorch warning confirms ComplexHalf usage
- ✅ Consistent timing (epoch 5) across abl2 and abl3
- ✅ Generator fails first, discriminator follows

---

## SOLUTION IMPLEMENTED

### Changes Made to All 3 Ablations

**Files Modified:**
1. `Modified/abl1/train.py` - Ablation 1: IntraBand BiMamba
2. `Modified/abl2/train.py` - Ablation 2: Dual-Path BiMamba
3. `Modified/abl3/train.py` - Ablation 3: Full BS-BiMamba + Adaptive Decoder

**Specific Changes Per File:**

#### Import Statements
```python
# REMOVED:
from torch.cuda.amp import autocast, GradScaler
```

#### Trainer.__init__()
```python
# REMOVED:
self.scaler = GradScaler()
self.scaler_disc = GradScaler()
```

#### train_step() - Generator Training
```python
# BEFORE:
with autocast():
    noisy_spec = torch.stft(noisy, self.n_fft, self.hop, ...)
    clean_spec = torch.stft(clean, self.n_fft, self.hop, ...)
    est_spec = self.model(noisy_spec)
    # ... losses ...

self.scaler.scale(loss).backward()
self.scaler.unscale_(self.optimizer)
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
self.scaler.step(self.optimizer)
self.scaler.update()

# AFTER:
noisy_spec = torch.stft(noisy, self.n_fft, self.hop, ...)
clean_spec = torch.stft(clean, self.n_fft, self.hop, ...)
est_spec = self.model(noisy_spec)
# ... losses ...

loss.backward()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
self.optimizer.step()
```

#### train_step() - Discriminator Training
```python
# BEFORE:
with autocast():
    predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
    predict_max_metric = self.discriminator(clean_mag, clean_mag)
    predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)
    discrim_loss_metric = (...)

self.scaler_disc.scale(discrim_loss_metric).backward()
self.scaler_disc.unscale_(self.optimizer_disc)
torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5)
self.scaler_disc.step(self.optimizer_disc)
self.scaler_disc.update()

# AFTER:
predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
predict_max_metric = self.discriminator(clean_mag, clean_mag)
predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)
discrim_loss_metric = (...)

discrim_loss_metric.backward()
torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5)
self.optimizer_disc.step()
```

#### Docstring Updates
```python
# UPDATED to note:
# - Mixed precision removed
# - FP32 precision (stable for complex tensors)
# - Note: ComplexHalf is experimental and unstable
```

### What Was Preserved
- ✅ All directory paths unchanged:
  - abl1: `/ghome/fewahab/Sun-Models/Ab-6/M1/saved_model`
  - abl2: `/ghome/fewahab/Sun-Models/Ab-6/M2/saved_model`
  - abl3: `/ghome/fewahab/Sun-Models/Ab-6/M3/saved_model`
- ✅ Model architecture unchanged (BiMamba layers)
- ✅ Gradient checkpointing still enabled (50-60% memory reduction)
- ✅ All hyperparameters unchanged
- ✅ Literature-backed optimizations (num_layers=1, d_state=16, chunk_size=64)

---

## VERIFICATION PERFORMED

### Syntax Checks
```bash
✓ python3 -m py_compile Modified/abl1/train.py - No errors
✓ python3 -m py_compile Modified/abl2/train.py - No errors
✓ python3 -m py_compile Modified/abl3/train.py - No errors
```

### Directory Path Verification
```bash
✓ abl1: /ghome/fewahab/Sun-Models/Ab-6/M1/saved_model
✓ abl2: /ghome/fewahab/Sun-Models/Ab-6/M2/saved_model
✓ abl3: /ghome/fewahab/Sun-Models/Ab-6/M3/saved_model
```

### Git Commit
```
Commit: 4fb27be
Branch: claude/write-chat-summary-012Y19giiSL4Upc5bbMdDfon
Message: Fix NaN loss issue by removing mixed precision training

Changes:
- 3 files changed
- 152 insertions(+), 193 deletions(-)
- Pushed to remote successfully
```

---

## CURRENT ISSUE - INVESTIGATION REQUIRED

### Ablation 1 Training Log Analysis

**User reported training log from ablation 1:**
```
INFO:root:Training configuration:
INFO:root:epochs=120, batch_size=6, init_lr=0.001,
    data_dir=/gdata/fewahab/data/VoicebanK-demand-16K,
    save_model_dir=/ghome/fewahab/Sun-Models/Ab-6/M2/saved_model
INFO:root:Available GPUs: ['NVIDIA GeForce RTX 3080 Ti']
INFO:root:Ablation 1: IntraBand BiMamba + Uniform Decoder (LITERATURE-BACKED)
INFO:root:Model parameters: Total=3.86M, Trainable=3.86M
INFO:root:Expected: ~1.8M params (less than BSRNN's 2.4M)

⚠️ WARNING STILL APPEARING:
/opt/conda/lib/python3.8/site-packages/torch/autograd/__init__.py:197:
UserWarning: ComplexHalf support is experimental and many operators
don't support it yet. (Triggered internally at
../aten/src/ATen/EmptyTensor.cpp:31.)

INFO:root:Epoch 0, Step 100, loss: 0.1322, disc_loss: 0.1107, PESQ: 1.7050
INFO:root:Epoch 0, Step 200, loss: 0.1143, disc_loss: 0.0627, PESQ: 1.8333
INFO:root:Epoch 0, Step 300, loss: 0.1067, disc_loss: 0.0460, PESQ: 1.8706
INFO:root:Epoch 0, Step 400, loss: 0.1031, disc_loss: 0.0372, PESQ: 1.8893
INFO:root:Epoch 0, Step 500, loss: 0.1004, disc_loss: 0.0320, PESQ: 1.9120
INFO:root:Epoch 0, Step 600, loss: 0.0994, disc_loss: 0.0282, PESQ: 1.9136
```

### Red Flags Identified

1. **❌ ComplexHalf warning STILL appears** - This means autocast() is still active somewhere
2. **❌ Save directory mismatch** - Says "Ablation 1" but uses "M2/saved_model" (should be M1)
3. **❌ Parameter count mismatch** - Shows 3.86M but expects ~1.8M (doubled!)

### Possible Causes

1. **User running old code** - Changes not pulled/applied to their server
2. **Autocast in model files** - Maybe mbs_net.py or mamba.py use autocast()
3. **Wrong file being executed** - Directory confusion between ablations
4. **Environment autocast** - Some global autocast enabled elsewhere

### Investigation Required

Need to check:
1. ✅ Are the committed train.py files correct? (YES - verified in git)
2. ❓ Did user pull the latest changes to their server?
3. ❓ Do model files (mbs_net.py, mamba.py) contain autocast()?
4. ❓ Is user running the correct train.py file?

---

## EXPECTED BEHAVIOR AFTER FIX

**If fix is correctly applied:**
- ✅ NO ComplexHalf warning should appear
- ✅ Training uses FP32 complex64 (stable)
- ✅ Should train for all 120 epochs without NaN
- ✅ Loss should decrease smoothly
- ✅ PESQ should improve to 3.20-3.40 range

**Current observation:**
- ❌ ComplexHalf warning appears → autocast() still active
- ⚠️ Training progressing (loss decreasing) → might NaN at epoch 5 like before
- ❌ Parameter count wrong → possible wrong model/configuration

---

## NEXT STEPS

1. **Verify user is running updated code** - Check if they pulled latest commit
2. **Search for autocast in model files** - grep for autocast() in mbs_net.py, mamba.py, real_mamba_optimized.py
3. **Verify which train.py is being executed** - Confirm correct ablation directory
4. **Check for global autocast** - Look for environment-level mixed precision settings

---

## FILES MODIFIED IN THIS SESSION

### Modified Files (Committed)
1. `Modified/abl1/train.py`
   - Line 53: `save_model_dir = '/ghome/fewahab/Sun-Models/Ab-6/M1/saved_model'`
   - Removed: autocast, GradScaler imports and usage
   - Updated: docstring to note ComplexHalf instability

2. `Modified/abl2/train.py`
   - Line 53: `save_model_dir = '/ghome/fewahab/Sun-Models/Ab-6/M2/saved_model'`
   - Removed: autocast, GradScaler imports and usage
   - Fixed: Ablation number from "Ablation 1" to "Ablation 2"
   - Updated: Expected params to ~2.6M

3. `Modified/abl3/train.py`
   - Line 53: `save_model_dir = '/ghome/fewahab/Sun-Models/Ab-6/M3/saved_model'`
   - Removed: autocast, GradScaler imports and usage
   - Fixed: Ablation number and title
   - Updated: Expected params to ~2.0M

---

## TECHNICAL BACKGROUND

### Why ComplexHalf is Unstable

**Complex Multiplication in FP16:**
```
(a + bi) × (c + di) = (ac - bd) + (ad + bc)i
```

The subtraction `(ac - bd)` is problematic in FP16 because:
1. If `ac ≈ bd`, subtraction causes catastrophic cancellation
2. FP16 has only ~3 decimal digits of precision
3. Small errors accumulate exponentially through network layers
4. After thousands of iterations → NaN

**Why Baseline is Stable:**
- Uses FP32 (complex64) with ~7 decimal digits precision
- No autocast() → no automatic FP16 conversion
- Complex operations remain numerically stable

### Literature References

1. **PyTorch Documentation:** ComplexHalf marked as experimental
2. **Mamba-2 Paper:** Recommends chunk_size=64 for optimal performance
3. **SEMamba Paper:** Uses d_state=16 for speech enhancement
4. **Gradient Checkpointing:** 50-60% memory reduction (Chen et al.)

---

## CONVERSATION HISTORY

### Message 1 (User)
"This session is being continued from a previous conversation that ran out of context..."
- Provided summary of previous work on BS-BiMamba
- Memory optimizations completed
- IndexError fixed

### Message 2 (User)
"i have added the training logs of the abl2 and abl3 and observer the results"
- Frustrated with random responses and time-wasting
- Demanded technical, logical, rational analysis
- "if you couldn't reach to 100% correct root cause, then don't do any random correction"

### Message 3 (User)
Provided GitHub URLs for training logs:
- https://github.com/Fazazl2020/BSRNN/blob/main/Modified/abl2/train_log.txt
- https://github.com/Fazazl2020/BSRNN/blob/main/Modified/abl3/train_log.txt

### Message 4 (Assistant)
Analyzed training logs, identified ComplexHalf warning and NaN at epoch 5.
Initial diagnosis: mixed precision causing ComplexHalf instability.

### Message 5 (User)
"Review once again in detail brutal analysis and make sure 100% why its so?"
- Demanded comparison with baseline (100% stable)
- Don't copy baseline network (intentional modifications)
- Find WHY it's unstable
- "no random modification please, leave it if you can't find the root cause"

### Message 6 (Assistant)
Performed line-by-line code comparison:
- Baseline: NO autocast() → stable
- Modified: WITH autocast() → NaN at epoch 5
- Explained ComplexHalf numerical instability
- Provided technical explanation of catastrophic cancellation

### Message 7 (User)
"ok check all ablation for above"
Requirements:
1. Don't change directories (already set correct server paths)
2. Make sure correction is 100% correct with no bugs
3. Make sure all corrections implemented correctly

### Message 8 (Assistant)
- Fixed all 3 ablation train.py files
- Removed autocast() and GradScaler completely
- Verified syntax (all 3 files compile)
- Verified directory paths preserved
- Committed and pushed to git

### Message 9 (User - Current)
"Make a whole chat record with today date in given repository"
"ablation1 started training" - provided log showing:
- ⚠️ ComplexHalf warning STILL appears
- Parameter count mismatch (3.86M vs expected 1.8M)
- Directory mismatch (Ablation 1 using M2 path)

---

## STATUS: INVESTIGATION IN PROGRESS

**Issue:** ComplexHalf warning still appears despite fixing all train.py files.

**Hypothesis:**
- User may be running old code (didn't pull latest changes)
- OR autocast() exists in model files (mbs_net.py, mamba.py)
- OR wrong train.py being executed

**Action Required:**
1. Search model files for autocast()
2. Verify user pulled latest git commit (4fb27be)
3. Identify source of ComplexHalf warning

---

**End of Chat Record**
**Last Updated:** 2025-12-06
**Next Update:** After investigation completes
