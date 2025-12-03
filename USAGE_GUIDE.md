# MBS-Net Training Guide - Complete Instructions

## Pull Latest Code

```bash
cd /home/user/BSRNN
git pull origin claude/setup-baseline-repo-01PeTEaEXa5rzSXg5q9BjfMj
```

---

## Quick Start: Training from Scratch

**For clean log files (RECOMMENDED):**

```bash
cd Modified
python train.py
```

This will train with **reduced verbosity** (only shows metrics every 500 steps, no progress bars in logs).

---

## Configuration Options

Edit `Modified/train.py` lines 20-50:

### 1. Reduce Log Verbosity (Fix the lengthy progress bar issue)

```python
class Config:
    # Progress bar settings
    disable_progress_bar = True  # ← Set to True for clean logs
```

**Before (verbose):**
```
  0%|          | 0/1928 [00:00<?, ?it/s]
  0%|          | 1/1928 [00:18<9:44:20, 18.19s/it]
  0%|          | 2/1928 [00:20<7:09:51, 13.39s/it]
  ... (1928 lines!)
```

**After (clean):**
```
Epoch 0, Step 500, loss: 0.4724, disc_loss: 0.0326, PESQ: 1.7356
Epoch 0, Step 1000, loss: 0.4620, disc_loss: 0.0213, PESQ: 1.8236
... (4 lines per epoch)
```

### 2. Resume Training from Checkpoint

```python
class Config:
    # Resume training from checkpoint
    resume = True  # ← Set to True to resume
    resume_path = None  # Auto-finds latest, or set specific path
```

**Auto-resume from latest:**
```bash
python train.py  # Automatically finds checkpoint_latest.pt
```

**Resume from specific epoch:**
```python
resume_path = '/path/to/saved_model_mbsnet/checkpoint_epoch_5.pt'
```

### 3. Other Important Settings

```python
class Config:
    init_lr = 1e-3  # Learning rate (try 5e-4 if unstable)
    batch_size = 6  # Reduce if OOM
    epochs = 120
    log_interval = 500  # Log every N steps
```

---

## How Resume Works

### Checkpoint Files Created

After each epoch, these files are saved:

```
saved_model_mbsnet/
├── checkpoint_latest.pt          ← Always the most recent
├── checkpoint_epoch_0.pt         ← Full state from epoch 0
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_2.pt
├── ...
├── gene_epoch_0_0.388            ← Old-style (model only)
└── disc_epoch_0                  ← Old-style (discriminator only)
```

### What's Saved in Checkpoints

**Full checkpoints** (`checkpoint_*.pt`) contain:
- Model weights
- Discriminator weights
- Optimizer states (Adam momentum, etc.)
- Scheduler states (learning rate schedule)
- Epoch number
- Loss value
- Training config

**Old-style checkpoints** (for compatibility):
- Just model/discriminator weights
- Cannot resume training state

### Resume Example Workflow

**Scenario**: Training crashes at epoch 5 due to power failure

```bash
# Check what checkpoints exist
ls saved_model_mbsnet/checkpoint_*.pt

# Output:
# checkpoint_epoch_0.pt
# checkpoint_epoch_1.pt
# checkpoint_epoch_2.pt
# checkpoint_epoch_3.pt
# checkpoint_epoch_4.pt
# checkpoint_latest.pt  ← This is epoch 4

# Resume training
python train.py  # With resume=True

# Output:
# INFO:root:Loading checkpoint: .../checkpoint_latest.pt
# INFO:root:✓ Resumed from epoch 4, starting epoch 5
# INFO:root:   Previous loss: 0.3867
```

Training continues from epoch 5 with optimizer state intact!

---

## NaN Detection & Debugging

### What Happens if NaN Occurs

**Old behavior (before fix):**
- NaN silently propagates
- Wasted hours of training
- No diagnostic info

**New behavior:**
```
⚠️  NaN/Inf detected in est_spec!
   NaN: True, Inf: False
   Min: -1.234, Max: 5.678
   Shape: torch.Size([6, 257, 251])
⚠️  Training stopped due to NaN at Epoch 2, Step 1547
   Saving emergency checkpoint before exit...
✓ Checkpoint saved: .../saved_model_mbsnet/checkpoint_epoch_2.pt
```

**Benefits:**
1. Immediate failure with diagnostic info
2. Emergency checkpoint saved (can analyze model state)
3. Knows EXACTLY where NaN occurred
4. Can resume from last good epoch

### NaN Detection Points

Code checks for NaN at:
1. After model forward: `est_spec`
2. After computing losses: `loss_mag`, `loss_ri`, `phase_loss`
3. Before backward: `total_loss`

If NaN detected → logs details → saves checkpoint → raises error

---

## Monitoring Training

### Log File Example

```bash
# Start training
python train.py > train.log 2>&1 &

# Monitor in real-time
tail -f train.log

# Or just check progress
tail -50 train.log
```

**What you should see (healthy training):**

```
INFO:root:Using MBS-Net Optimized (memory-efficient, ~2.3M params)
INFO:root:Model parameters: Total=3.95M, Trainable=3.95M
INFO:root:Epoch 0, Step 500, loss: 0.4724, disc_loss: 0.0326, PESQ: 1.7356
INFO:root:Epoch 0, Step 1000, loss: 0.4620, disc_loss: 0.0213, PESQ: 1.8236
INFO:root:Epoch 0, Step 1500, loss: 0.4583, disc_loss: 0.0169, PESQ: 1.8558
INFO:root:TEST - Generator loss: 0.3882, Discriminator loss: 0.0098, PESQ: 2.4178
INFO:root:✓ Checkpoint saved: .../checkpoint_epoch_0.pt
INFO:root:Epoch 1, Step 500, loss: 0.4505, disc_loss: 0.0069, PESQ: 1.9340
...
```

**Red flags (unhealthy):**

```
⚠️  NaN/Inf detected in phase_loss!     ← Immediate stop, diagnostic info
⚠️  Large gradient norm: 1234.5         ← Gradients exploding
TEST - PESQ: 0.0000                      ← Model producing garbage
loss: inf                                 ← Loss exploded
```

---

## Expected Performance

Based on fixes and clamping, here's what to expect:

| Metric | Best Case | Likely | Worst Case |
|--------|-----------|--------|------------|
| **Stability** | Trains to epoch 120 | Trains to epoch 120 | Stops at epoch 10-20 |
| **PESQ (test)** | 2.5-2.7 | 2.3-2.5 | 2.0-2.3 |
| **Comparison to baseline** | -10% | -15% | -25% |
| **Training time/epoch** | ~1.5 hours | ~1.5 hours | ~1.5 hours |

**Baseline BSRNN for reference:**
- PESQ: 2.8-3.0
- Params: 6.3M
- Always stable

---

## Troubleshooting

### Issue: Still getting NaN

**Solution 1: Reduce learning rate**
```python
init_lr = 5e-4  # or 1e-4
```

**Solution 2: Remove phase loss temporarily**
```python
loss_weights = [0.5, 0.5, 0.0, 1.0]  # No phase loss
```

**Solution 3: Check emergency checkpoint**
```bash
# Load the emergency checkpoint and inspect
python
>>> import torch
>>> ckpt = torch.load('saved_model_mbsnet/checkpoint_epoch_2.pt')
>>> print(ckpt['gen_loss'])  # See what loss was before crash
```

### Issue: Performance worse than expected

**Solution 1: Train longer**
- First few epochs are always bad
- Give it at least 20-30 epochs

**Solution 2: Check for clipping artifacts**
```bash
# Add this to mbs_net_optimized.py temporarily:
print(f"h range: [{h.min():.2f}, {h.max():.2f}]")
# If always hitting clamp boundaries (-100/+100), need to relax clamps
```

**Solution 3: Compare with baseline**
```bash
# Train baseline BSRNN for comparison
cd Baseline
# Edit train script to use BSRNN
python train.py
```

### Issue: Out of Memory

**Solution: Reduce batch size**
```python
batch_size = 4  # or even 2
```

### Issue: Too slow

**Solution: Reduce chunk_size in Mamba (trades speed for memory)**
```python
# In train.py where model is created:
chunk_size=16  # instead of 32
```

---

## Advanced: Analyzing Checkpoints

### Load and inspect a checkpoint:

```python
import torch

# Load checkpoint
ckpt = torch.load('saved_model_mbsnet/checkpoint_epoch_10.pt')

# Check what's inside
print("Epoch:", ckpt['epoch'])
print("Loss:", ckpt['gen_loss'])
print("Config:", ckpt['config'])

# Load just the model for inference
from mbs_net_optimized import MBS_Net
model = MBS_Net(num_channel=128, num_layers=4, num_bands=30,
                d_state=12, chunk_size=32)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Now use model for inference...
```

### Compare two checkpoints:

```python
ckpt1 = torch.load('checkpoint_epoch_5.pt')
ckpt2 = torch.load('checkpoint_epoch_10.pt')

print(f"Epoch 5 loss: {ckpt1['gen_loss']}")
print(f"Epoch 10 loss: {ckpt2['gen_loss']}")
print(f"Improvement: {ckpt1['gen_loss'] - ckpt2['gen_loss']}")
```

---

## Best Practices

1. **Always use reduced verbosity for log files:**
   ```python
   disable_progress_bar = True
   ```

2. **Keep checkpoints for at least 3-5 epochs:**
   - Allows rollback if training degrades
   - Can analyze what went wrong

3. **Monitor test PESQ every epoch:**
   - Should steadily increase (or stay stable)
   - If drops >0.5, something is wrong

4. **Don't delete `checkpoint_latest.pt`:**
   - This is your safety net for resume

5. **Check logs after first 2 epochs:**
   - Verify no NaN
   - Verify PESQ > 1.5
   - Verify losses decreasing

6. **If training is stable but performance poor:**
   - Train to at least epoch 30 before judging
   - Speech enhancement needs many epochs
   - Early epochs always look bad

---

## Summary

**To train with reduced verbosity and resume:**

1. Edit `Modified/train.py`:
   ```python
   disable_progress_bar = True
   resume = True  # if resuming
   ```

2. Run:
   ```bash
   cd Modified
   python train.py > train.log 2>&1 &
   ```

3. Monitor:
   ```bash
   tail -f train.log
   ```

4. Resume if crashed:
   ```bash
   # Just run again (auto-resumes from latest):
   python train.py
   ```

**Checkpoints are your friend!**
- Saved every epoch
- Contains full training state
- Can resume anytime
- Emergency save on NaN crash
