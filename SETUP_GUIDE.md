# BSRNN Training Setup Guide for University Server

## ✅ MODIFICATIONS COMPLETED

### 1. **Argparse Removed** ✓
- All command-line arguments removed
- Configuration now hardcoded in `Config` class (lines 20-32)
- Can run training with simple: `python train.py`

---

## 📁 DIRECTORY CONFIGURATION

### Your Server Paths (Already Configured):

```python
# In Modified/train.py (lines 30-32)
data_dir = '/gdata/fewahab/data/VoicebanK-demand-16K'
save_model_dir = '/ghome/fewahab/Sun-Models/Ab-5/CMGAN/saved_model'
```

### Required Dataset Structure (You Already Have This):
```
/gdata/fewahab/data/VoicebanK-demand-16K/
├── train/
│   ├── clean/     # Clean training audio files
│   └── noisy/     # Noisy training audio files
└── test/
    ├── clean/     # Clean test audio files
    └── noisy/     # Noisy test audio files
```

### Model Save Location:
- Models will be saved to: `/ghome/fewahab/Sun-Models/Ab-5/CMGAN/saved_model/`
- Directory will be created automatically if it doesn't exist

---

## 💾 MODEL SAVING MECHANISM

### How Models Are Saved:
1. **Frequency**: After EVERY epoch (120 epochs total)
2. **Generator Model**: `gene_epoch_{epoch_number}_{loss_value}`
   - Example: `gene_epoch_0_0.123`, `gene_epoch_1_0.098`
3. **Discriminator Model**: `disc_epoch_{epoch_number}`
   - Example: `disc_epoch_0`, `disc_epoch_1`

### What Gets Saved:
- ✅ Generator (BSRNN model) state_dict
- ✅ Discriminator state_dict
- ✅ Both saved every epoch
- ✅ 120 checkpoints total (one per epoch)

---

## 📊 PESQ MONITORING (NEW FEATURE ADDED)

### ✨ PESQ is Now Logged During Training!

**Original Code**: PESQ was calculated but NOT printed
**Modified Code**: PESQ is now displayed in logs

### Training Logs Will Show:
```
Epoch 0, Step 500, loss: 0.4523, disc_loss: 0.0123, PESQ: 2.4567
```

### Test Logs Will Show:
```
TEST - Generator loss: 0.4234, Discriminator loss: 0.0098, PESQ: 2.5432
```

### PESQ Details:
- **Range**: -0.5 to 4.5 (standard PESQ scale)
- **Calculation**: Every training step (using batch_pesq function)
- **Display**: Every 500 steps (configurable via `log_interval`)
- **Test PESQ**: Calculated on full test set after each epoch

---

## ⚙️ TRAINING PARAMETERS (Hardcoded)

```python
epochs = 120                    # Total training epochs
batch_size = 6                  # Batch size
log_interval = 500              # Log every 500 steps
decay_epoch = 10                # LR decay every 10 epochs
init_lr = 1e-3                  # Initial learning rate: 0.001
cut_len = 32000                 # Audio length: 2 seconds @ 16kHz
loss_weights = [0.5, 0.5, 1]   # [RI loss, Mag loss, GAN loss]
```

### Model Architecture:
- **BSRNN**: num_channel=64, num_layer=5
- **Discriminator**: ndf=16

---

## 🚀 HOW TO RUN ON SERVER

### Step 1: Navigate to Modified Directory
```bash
cd /ghome/fewahab/Sun-Models/Ab-5/CMGAN/BSRNN/Modified
```

### Step 2: Verify Dataset Exists
```bash
ls /gdata/fewahab/data/VoicebanK-demand-16K/train/clean
ls /gdata/fewahab/data/VoicebanK-demand-16K/train/noisy
```

### Step 3: Run Training (Simple!)
```bash
python train.py
```

### Step 4: Monitor Output
You should see:
- GPU information
- Configuration parameters
- Training progress bars
- Logs every 500 steps with PESQ scores
- Test evaluation after each epoch

---

## 📈 EXPECTED OUTPUT

### At Start:
```
Namespace(epochs=120, batch_size=6, ...)
['NVIDIA GeForce RTX 3090']
```

### During Training:
```
INFO:root:Epoch 0, Step 500, loss: 0.4523, disc_loss: 0.0123, PESQ: 2.4567
INFO:root:Epoch 0, Step 1000, loss: 0.4234, disc_loss: 0.0156, PESQ: 2.5012
```

### After Each Epoch:
```
INFO:root:TEST - Generator loss: 0.4234, Discriminator loss: 0.0098, PESQ: 2.5432
```

---

## 🔧 IF YOU NEED TO MODIFY PATHS

Edit `Modified/train.py` lines 30-32:
```python
class Config:
    # ...
    data_dir = '/your/new/dataset/path'
    save_model_dir = '/your/new/model/save/path'
```

---

## ⚠️ IMPORTANT NOTES

1. **Discriminator Training**:
   - First 60 epochs: Generator only
   - Last 60 epochs: Generator + Discriminator (use_disc=True)

2. **PESQ Calculation**:
   - Uses parallel processing (joblib)
   - May return None for silent audio (handled gracefully)
   - Normalized to [0,1] for discriminator, denormalized for display

3. **GPU Required**:
   - Code uses `.cuda()` - GPU is mandatory
   - Check GPU availability before running

4. **Expected Training Time**:
   - Depends on dataset size and GPU
   - 120 epochs with VoiceBank-DEMAND typically takes several hours

---

## 📝 FILES MODIFIED

- ✅ `Modified/train.py` - All modifications made here
- ⚠️ `Baseline/train.py` - Unchanged (kept as reference)

---

## 🎯 SUMMARY

| Feature | Status |
|---------|--------|
| Argparse removed | ✅ Done |
| Paths configured | ✅ Set to your server |
| PESQ logging added | ✅ Now prints during training |
| Model saving verified | ✅ Every epoch, automatic |
| Ready to run | ✅ Just `python train.py` |

---

## 🆘 TROUBLESHOOTING

**If dataset not found:**
```bash
ls -la /gdata/fewahab/data/VoicebanK-demand-16K/
```

**If CUDA error:**
```bash
nvidia-smi  # Check GPU availability
```

**If permission error on save path:**
```bash
mkdir -p /ghome/fewahab/Sun-Models/Ab-5/CMGAN/saved_model
chmod 755 /ghome/fewahab/Sun-Models/Ab-5/CMGAN/saved_model
```
