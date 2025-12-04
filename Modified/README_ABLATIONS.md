# BS-BiMamba: Band-Split Bidirectional Mamba Network
## Ablation Study Implementation

This directory contains three ablation variants of the BS-BiMamba architecture for parallel training and comparison.

---

## Quick Start: Run All Ablations in Parallel

```bash
# 1. First, test all models
cd Modified
python test_all_ablations.py

# 2. If tests pass, launch all three trainings in parallel
python train_abl1.py > abl1.log 2>&1 &
python train_abl2.py > abl2.log 2>&1 &
python train_abl3.py > abl3.log 2>&1 &

# 3. Monitor progress
tail -f abl1.log  # Check ablation 1
tail -f abl2.log  # Check ablation 2
tail -f abl3.log  # Check ablation 3

# 4. Watch PESQ scores in real-time
watch -n 60 "grep 'PESQ' abl*.log"
```

---

## Ablation Study Design

### Ablation 1: IntraBand BiMamba + Uniform Decoder
**File:** `bs_bimamba_abl1.py`
**Training:** `train_abl1.py`
**Checkpoints:** `./checkpoints_abl1/`

**Tests:** Does bidirectional Mamba work with band-split processing?

**Architecture:**
- BandSplit (30 psychoacoustic bands)
- 4 layers of IntraBand BiMamba (temporal modeling)
- Uniform 4x decoder (BSRNN baseline)

**Parameters:** ~3.96M
**Expected PESQ:** 3.0-3.1
**Rationale:** Bidirectional should recover ~0.3 PESQ vs unidirectional

---

### Ablation 2: Dual-Path BiMamba + Uniform Decoder
**File:** `bs_bimamba_abl2.py`
**Training:** `train_abl2.py`
**Checkpoints:** `./checkpoints_abl2/`

**Tests:** Does cross-band Mamba effectively model inter-band dependencies?

**Architecture:**
- BandSplit (30 psychoacoustic bands)
- 4 layers of [IntraBand BiMamba + CrossBand BiMamba]
- Uniform 4x decoder (BSRNN baseline)

**Parameters:** ~4.42M
**Expected PESQ:** 3.1-3.2
**Rationale:** Cross-band modeling captures harmonics (+0.05-0.1 PESQ)

**Novel:** First to use bidirectional Mamba for cross-band modeling

---

### Ablation 3: Full BS-BiMamba with Adaptive Decoder (BEST)
**File:** `bs_bimamba_abl3.py`
**Training:** `train_abl3.py`
**Checkpoints:** `./checkpoints_abl3/`

**Tests:** Does frequency-adaptive decoder improve performance while reducing parameters?

**Architecture:**
- BandSplit (30 psychoacoustic bands)
- 4 layers of [IntraBand BiMamba + CrossBand BiMamba]
- Frequency-Adaptive Decoder (2x/3x/4x expansion by frequency)

**Parameters:** ~2.82M (MOST EFFICIENT!)
**Expected PESQ:** 3.2-3.5 (BEST PERFORMANCE)
**Rationale:** Smart capacity allocation improves high-freq enhancement

**Novel:** All three components combined

---

## Core Components

### `mamba_blocks.py`
Reusable bidirectional Mamba building blocks:

1. **IntraBandBiMamba**
   - Temporal modeling within each frequency band
   - Forward + backward Mamba processing
   - ~116K params per layer

2. **CrossBandBiMamba**
   - Spectral modeling across the 30 bands
   - Captures harmonic structure
   - ~116K params per layer

3. **MaskDecoderUniform**
   - BSRNN baseline decoder (4x expansion all bands)
   - ~3.45M params

4. **MaskDecoderAdaptive**
   - Novel frequency-adaptive decoder
   - Low freq: 2x, Mid freq: 3x, High freq: 4x
   - ~1.85M params (46% reduction)

---

## File Structure

```
Modified/
├── mamba_blocks.py              # Core bidirectional Mamba components
├── bs_bimamba_abl1.py          # Ablation 1 model
├── bs_bimamba_abl2.py          # Ablation 2 model
├── bs_bimamba_abl3.py          # Ablation 3 model
├── train_abl1.py               # Training script for ablation 1
├── train_abl2.py               # Training script for ablation 2
├── train_abl3.py               # Training script for ablation 3
├── test_all_ablations.py       # Comprehensive test suite
├── README_ABLATIONS.md         # This file
├── checkpoints_abl1/           # Ablation 1 checkpoints (created on train)
├── checkpoints_abl2/           # Ablation 2 checkpoints (created on train)
└── checkpoints_abl3/           # Ablation 3 checkpoints (created on train)
```

---

## Testing Before Training

**Run comprehensive tests:**
```bash
cd Modified
python test_all_ablations.py
```

**Tests performed:**
1. ✓ Parameter count verification
2. ✓ Shape compatibility
3. ✓ Forward pass (complex input)
4. ✓ Forward pass (real/imag input)
5. ✓ Gradient flow
6. ✓ Memory usage

**Expected output:**
```
✓ ALL TESTS PASSED FOR Ablation 1
✓ ALL TESTS PASSED FOR Ablation 2
✓ ALL TESTS PASSED FOR Ablation 3
🎉 ALL MODELS READY FOR TRAINING!
```

---

## Training Details

### Parallel Training Setup

Each ablation uses separate:
- Checkpoint directory (`checkpoints_abl1/`, `checkpoints_abl2/`, `checkpoints_abl3/`)
- Log file (`abl1.log`, `abl2.log`, `abl3.log`)
- Process (can run simultaneously)

**Launch all three:**
```bash
# Background processes with separate logs
python train_abl1.py > abl1.log 2>&1 &
python train_abl2.py > abl2.log 2>&1 &
python train_abl3.py > abl3.log 2>&1 &

# Check process IDs
jobs -l

# Kill specific training if needed
kill %1  # Kill job 1 (abl1)
kill %2  # Kill job 2 (abl2)
kill %3  # Kill job 3 (abl3)
```

### Monitoring Training

**Real-time log monitoring:**
```bash
# Watch specific ablation
tail -f abl1.log

# Watch all PESQ scores
watch -n 60 "grep 'PESQ' abl*.log | tail -30"

# Check latest epoch for each
grep 'Epoch' abl1.log | tail -1
grep 'Epoch' abl2.log | tail -1
grep 'Epoch' abl3.log | tail -1
```

**Extract best PESQ for each:**
```bash
# Best PESQ from each ablation
grep 'PESQ' abl1.log | sort -k4 -n | tail -1
grep 'PESQ' abl2.log | sort -k4 -n | tail -1
grep 'PESQ' abl3.log | sort -k4 -n | tail -1
```

---

## Expected Results

### Performance Predictions

| Model | Parameters | Expected PESQ | Key Feature |
|-------|-----------|---------------|-------------|
| BSRNN (baseline) | 2.4M | 3.00 | Bidirectional LSTM |
| Current Modified | 2.14M | 2.62 | Unidirectional Mamba (FAILED) |
| **Ablation 1** | **3.96M** | **3.0-3.1** | **BiMamba temporal** |
| **Ablation 2** | **4.42M** | **3.1-3.2** | **+ Cross-band** |
| **Ablation 3** | **2.82M** | **3.2-3.5** | **+ Adaptive decoder** ✓ |

### Success Criteria

**Minimum Acceptable:**
- Ablation 1: PESQ ≥ 2.9 (close to baseline)
- Ablation 2: PESQ ≥ 3.0 (match baseline)
- Ablation 3: PESQ ≥ 3.1 (beat baseline)

**Target (Expected):**
- Ablation 1: PESQ = 3.0-3.1
- Ablation 2: PESQ = 3.1-3.2
- Ablation 3: PESQ = 3.2-3.5

**Stretch Goal:**
- Ablation 3: PESQ > 3.5 (match SEMamba)

---

## Analysis After Training

### Compare All Results

```bash
# Create comparison table
echo "Model,Params(M),Best PESQ,Epoch"
echo "BSRNN,2.4,3.00,12"
echo "Modified,2.14,2.62,20"
echo -n "Ablation1,3.96," && grep 'PESQ' abl1.log | sort -k4 -n | tail -1 | awk '{print $4","$2}'
echo -n "Ablation2,4.42," && grep 'PESQ' abl2.log | sort -k4 -n | tail -1 | awk '{print $4","$2}'
echo -n "Ablation3,2.82," && grep 'PESQ' abl3.log | sort -k4 -n | tail -1 | awk '{print $4","$2}'
```

### Convergence Analysis

```bash
# Extract PESQ progression for plotting
grep 'PESQ' abl1.log | awk '{print $2","$4}' > abl1_pesq.csv
grep 'PESQ' abl2.log | awk '{print $2","$4}' > abl2_pesq.csv
grep 'PESQ' abl3.log | awk '{print $2","$4}' > abl3_pesq.csv
```

---

## Troubleshooting

### Common Issues

**1. Import Error: "No module named 'real_mamba_optimized'"**
- Solution: Ensure `real_mamba_optimized.py` exists in Modified/ directory
- This contains the MambaBlock implementation

**2. CUDA Out of Memory**
- Reduce batch size in TrainingConfig
- Try gradient checkpointing
- Run ablations sequentially instead of parallel

**3. Tests Fail - NaN in Output**
- Check input normalization
- Verify learning rate not too high
- Test with smaller batch size first

**4. Parameter Count Mismatch**
- Run `python bs_bimamba_abl1.py` (or abl2, abl3) directly
- Check parameter breakdown
- Verify all components loaded correctly

### Getting Help

**Check model structure:**
```python
from bs_bimamba_abl3 import BS_BiMamba_Abl3
model = BS_BiMamba_Abl3()
print(model)  # View architecture
```

**Verify shapes:**
```python
import torch
x = torch.randn(2, 257, 100, dtype=torch.complex64)
out = model(x)
print(f"Input: {x.shape}, Output: {out.shape}")
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{bs_bimamba_2025,
  title={Band-Split Bidirectional Mamba for Speech Enhancement},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

**References:**
- BSRNN: Band-Split RNN (Interspeech 2023)
- SEMamba: Mamba for Speech Enhancement (IEEE SLT 2024)
- Mamba-SEUNet: Mamba UNet (Jan 2025)
- Dual-path Mamba: Speech Separation (2024)

---

## Contact

For questions or issues, please:
1. Check test_all_ablations.py output
2. Review logs (abl1.log, abl2.log, abl3.log)
3. Open an issue with error details

---

**Good luck with training! 🚀**

Expected timeline: ~12-20 epochs to converge
Expected time per epoch: ~30-60 minutes (depends on hardware)
Total training time: ~6-20 hours per model
