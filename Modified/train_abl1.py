"""
Training script for Ablation 1: IntraBand BiMamba + Uniform Decoder

Saves checkpoints to: ./checkpoints_abl1/
Logs to: train_abl1.log
"""

import sys
import os

# Set checkpoint directory for this ablation
CHECKPOINT_DIR = 'checkpoints_abl1'
LOG_FILE = 'train_abl1.log'
MODEL_TYPE = 'BS_BiMamba_Abl1'

# Import and modify the training config
sys.path.insert(0, os.path.dirname(__file__))
from train import Trainer, TrainingConfig
from bs_bimamba_abl1 import BS_BiMamba_Abl1

# Override config for Ablation 1
TrainingConfig.model_type = MODEL_TYPE
TrainingConfig.checkpoint_dir = CHECKPOINT_DIR
TrainingConfig.log_file = LOG_FILE

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if __name__ == '__main__':
    print("=" * 70)
    print("TRAINING ABLATION 1: IntraBand BiMamba + Uniform Decoder")
    print("=" * 70)
    print(f"Model: {MODEL_TYPE}")
    print(f"Checkpoints: ./{CHECKPOINT_DIR}/")
    print(f"Logs: {LOG_FILE}")
    print(f"Expected PESQ: 3.0-3.1")
    print("=" * 70)

    # Create trainer
    trainer = Trainer()

    # Replace model with Ablation 1
    trainer.model = BS_BiMamba_Abl1(
        num_channel=128,
        num_layers=4,
        num_bands=30,
        d_state=16,
        chunk_size=32
    ).cuda()

    # Log parameter count
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"\nModel parameters: {total_params/1e6:.2f}M")
    print(f"Expected: ~3.96M\n")

    # Train
    trainer.train()
