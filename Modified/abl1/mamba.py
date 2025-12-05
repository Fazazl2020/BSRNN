"""
Mamba Building Blocks for Ablation 1: IntraBand BiMamba

Based on:
- SEMamba (IEEE SLT 2024): Bidirectional Mamba for speech enhancement
- Mamba-SEUNet (Jan 2025): Bidirectional SSM implementation

This file contains core Mamba components needed for Ablation 1.
"""

import torch
import torch.nn as nn
import sys
import os

# Import base MambaBlock from real_mamba_optimized
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from real_mamba_optimized import MambaBlock


class IntraBandBiMamba(nn.Module):
    """
    Bidirectional Mamba for temporal processing within each frequency band.
    Literature-backed parameters:
    - d_state=16 (SEMamba standard)
    - chunk_size=64 (Mamba-2 recommendation)
    - use_checkpoint=True for 50-60% memory savings
    """
    def __init__(self, channels=128, d_state=16, d_conv=4, chunk_size=64, use_checkpoint=True):
        super().__init__()
        self.channels = channels

        self.norm = nn.LayerNorm(channels)

        self.mamba_fwd = MambaBlock(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=1,
            chunk_size=chunk_size,
            use_checkpoint=use_checkpoint
        )

        self.mamba_bwd = MambaBlock(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=1,
            chunk_size=chunk_size,
            use_checkpoint=use_checkpoint
        )

        self.combine = nn.Linear(2 * channels, channels)

    def forward(self, x):
        """
        Args:
            x: [B, N, T, K] - Band-split features
        Returns:
            out: [B, N, T, K] - Bidirectionally processed features
        """
        B, N, T, K = x.shape

        x = x.permute(0, 3, 2, 1).contiguous()  # [B, K, T, N]
        x = x.reshape(B * K, T, N)

        x_norm = self.norm(x)

        out_fwd = self.mamba_fwd(x_norm)

        x_rev = torch.flip(x_norm, dims=[1])
        out_bwd = self.mamba_bwd(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])

        out_concat = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.combine(out_concat)
        out = out + x

        out = out.reshape(B, K, T, N)
        out = out.permute(0, 3, 2, 1)

        return out


class MaskDecoderUniform(nn.Module):
    """
    Uniform mask decoder with 4x expansion for all bands (BSRNN baseline).
    """
    def __init__(self, channels=128):
        super().__init__()
        self.channels = channels

        self.band = torch.Tensor([
            2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            16, 16, 16, 16, 16, 16, 16, 17
        ])

        for i in range(len(self.band)):
            setattr(self, f'norm{i+1}', nn.GroupNorm(1, channels))
            setattr(self, f'fc1{i+1}', nn.Linear(channels, 4*channels))
            setattr(self, f'tanh{i+1}', nn.Tanh())
            setattr(self, f'fc2{i+1}', nn.Linear(4*channels, int(self.band[i]*12)))
            setattr(self, f'glu{i+1}', nn.GLU())

    def forward(self, x):
        """
        Args:
            x: [B, N, T, K] where K=30 bands
        Returns:
            m: [B, F, T, 3, 2] where F=257 frequencies
        """
        for i in range(len(self.band)):
            x_band = x[:, :, :, i]
            out = getattr(self, f'norm{i+1}')(x_band)
            out = out.transpose(1, 2)
            out = getattr(self, f'fc1{i+1}')(out)
            out = getattr(self, f'tanh{i+1}')(out)
            out = getattr(self, f'fc2{i+1}')(out)
            out = getattr(self, f'glu{i+1}')(out)
            out = torch.reshape(out, [out.size(0), out.size(1), int(out.size(2)/6), 3, 2])

            if i == 0:
                m = out
            else:
                m = torch.cat((m, out), dim=2)

        return m.transpose(1, 2)
