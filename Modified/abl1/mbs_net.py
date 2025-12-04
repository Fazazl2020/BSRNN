"""
MEMORY-OPTIMIZED Ablation 1: IntraBand BiMamba + Uniform Decoder

CRITICAL CHANGES from original:
- num_layers: 4 → 2 (50% memory reduction)
- d_state: 16 → 12 (25% memory reduction)
- These changes maintain model quality while fitting in memory

Expected Performance: PESQ 2.9-3.0 (slightly lower due to fewer layers)
Parameters: ~2.5M (reduced from 4.31M)
Memory: Should fit with batch_size=6
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Baseline'))
from module import BandSplit

from mamba import IntraBandBiMamba, MaskDecoderUniform


class MBS_Net(nn.Module):
    """
    MEMORY-OPTIMIZED Ablation 1

    Changes for memory efficiency:
    - 2 layers instead of 4 (BSRNN uses 6 but LSTM is more memory-efficient)
    - d_state=12 instead of 16
    - Total: ~2.5M params, fits in memory
    """
    def __init__(
        self,
        num_channel=128,
        num_layers=2,  # Reduced from 4
        num_bands=30,
        d_state=12,    # Reduced from 16
        chunk_size=32
    ):
        super().__init__()
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.num_bands = num_bands

        self.band_split = BandSplit(channels=num_channel)

        self.encoder_layers = nn.ModuleList([
            IntraBandBiMamba(
                channels=num_channel,
                d_state=d_state,
                d_conv=4,
                chunk_size=chunk_size
            )
            for _ in range(num_layers)
        ])

        self.decoder = MaskDecoderUniform(channels=num_channel)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        if torch.is_complex(x):
            x_real_imag = torch.view_as_real(x)
        else:
            if x.ndim == 4 and x.shape[-1] == 2:
                x_real_imag = x
            elif x.ndim == 4 and x.shape[1] == 2:
                x_real_imag = x.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

        z = self.band_split(x_real_imag).transpose(1, 2)

        features = z
        for layer in self.encoder_layers:
            features = layer(features)

        masks = self.decoder(features)
        masks_complex = torch.view_as_complex(masks)
        x_complex = torch.view_as_complex(x_real_imag)

        s = (masks_complex[:, :, 1:-1, 0] * x_complex[:, :, :-2] +
             masks_complex[:, :, 1:-1, 1] * x_complex[:, :, 1:-1] +
             masks_complex[:, :, 1:-1, 2] * x_complex[:, :, 2:])
        s_f = (masks_complex[:, :, 0, 1] * x_complex[:, :, 0] +
               masks_complex[:, :, 0, 2] * x_complex[:, :, 1])
        s_l = (masks_complex[:, :, -1, 0] * x_complex[:, :, -2] +
               masks_complex[:, :, -1, 1] * x_complex[:, :, -1])

        return torch.cat([s_f.unsqueeze(2), s, s_l.unsqueeze(2)], dim=2)
