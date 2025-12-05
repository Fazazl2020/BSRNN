"""
Ablation 2: Dual-Path BiMamba + Uniform Decoder (LITERATURE-BACKED)

Based on 2024-2025 research with proper parameters:
- num_layers=1 (single dual-path layer sufficient with gradient checkpointing)
- d_state=16 (SEMamba standard)
- chunk_size=64 (Mamba-2 recommendation)
- Gradient checkpointing + mixed precision

Expected Performance: PESQ 3.25-3.35
Parameters: ~2.6M
Memory: Fits with batch_size=6
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Baseline'))
from module import BandSplit

from mamba import IntraBandBiMamba, CrossBandBiMamba, MaskDecoderUniform


class MBS_Net(nn.Module):
    """
    Ablation 2: Single-Layer Dual-Path BiMamba (Literature-Backed)

    Architecture:
    - 1 dual-path layer (intra+cross BiMamba)
    - d_state=16 (SEMamba standard)
    - chunk_size=64 (Mamba-2)
    - Gradient checkpointing enabled
    Total: ~2.6M params
    """
    def __init__(
        self,
        num_channel=128,
        num_layers=1,  # Literature-backed single layer
        num_bands=30,
        d_state=16,    # SEMamba standard
        chunk_size=64,  # Mamba-2 recommendation
        use_checkpoint=True
    ):
        super().__init__()
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.num_bands = num_bands

        self.band_split = BandSplit(channels=num_channel)

        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_layers.append(nn.ModuleDict({
                'intra_band': IntraBandBiMamba(channels=num_channel, d_state=d_state, d_conv=4, chunk_size=chunk_size, use_checkpoint=use_checkpoint),
                'cross_band': CrossBandBiMamba(channels=num_channel, d_state=d_state, d_conv=4, num_bands=num_bands, chunk_size=chunk_size, use_checkpoint=use_checkpoint)
            }))

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
            features = layer['intra_band'](features)
            features = layer['cross_band'](features)

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
