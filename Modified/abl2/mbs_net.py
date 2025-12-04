"""
Ablation 2: Dual-Path BiMamba + Uniform Decoder

Model with both IntraBand and CrossBand bidirectional Mamba.
Tests effectiveness of cross-band modeling.

Expected Performance: PESQ 3.1-3.2
Parameters: ~4.42M
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
    Ablation 2: Dual-Path Bidirectional Mamba + Uniform Decoder

    Architecture:
        1. BandSplit (30 psychoacoustic bands): ~50K params
        2. Dual-Path BiMamba (4 layers × [intra + cross]): ~920K params
        3. Uniform Decoder (4x): ~3.45M params
        Total: ~4.42M params
    """
    def __init__(
        self,
        num_channel=128,
        num_layers=4,
        num_bands=30,
        d_state=16,
        chunk_size=32
    ):
        super().__init__()
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.num_bands = num_bands

        self.band_split = BandSplit(channels=num_channel)

        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_layers.append(nn.ModuleDict({
                'intra_band': IntraBandBiMamba(channels=num_channel, d_state=d_state, d_conv=4, chunk_size=chunk_size),
                'cross_band': CrossBandBiMamba(channels=num_channel, d_state=d_state, d_conv=4, num_bands=num_bands, chunk_size=chunk_size)
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
