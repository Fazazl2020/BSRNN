"""
Ablation 3: Full BS-BiMamba with Adaptive Decoder

Complete model with all novel components.
Expected to achieve best performance with optimal parameter efficiency.

Expected Performance: PESQ 3.2-3.5
Parameters: ~2.82M
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Baseline'))
from module import BandSplit

from mamba import IntraBandBiMamba, CrossBandBiMamba, MaskDecoderAdaptive


class MBS_Net(nn.Module):
    """
    Ablation 3: Full BS-BiMamba with Adaptive Decoder

    Architecture:
        1. BandSplit (30 psychoacoustic bands): ~50K params
        2. Dual-Path BiMamba (4 layers × [intra + cross]): ~920K params
        3. Adaptive Decoder (2x/3x/4x): ~1.85M params
        Total: ~2.82M params

    This is the complete model with all three novel contributions:
    - Bidirectional Mamba for temporal modeling
    - Cross-band Mamba for spectral modeling
    - Frequency-adaptive decoder for optimal parameter allocation
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

        self.decoder = MaskDecoderAdaptive(channels=num_channel)
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
