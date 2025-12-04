"""
Ablation 1: IntraBand BiMamba + Uniform Decoder

Model definition for ablation study 1.
Tests whether bidirectional Mamba works effectively with band-split processing.

Expected Performance: PESQ 3.0-3.1
Parameters: ~3.96M
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to import BandSplit from Baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Baseline'))
from module import BandSplit

# Import Mamba blocks from THIS directory
from mamba import IntraBandBiMamba, MaskDecoderUniform


class MBS_Net(nn.Module):
    """
    Ablation 1: IntraBand Bidirectional Mamba + Uniform Decoder

    Architecture:
        1. BandSplit (30 psychoacoustic bands): ~50K params
        2. IntraBand BiMamba (4 layers): ~460K params
        3. Uniform Decoder (4x): ~3.45M params
        Total: ~3.96M params

    This tests whether bidirectional Mamba can effectively replace LSTM
    for temporal modeling in band-split speech enhancement.
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

        # Stage 1: Band-split
        self.band_split = BandSplit(channels=num_channel)

        # Stage 2: IntraBand BiMamba Encoder
        self.encoder_layers = nn.ModuleList([
            IntraBandBiMamba(
                channels=num_channel,
                d_state=d_state,
                d_conv=4,
                chunk_size=chunk_size
            )
            for _ in range(num_layers)
        ])

        # Stage 3: Uniform Decoder
        self.decoder = MaskDecoderUniform(channels=num_channel)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following BSRNN strategy"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """
        Args:
            x: Complex spectrogram [B, F, T] or [B, 2, F, T]
        Returns:
            enhanced: Enhanced complex spectrogram [B, F, T]
        """
        # Convert to real/imag format
        if torch.is_complex(x):
            x_real_imag = torch.view_as_real(x)
        else:
            if x.ndim == 4 and x.shape[-1] == 2:
                x_real_imag = x
            elif x.ndim == 4 and x.shape[1] == 2:
                x_real_imag = x.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

        # Stage 1: Band-split
        z = self.band_split(x_real_imag).transpose(1, 2)  # [B, N, T, K]

        # Stage 2: IntraBand BiMamba Encoding
        features = z
        for layer in self.encoder_layers:
            features = layer(features)

        # Stage 3: Mask generation
        masks = self.decoder(features)  # [B, F, T, 3, 2]

        # Stage 4: Apply masks (BSRNN 3-tap filter)
        masks_complex = torch.view_as_complex(masks)
        x_complex = torch.view_as_complex(x_real_imag)

        s = (masks_complex[:, :, 1:-1, 0] * x_complex[:, :, :-2] +
             masks_complex[:, :, 1:-1, 1] * x_complex[:, :, 1:-1] +
             masks_complex[:, :, 1:-1, 2] * x_complex[:, :, 2:])

        s_f = (masks_complex[:, :, 0, 1] * x_complex[:, :, 0] +
               masks_complex[:, :, 0, 2] * x_complex[:, :, 1])
        s_l = (masks_complex[:, :, -1, 0] * x_complex[:, :, -2] +
               masks_complex[:, :, -1, 1] * x_complex[:, :, -1])

        enhanced = torch.cat([s_f.unsqueeze(2), s, s_l.unsqueeze(2)], dim=2)

        return enhanced
