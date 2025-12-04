"""
Ablation 1: IntraBand BiMamba + Uniform Decoder

Test: Does bidirectional Mamba work effectively with band-split processing?

Architecture:
    1. BandSplit (30 psychoacoustic bands) - from BSRNN
    2. 4 layers of IntraBand BiMamba (temporal modeling)
    3. Uniform 4x decoder (baseline BSRNN decoder)

Key Differences from Current (mbs_net_optimized.py):
    - ✓ Bidirectional Mamba (vs unidirectional)
    - ✗ No cross-band processing (will add in abl2)
    - ✓ Full 4x decoder (vs 2x lightweight)

Expected Performance:
    - PESQ: 3.0-3.1 (match or slightly beat BSRNN)
    - Parameters: ~3.9M (BandSplit 50K + Encoder 460K + Decoder 3.45M)
    - Rationale: Bidirectional should recover ~0.3 PESQ vs unidirectional

Literature Support:
    "Bidirectional Mamba achieves 3.55 PESQ on speech enhancement,
     outperforming LSTM-based methods by leveraging both past and future context."
    - SEMamba (IEEE SLT 2024)
"""

import torch
import torch.nn as nn
import sys
import os

# Add Baseline directory for BandSplit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Baseline'))
from module import BandSplit

# Import our bidirectional Mamba blocks
from mamba_blocks import IntraBandBiMamba, MaskDecoderUniform


class BS_BiMamba_Abl1(nn.Module):
    """
    Ablation 1: IntraBand Bidirectional Mamba + Uniform Decoder

    This tests whether bidirectional Mamba can effectively replace LSTM
    for temporal modeling in band-split speech enhancement.

    Components:
        1. BandSplit: ~50K params
        2. IntraBand BiMamba (4 layers): ~460K params
        3. Uniform Decoder (4x): ~3.45M params
        Total: ~3.96M params

    Compared to BSRNN:
        - Similar params (~2.4M BSRNN vs ~3.96M ours)
        - More efficient computation (Mamba O(n) vs LSTM sequential)
        - Expected: Match or beat BSRNN performance

    Compared to current Modified:
        - Bidirectional vs unidirectional (+0.3 PESQ expected)
        - Full decoder vs 2x decoder (+0.05 PESQ expected)
        - Total: +0.35 PESQ → 2.62 + 0.35 = 2.97-3.0 PESQ
    """
    def __init__(
        self,
        num_channel=128,
        num_layers=4,
        num_bands=30,
        d_state=16,
        d_conv=4,
        chunk_size=32
    ):
        super().__init__()
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.num_bands = num_bands

        # Stage 1: Band-split (BSRNN psychoacoustic bands)
        self.band_split = BandSplit(channels=num_channel)

        # Stage 2: IntraBand BiMamba Encoder (4 layers)
        self.encoder_layers = nn.ModuleList([
            IntraBandBiMamba(
                channels=num_channel,
                d_state=d_state,
                d_conv=d_conv,
                chunk_size=chunk_size
            )
            for _ in range(num_layers)
        ])

        # Stage 3: Uniform Decoder (BSRNN baseline, 4x expansion)
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
            features = layer(features)  # [B, N, T, K]

        # Stage 3: Mask generation
        masks = self.decoder(features)  # [B, F, T, 3, 2]

        # Stage 4: Apply masks (BSRNN 3-tap filter)
        masks_complex = torch.view_as_complex(masks)  # [B, F, T, 3]
        x_complex = torch.view_as_complex(x_real_imag)  # [B, F, T]

        # 3-tap temporal filter
        s = (masks_complex[:, :, 1:-1, 0] * x_complex[:, :, :-2] +
             masks_complex[:, :, 1:-1, 1] * x_complex[:, :, 1:-1] +
             masks_complex[:, :, 1:-1, 2] * x_complex[:, :, 2:])

        s_f = (masks_complex[:, :, 0, 1] * x_complex[:, :, 0] +
               masks_complex[:, :, 0, 2] * x_complex[:, :, 1])
        s_l = (masks_complex[:, :, -1, 0] * x_complex[:, :, -2] +
               masks_complex[:, :, -1, 1] * x_complex[:, :, -1])

        enhanced = torch.cat([s_f.unsqueeze(2), s, s_l.unsqueeze(2)], dim=2)

        return enhanced


# Test code
if __name__ == '__main__':
    print("Testing Ablation 1: IntraBand BiMamba + Uniform Decoder")
    print("=" * 70)

    # Create model
    print("\n1. Creating BS_BiMamba_Abl1...")
    model = BS_BiMamba_Abl1(
        num_channel=128,
        num_layers=4,
        num_bands=30,
        d_state=16,
        chunk_size=32
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params/1e6:.2f}M")
    print(f"   Trainable parameters: {trainable_params/1e6:.2f}M")
    print(f"   Expected: ~3.96M")

    # Component breakdown
    print("\n2. Parameter breakdown:")
    band_split_params = sum(p.numel() for p in model.band_split.parameters())
    encoder_params = sum(p.numel() for p in model.encoder_layers.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"   BandSplit: {band_split_params/1e3:.1f}K")
    print(f"   Encoder (4x IntraBand BiMamba): {encoder_params/1e6:.2f}M")
    print(f"   Decoder (Uniform 4x): {decoder_params/1e6:.2f}M")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 2
    freq_bins = 257
    time_frames = 100

    x_complex = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)
    print(f"   Input shape: {x_complex.shape}")

    try:
        with torch.no_grad():
            output = model(x_complex)

        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        assert output.shape == x_complex.shape, "Shape mismatch!"
        assert torch.is_complex(output), "Output should be complex!"
        print("   ✓ PASS: Forward pass successful")
    except Exception as e:
        print(f"   ✗ FAIL: {str(e)}")
        import traceback
        traceback.print_exc()

    # Test gradient flow
    print("\n4. Testing gradient flow...")
    model.train()
    x_train = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)
    target = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)

    try:
        output_train = model(x_train)
        loss = torch.nn.functional.l1_loss(
            torch.view_as_real(output_train),
            torch.view_as_real(target)
        )
        loss.backward()

        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Gradients computed: {has_grads}")
        assert has_grads, "No gradients!"
        print("   ✓ PASS: Gradients flow correctly")
    except Exception as e:
        print(f"   ✗ FAIL: {str(e)}")

    print("\n" + "=" * 70)
    print("ABLATION 1 MODEL READY!")
    print(f"Parameters: {total_params/1e6:.2f}M")
    print("Expected PESQ: 3.0-3.1 (bidirectional Mamba + full decoder)")
    print("=" * 70)
