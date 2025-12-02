"""
MBS-Net Optimized: Memory-Efficient Mamba Band-Split Network (FIXED)

This is the CORRECTED optimized implementation solving OOM and bugs.

Key fixes:
1. Proper BandSplit input format (real/imag)
2. Correct decoder using BSRNN MaskDecoder
3. Proper mask application
4. Fixed parameter count

Expected: ~2.3M params, 3.50-3.65 PESQ (with PCS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add Baseline directory to path for BandSplit and MaskDecoder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Baseline'))
from module import BandSplit, MaskDecoder

# Import optimized Mamba
from real_mamba_optimized import MambaBlock


class SharedMambaEncoder(nn.Module):
    """
    Shared Mamba encoder for both magnitude and phase features.
    Uses unidirectional Mamba for temporal processing and simple MLP for cross-band.
    """
    def __init__(self, num_channel=128, num_bands=30, num_layers=4, d_state=12, chunk_size=32):
        super().__init__()
        self.num_channel = num_channel
        self.num_bands = num_bands
        self.num_layers = num_layers

        # Stack of unidirectional Mamba layers for temporal processing
        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=num_channel,
                d_state=d_state,
                d_conv=4,
                expand_factor=1,
                chunk_size=chunk_size
            )
            for _ in range(num_layers)
        ])

        # Cross-band fusion: Simple MLP (not Mamba)
        self.cross_band_net = nn.Sequential(
            nn.Linear(num_channel, num_channel),
            nn.LayerNorm(num_channel),
            nn.GELU()
        )
        self.cross_band_norm = nn.LayerNorm(num_channel)

    def forward(self, x):
        """
        Args:
            x: [B, N, T, K] - Band-split features
        Returns:
            features: [B, N, T, K] - Encoded features
        """
        B, N, T, K = x.shape

        # Temporal processing per band with unidirectional Mamba
        x_bands = x.permute(0, 3, 2, 1).contiguous()  # [B, K, T, N]
        x_bands = x_bands.reshape(B * K, T, N)  # [B*K, T, N]

        # Apply Mamba layers
        out = x_bands
        for layer in self.mamba_layers:
            out = layer(out)  # Unidirectional Mamba with residual

        # Reshape back
        out = out.reshape(B, K, T, N).permute(0, 3, 2, 1)  # [B, N, T, K]

        # Cross-band fusion with lightweight MLP
        out_cross = out.permute(0, 2, 3, 1).contiguous()  # [B, T, K, N]
        out_cross = self.cross_band_net(out_cross)  # [B, T, K, N]
        out_cross = out_cross.permute(0, 3, 1, 2)  # [B, N, T, K]

        # Residual + norm
        features = self.cross_band_norm((out + out_cross).permute(0, 2, 3, 1))
        features = features.permute(0, 3, 1, 2)  # [B, N, T, K]

        return features


class MBS_Net(nn.Module):
    """
    MBS-Net: Memory-Efficient Mamba Band-Split Network (Optimized and Fixed)

    Architecture:
    1. BandSplit: 30 psychoacoustic bands
    2. Shared Mamba Encoder: 4 unidirectional Mamba layers
    3. MaskDecoder: Generates enhancement masks (from BSRNN)
    4. Mask Application: Apply masks to input spectrum

    Expected: ~2.3M params, 3.50-3.65 PESQ (with PCS)
    """
    def __init__(self, num_channel=128, num_layers=4, num_bands=30, d_state=12, chunk_size=32):
        super().__init__()
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.num_bands = num_bands

        # Stage 1: Band-split (from BSRNN)
        self.band_split = BandSplit(channels=num_channel)

        # Stage 2: Shared Mamba encoder
        self.encoder = SharedMambaEncoder(
            num_channel=num_channel,
            num_bands=num_bands,
            num_layers=num_layers,
            d_state=d_state,
            chunk_size=chunk_size
        )

        # Stage 3: Mask decoder (from BSRNN)
        self.mask_decoder = MaskDecoder(channels=num_channel)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following BSRNN strategy"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, use_pcs=False, pcs_alpha=0.3):
        """
        Args:
            x: Complex spectrogram [B, F, T] or [B, 2, F, T]
            use_pcs: Whether to apply Perceptual Contrast Stretching
            pcs_alpha: PCS strength parameter
        Returns:
            enhanced: Enhanced complex spectrogram [B, F, T]
        """
        # Convert to real/imag format for BandSplit
        # BandSplit expects [B, F, T, 2] format (like torch.view_as_real output)
        if torch.is_complex(x):
            # [B, F, T] complex -> [B, F, T, 2] real/imag
            x_real_imag = torch.view_as_real(x)
        else:
            if x.ndim == 4 and x.shape[-1] == 2:
                # Already in [B, F, T, 2] format
                x_real_imag = x
            elif x.ndim == 4 and x.shape[1] == 2:
                # [B, 2, F, T] -> [B, F, T, 2]
                x_real_imag = x.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

        # Stage 1: Band-split
        # BandSplit outputs [B, T, N, K], need to transpose to [B, N, T, K]
        z = self.band_split(x_real_imag).transpose(1, 2)  # [B, N, T, 30]

        # Stage 2: Shared encoding with Mamba
        features = self.encoder(z)  # [B, N, T, 30]

        # Stage 3: Mask generation
        masks = self.mask_decoder(features)  # [B, 2, F, 3, 2]

        # Stage 4: Apply masks to input spectrum
        # Following BSRNN pattern (see module.py lines 62-69)
        # masks shape: [B, 2, F, 3, 2]
        # Convert masks to complex for easier manipulation
        masks_complex = torch.view_as_complex(masks)  # [B, 2, F, 3]
        x_complex = torch.view_as_complex(x_real_imag)  # [B, F, T]

        # Apply 3-tap filter (like BSRNN)
        # m[:,:,1:-1,0]*x[:,:,:-2] + m[:,:,1:-1,1]*x[:,:,1:-1] + m[:,:,1:-1,2]*x[:,:,2:]
        s = (masks_complex[:, 0, 1:-1, 0].unsqueeze(-1) * x_complex[:, :-2, :] +
             masks_complex[:, 0, 1:-1, 1].unsqueeze(-1) * x_complex[:, 1:-1, :] +
             masks_complex[:, 0, 1:-1, 2].unsqueeze(-1) * x_complex[:, 2:, :])

        # First and last frequency bins (special cases)
        s_f = (masks_complex[:, 0, 0, 1].unsqueeze(-1) * x_complex[:, 0, :] +
               masks_complex[:, 0, 0, 2].unsqueeze(-1) * x_complex[:, 1, :])
        s_l = (masks_complex[:, 0, -1, 0].unsqueeze(-1) * x_complex[:, -2, :] +
               masks_complex[:, 0, -1, 1].unsqueeze(-1) * x_complex[:, -1, :])

        # Concatenate
        enhanced = torch.cat([s_f.unsqueeze(1), s, s_l.unsqueeze(1)], dim=1)  # [B, F, T]

        # Optional PCS post-processing
        if use_pcs:
            enhanced = self._apply_pcs(enhanced, pcs_alpha)

        return enhanced

    def _apply_pcs(self, spec, alpha=0.3):
        """
        Perceptual Contrast Stretching (PCS) post-processing.
        From SEMamba (2024): +0.14 PESQ improvement, no trainable params.
        """
        mag = torch.abs(spec)
        phase = torch.angle(spec)

        # Contrast stretching on magnitude
        mag_mean = mag.mean(dim=(1, 2), keepdim=True)
        mag_std = mag.std(dim=(1, 2), keepdim=True)
        mag_stretched = (mag - mag_mean) / (mag_std + 1e-8)
        mag_stretched = mag_stretched * alpha + mag

        # Reconstruct complex spectrum
        enhanced_pcs = mag_stretched * torch.exp(1j * phase)
        return enhanced_pcs


# Test code
if __name__ == '__main__':
    print("Testing Fixed Optimized MBS-Net")
    print("=" * 60)

    # Create model
    print("\n1. Creating MBS_Net (Optimized & Fixed)...")
    model = MBS_Net(
        num_channel=128,
        num_layers=4,
        num_bands=30,
        d_state=12,
        chunk_size=32
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params/1e6:.2f}M")
    print(f"   Trainable parameters: {trainable_params/1e6:.2f}M")

    # Verify parameter count
    if total_params < 1.5e6:
        print(f"   WARNING: Parameters too low! Expected ~2.3M")
    elif total_params > 3.5e6:
        print(f"   WARNING: Parameters too high! Expected ~2.3M")
    else:
        print(f"   GOOD: Parameters within target range")

    # Test with complex input
    print("\n2. Testing forward pass with complex input...")
    batch_size = 2
    freq_bins = 257
    time_frames = 100

    x_complex = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)
    print(f"   Input shape: {x_complex.shape}")

    try:
        with torch.no_grad():
            output = model(x_complex, use_pcs=False)

        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        assert output.shape == x_complex.shape, "Shape mismatch!"
        assert torch.is_complex(output), "Output should be complex!"
        print("   PASS: Forward pass successful")
    except Exception as e:
        print(f"   FAIL: {str(e)}")
        import traceback
        traceback.print_exc()

    # Test with real/imag input
    print("\n3. Testing forward pass with real/imag input...")
    x_real_imag = torch.randn(batch_size, 2, freq_bins, time_frames)
    print(f"   Input shape: {x_real_imag.shape}")

    try:
        with torch.no_grad():
            output2 = model(x_real_imag, use_pcs=False)

        print(f"   Output shape: {output2.shape}")
        print(f"   Output dtype: {output2.dtype}")
        print("   PASS: Real/imag input works")
    except Exception as e:
        print(f"   FAIL: {str(e)}")

    # Test with PCS
    print("\n4. Testing PCS post-processing...")
    try:
        with torch.no_grad():
            output_pcs = model(x_complex, use_pcs=True, pcs_alpha=0.3)
        print(f"   Output with PCS shape: {output_pcs.shape}")
        print("   PASS: PCS works")
    except Exception as e:
        print(f"   FAIL: {str(e)}")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    x_train = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64, requires_grad=False)
    target = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)

    model.train()
    try:
        output_train = model(x_train)
        loss = F.l1_loss(torch.view_as_real(output_train), torch.view_as_real(target))
        loss.backward()

        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Gradients computed: {has_grads}")
        assert has_grads, "No gradients!"
        print("   PASS: Gradients flow correctly")
    except Exception as e:
        print(f"   FAIL: {str(e)}")

    # Verify Mamba usage
    print("\n6. Verifying Mamba usage...")
    from real_mamba_optimized import MambaBlock
    mamba_blocks = [m for m in model.modules() if isinstance(m, MambaBlock)]
    print(f"   MambaBlock instances: {len(mamba_blocks)}")
    if len(mamba_blocks) > 0:
        print("   PASS: Using real optimized Mamba")
    else:
        print("   ERROR: Not using Mamba!")

    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print(f"MBS-Net Optimized: {total_params/1e6:.2f}M params")
    print("Ready for training if all tests passed!")
