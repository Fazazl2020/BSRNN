"""
Memory-Optimized Mamba SSM Implementation

This is a memory-efficient implementation of Selective State-Space Model (Mamba)
for speech enhancement. Optimized based on 2024 literature:
- SEMamba (IEEE SLT 2024): https://github.com/RoyChao19477/SEMamba
- Mamba-SEUNet (Dec 2024): https://arxiv.org/abs/2412.16626
- Vision Mamba (ICML 2024): https://github.com/hustvl/Vim

Key optimizations (literature-backed):
1. Gradient checkpointing (Vision Mamba): 50-60% memory reduction
2. Chunked selective scan (Mamba-2): Avoid materializing huge tensors
3. Standard d_state=16 (SEMamba, Mamba-1 standard)
4. Chunk size 64 (Mamba-2 recommended)
5. Mixed precision training support

Expected: ~50K params per block with 50-60% memory savings via checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math


class SelectiveSSM(nn.Module):
    """
    Selective State-Space Model (S6) - Core of Mamba.

    Memory-optimized with chunked processing.
    Literature-backed parameters:
    - d_state=16 (Mamba-1/SEMamba standard)
    - chunk_size=64 (Mamba-2 recommendation)
    """
    def __init__(self, d_model, d_state=16, dt_rank='auto', d_conv=4, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.chunk_size = chunk_size

        # dt_rank: rank for delta projection
        if dt_rank == 'auto':
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        # A: SSM parameter (diagonal, log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection parameter
        self.D = nn.Parameter(torch.ones(d_model))

        # Projections for selective parameters (input-dependent)
        self.x_proj_dt = nn.Linear(d_model, self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

        # Depthwise convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            bias=True,
            groups=d_model,
            padding=d_conv - 1
        )

        # Initialize dt projection bias
        self._init_dt_bias()

    def _init_dt_bias(self):
        """Initialize dt projection bias for stability"""
        dt = torch.exp(
            torch.rand(self.d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        """
        Forward pass with chunked processing to save memory.

        Args:
            x: (B, L, D) - input sequence
        Returns:
            y: (B, L, D) - output sequence
        """
        B, L, D = x.shape

        # 1. Convolutional projection for local context
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # 2. Compute input-dependent parameters
        dt_input = self.x_proj_dt(x_conv)  # (B, L, dt_rank)
        dt = self.dt_proj(dt_input)  # (B, L, D)
        dt = F.softplus(dt)  # Ensure positive

        B_ssm = self.B_proj(x_conv)  # (B, L, N)
        C_ssm = self.C_proj(x_conv)  # (B, L, N)

        # 3. Get A matrix
        A = -torch.exp(self.A_log.float())  # (D, N) - negative for stability

        # 4. CHUNKED selective scan to avoid huge tensors
        y = self._chunked_selective_scan(x_conv, dt, B_ssm, C_ssm, A, self.D)

        return y

    def _chunked_selective_scan(self, x, dt, B, C, A, D):
        """
        Chunked selective scan to reduce memory usage.

        Instead of materializing A_bar and B_bar for entire sequence,
        process in chunks of size chunk_size.

        Args:
            x: (B, L, D)
            dt: (B, L, D)
            B: (B, L, N)
            C: (B, L, N)
            A: (D, N)
            D: (D,)
        Returns:
            y: (B, L, D)
        """
        B_batch, L, D = x.shape
        N = self.d_state

        # Initialize hidden state
        h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)

        outputs = []

        # Process in chunks
        for i in range(0, L, self.chunk_size):
            end = min(i + self.chunk_size, L)
            chunk_len = end - i

            # Get chunk slices
            x_chunk = x[:, i:end]  # (B, chunk_len, D)
            dt_chunk = dt[:, i:end]  # (B, chunk_len, D)
            B_chunk = B[:, i:end]  # (B, chunk_len, N)
            C_chunk = C[:, i:end]  # (B, chunk_len, N)

            # Discretize for this chunk only (MEMORY SAVING!)
            # A_bar = exp(dt * A)
            dt_A = dt_chunk.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, chunk_len, D, N)

            # NUMERICAL STABILITY: Clamp dt_A to prevent overflow in exp()
            dt_A = torch.clamp(dt_A, min=-10.0, max=0.0)  # Should be negative since A is negative
            A_bar = torch.exp(dt_A)  # Only chunk_len frames, not full L!

            # B_bar = dt * B (approximately, for small dt)
            # NUMERICAL STABILITY: Clamp dt to prevent extreme values
            dt_chunk_clamped = torch.clamp(dt_chunk, min=0.0, max=1.0)
            B_bar = dt_chunk_clamped.unsqueeze(-1) * B_chunk.unsqueeze(2)  # (B, chunk_len, D, N)

            # Scan this chunk
            chunk_output, h = self._scan_chunk(x_chunk, A_bar, B_bar, C_chunk, D, h)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=1)  # (B, L, D)

    def _scan_chunk(self, x, A_bar, B_bar, C, D, h_init):
        """
        Perform selective scan on a single chunk.

        Args:
            x: (B, chunk_len, D)
            A_bar: (B, chunk_len, D, N)
            B_bar: (B, chunk_len, D, N)
            C: (B, chunk_len, N)
            D: (D,)
            h_init: (B, D, N) - initial hidden state
        Returns:
            outputs: (B, chunk_len, D)
            h_final: (B, D, N) - final hidden state
        """
        B, chunk_len, D_dim, N = A_bar.shape
        h = h_init
        outputs = []

        # NUMERICAL STABILITY: Clamp A_bar to prevent state explosion
        # A_bar should be in (0, 1) for proper decay, but numerical errors can push it >= 1
        A_bar = torch.clamp(A_bar, min=0.0, max=0.999)

        # NUMERICAL STABILITY: Clamp B_bar to prevent input explosion
        B_bar = torch.clamp(B_bar, min=-10.0, max=10.0)

        # Recurrent computation
        for t in range(chunk_len):
            # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)  # (B, D, N)

            # NUMERICAL STABILITY: Clamp hidden state to prevent explosion
            h = torch.clamp(h, min=-100.0, max=100.0)

            # y_t = C_t * h_t + D * x_t
            y_t = torch.einsum('bdn,bn->bd', h, C[:, t]) + D * x[:, t]  # (B, D)

            # NUMERICAL STABILITY: Clamp output to prevent NaN propagation
            y_t = torch.clamp(y_t, min=-50.0, max=50.0)

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, chunk_len, D)
        return y, h


class MambaBlock(nn.Module):
    """
    Complete Mamba block with projections and residual connections.

    Optimized with expand_factor=1 for speech (vs 2 for NLP).
    Supports gradient checkpointing for memory efficiency.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=1, chunk_size=64, use_checkpoint=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor  # For expand=1, d_inner = d_model
        self.use_checkpoint = use_checkpoint

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Input projection (expansion)
        # Projects to 2*d_inner for x and z paths
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Selective SSM
        self.ssm = SelectiveSSM(
            d_model=self.d_inner,
            d_state=d_state,
            d_conv=d_conv,
            chunk_size=chunk_size
        )

        # Output projection (compression)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _forward_impl(self, x):
        """Actual forward implementation (for checkpointing)"""
        # Pre-norm
        x = self.norm(x)

        # Split into x and z paths
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Apply SSM to x path
        x = self.ssm(x)

        # Gated multiplication: x * SiLU(z)
        x = x * F.silu(z)

        # Output projection
        output = self.out_proj(x)

        return output

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        residual = x

        # Use gradient checkpointing if enabled (50-60% memory savings)
        if self.use_checkpoint and self.training:
            output = checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            output = self._forward_impl(x)

        # Residual connection
        return output + residual


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba block (optional, for last layers if needed).

    Processes sequence in both forward and backward directions.
    Use sparingly due to 2x memory cost.
    Supports gradient checkpointing for memory efficiency.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=1, chunk_size=64, use_checkpoint=False):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint

        # Forward and backward Mamba blocks
        self.mamba_forward = MambaBlock(d_model, d_state, d_conv, expand_factor, chunk_size, use_checkpoint)
        self.mamba_backward = MambaBlock(d_model, d_state, d_conv, expand_factor, chunk_size, use_checkpoint)

        # Combine forward and backward
        self.combine = nn.Linear(d_model * 2, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        # Forward pass
        x_forward = self.mamba_forward(x)

        # Backward pass (flip sequence)
        x_reversed = torch.flip(x, dims=[1])
        x_backward = self.mamba_backward(x_reversed)
        x_backward = torch.flip(x_backward, dims=[1])

        # Combine
        x_combined = torch.cat([x_forward, x_backward], dim=-1)
        output = self.combine(x_combined)

        return self.norm(output)


# Test code
if __name__ == '__main__':
    print("Testing Optimized Mamba SSM Implementation")
    print("=" * 60)

    # Test parameters (literature-backed values)
    batch_size = 2
    seq_len = 100
    d_model = 128
    d_state = 16  # Mamba-1/SEMamba standard
    expand_factor = 1
    chunk_size = 64  # Mamba-2 recommendation
    use_checkpoint = True  # Enable gradient checkpointing

    # Create model
    print("\n1. Creating MambaBlock with gradient checkpointing...")
    model = MambaBlock(d_model=d_model, d_state=d_state, expand_factor=expand_factor,
                      chunk_size=chunk_size, use_checkpoint=use_checkpoint)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params/1e3:.1f}K")

    # Test forward pass
    print("\n2. Testing forward pass...")
    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        y = model(x)

    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {y.shape}")
    assert y.shape == x.shape, "Shape mismatch!"
    print("   PASS: Shape correct")

    # Test gradient flow
    print("\n3. Testing gradient flow...")
    x_train = torch.randn(batch_size, seq_len, d_model, requires_grad=False)
    target = torch.randn(batch_size, seq_len, d_model)

    model.train()
    output = model(x_train)
    loss = F.mse_loss(output, target)
    loss.backward()

    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Gradients computed: {has_grads}")
    assert has_grads, "No gradients!"
    print("   PASS: Gradients flow correctly")

    # Test bidirectional
    print("\n4. Testing BidirectionalMambaBlock with gradient checkpointing...")
    bidir_model = BidirectionalMambaBlock(d_model=d_model, d_state=d_state, expand_factor=expand_factor,
                                         chunk_size=chunk_size, use_checkpoint=use_checkpoint)
    bidir_params = sum(p.numel() for p in bidir_model.parameters())
    print(f"   Parameters: {bidir_params/1e3:.1f}K")

    with torch.no_grad():
        y_bidir = bidir_model(x)
    print(f"   Output shape: {y_bidir.shape}")
    assert y_bidir.shape == x.shape, "Shape mismatch!"
    print("   PASS: Bidirectional works")

    # Memory comparison
    print("\n5. Memory Comparison:")
    print(f"   Chunk size: {chunk_size} frames")
    print(f"   Full sequence would create: ({batch_size}, {seq_len}, {d_model}, {d_state})")
    print(f"   Chunked creates: ({batch_size}, {chunk_size}, {d_model}, {d_state}) at a time")
    memory_reduction = seq_len / chunk_size
    print(f"   Memory reduction: ~{memory_reduction:.1f}x for discretization tensors")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print(f"Optimized MambaBlock: {total_params/1e3:.1f}K params")
    print(f"Memory-efficient chunked processing ready")
