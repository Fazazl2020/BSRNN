"""
DB-Transform: Differential Band-Split Transformer with Learnable Harmonic Graph

Literature grounding:
1. BandSplit: BSRNN (Yu et al., Interspeech 2023)
2. Differential Attention: Microsoft (Ye et al., Oct 2024)
3. Multi-scale: Li et al., Interspeech 2023
4. Harmonic Graph: Graph audio literature (2023-2024)

Novel contributions:
1. First application of differential attention to frequency bands
2. Frequency-adaptive multi-scale fusion (psychoacoustic grounding)
3. Learnable harmonic graph for inter-band communication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add Baseline directory to path to import BandSplit and MaskDecoder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Baseline'))
from module import BandSplit, MaskDecoder


class DifferentialBandAttention(nn.Module):
    """
    Differential attention applied per frequency band.

    Based on: "Differential Transformer" (Microsoft, Oct 2024)
    Applied to: Frequency bands (novel contribution)
    """
    def __init__(self, num_channel=128, num_heads=4, num_bands=30):
        super().__init__()
        self.num_bands = num_bands
        self.num_heads = num_heads
        self.head_dim = num_channel // num_heads // 2  # Split for differential

        # Per-band attention parameters
        self.band_attns = nn.ModuleList([
            self._create_diff_attn(num_channel)
            for _ in range(num_bands)
        ])

        # Frequency-dependent lambda (learned per band)
        self.lambda_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(8, 16),  # 8 statistics from 2 attention maps
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            for _ in range(num_bands)
        ])

    def _create_diff_attn(self, dim):
        return nn.ModuleDict({
            'q_proj': nn.Linear(dim, dim, bias=False),
            'k_proj': nn.Linear(dim, dim, bias=False),
            'v_proj': nn.Linear(dim, dim // 2, bias=False),  # Half for differential
            'out_proj': nn.Linear(dim // 2, dim)
        })

    def forward(self, x):
        """
        x: [B, N, T, 30]
        """
        B, N, T, K = x.shape

        outputs = []
        for k in range(K):
            x_band = x[:, :, :, k].transpose(1, 2)  # [B, T, N]

            # Project
            q = self.band_attns[k]['q_proj'](x_band)  # [B, T, N]
            K_mat = self.band_attns[k]['k_proj'](x_band)
            v = self.band_attns[k]['v_proj'](x_band)  # [B, T, N/2]

            # Split for differential attention
            q1, q2 = q.chunk(2, dim=-1)  # [B, T, N/2] each
            k1, k2 = K_mat.chunk(2, dim=-1)

            # Compute two attention maps
            attn1 = torch.softmax(q1 @ k1.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
            attn2 = torch.softmax(q2 @ k2.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)

            # Predict lambda (noise cancellation weight)
            stats = torch.cat([
                attn1.mean(dim=(-2, -1), keepdim=True),
                attn1.std(dim=(-2, -1), keepdim=True),
                attn1.max(dim=-1)[0].max(dim=-1, keepdim=True)[0].unsqueeze(-1),
                attn1.min(dim=-1)[0].min(dim=-1, keepdim=True)[0].unsqueeze(-1),
                attn2.mean(dim=(-2, -1), keepdim=True),
                attn2.std(dim=(-2, -1), keepdim=True),
                attn2.max(dim=-1)[0].max(dim=-1, keepdim=True)[0].unsqueeze(-1),
                attn2.min(dim=-1)[0].min(dim=-1, keepdim=True)[0].unsqueeze(-1),
            ], dim=-1).squeeze(-2)  # [B, 8]

            lambda_k = self.lambda_predictor[k](stats).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

            # Differential attention (key innovation from Microsoft)
            attn_diff = attn1 - lambda_k * attn2

            # Apply to values
            out_band = attn_diff @ v  # [B, T, N/2]
            out_band = self.band_attns[k]['out_proj'](out_band)  # [B, T, N]

            outputs.append(out_band.transpose(1, 2).unsqueeze(-1))  # [B, N, T, 1]

        return torch.cat(outputs, dim=-1)  # [B, N, T, 30]


class FrequencyAdaptiveMultiScale(nn.Module):
    """
    Multi-scale fusion with frequency-dependent scale selection.

    Based on: "Multi-Scale Temporal Transformer" (Li et al., Interspeech 2023)
    Novel: Frequency-adaptive scale assignment (psychoacoustic grounding)
    """
    def __init__(self, num_channel=128, num_bands=30):
        super().__init__()

        # Get BSRNN band frequencies
        self.band_freqs = self._get_band_frequencies()

        # Scale configurations per frequency range
        self.scale_configs = self._assign_scales()

        # Learnable fusion weights (initialized with psychoacoustic priors)
        self.fusion_weights = nn.ParameterDict({
            f'band_{k}': nn.Parameter(torch.tensor(self.scale_configs[k]['init_weights']))
            for k in range(num_bands)
        })

        # Multi-scale processors
        self.scale_processors = nn.ModuleDict({
            f's_{s}': nn.Conv1d(num_channel, num_channel, kernel_size=s, padding=s//2)
            for s in [2, 4, 8, 16, 32]
        })

    def _get_band_frequencies(self):
        """Extract center frequencies from BSRNN band structure"""
        # From BSRNN: [2, 3, 3, ..., 16, 16, 17]
        band_widths = [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                       8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                       16, 16, 16, 16, 16, 16, 16, 17]

        # Assuming n_fft=512, sr=16000: freq_bin = sr/n_fft = 31.25 Hz
        freqs = []
        cumsum = 0
        for w in band_widths:
            center = (cumsum + w/2) * 31.25
            freqs.append(center)
            cumsum += w
        return freqs

    def _assign_scales(self):
        """Assign scales based on psychoacoustic TMTF"""
        configs = {}
        for k, f_center in enumerate(self.band_freqs):
            if f_center < 500:  # Low freq
                configs[k] = {
                    'scales': [8, 16, 32],
                    'init_weights': [0.1, 0.3, 0.6]  # Prefer long
                }
            elif f_center < 2000:  # Mid freq
                configs[k] = {
                    'scales': [4, 8, 16],
                    'init_weights': [0.3, 0.4, 0.3]  # Balanced
                }
            else:  # High freq
                configs[k] = {
                    'scales': [2, 4, 8],
                    'init_weights': [0.6, 0.3, 0.1]  # Prefer short
                }
        return configs

    def forward(self, x):
        """
        x: [B, N, T, 30]
        """
        B, N, T, K = x.shape

        outputs = []
        for k in range(K):
            x_band = x[:, :, :, k]  # [B, N, T]

            # Process at each scale
            scales = self.scale_configs[k]['scales']
            multi_scale_feats = []

            for s in scales:
                feat_s = self.scale_processors[f's_{s}'](x_band)  # [B, N, T]
                multi_scale_feats.append(feat_s)

            # Adaptive fusion
            weights = F.softmax(self.fusion_weights[f'band_{k}'], dim=0)

            fused = sum(w * f for w, f in zip(weights, multi_scale_feats))
            outputs.append(fused.unsqueeze(-1))

        return torch.cat(outputs, dim=-1)  # [B, N, T, 30]


class LearnableHarmonicGraph(nn.Module):
    """
    Learnable graph attention for inter-band harmonic relationships.

    Based on: Graph modeling for audio (2023-2024)
    Novel: Learnable harmonic connections for speech enhancement bands
    """
    def __init__(self, num_channel=128, num_bands=30, sparsity_k=5):
        super().__init__()
        self.num_bands = num_bands
        self.sparsity_k = sparsity_k

        # Learnable band embeddings (frequency-aware initialization)
        band_freqs = self._get_band_frequencies()
        freq_init = torch.log(torch.tensor(band_freqs) + 1.0)  # Log-frequency
        self.band_embed = nn.Parameter(
            freq_init.unsqueeze(-1) * torch.randn(num_bands, 64) * 0.1
        )

        # Graph attention layers
        self.graph_attn = GraphAttention(num_channel, num_channel, num_heads=4)

    def _get_band_frequencies(self):
        """Extract center frequencies from BSRNN band structure"""
        band_widths = [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                       8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                       16, 16, 16, 16, 16, 16, 16, 17]

        freqs = []
        cumsum = 0
        for w in band_widths:
            center = (cumsum + w/2) * 31.25
            freqs.append(center)
            cumsum += w
        return freqs

    def compute_adjacency(self):
        """
        Compute sparse adjacency based on band embeddings.

        Key idea: Harmonically related bands should have similar
        embeddings in learned space.
        """
        # Similarity matrix (all pairs)
        norm = self.band_embed.norm(dim=-1, keepdim=True)
        normalized_embed = self.band_embed / (norm + 1e-8)
        sim = torch.matmul(normalized_embed, normalized_embed.T)  # [30, 30]

        # Sparsify: Top-k per band
        values, indices = sim.topk(k=self.sparsity_k, dim=-1)

        # Symmetric sparse adjacency
        A_sparse = torch.zeros(self.num_bands, self.num_bands, device=sim.device)
        for i in range(self.num_bands):
            A_sparse[i, indices[i]] = values[i]
        A_sparse = (A_sparse + A_sparse.T) / 2

        return A_sparse

    def forward(self, x):
        """
        x: [B, N, T, 30] - band features
        """
        B, N, T, K = x.shape

        # Compute learned adjacency (which bands communicate)
        A_learned = self.compute_adjacency()  # [30, 30]

        # Reshape for graph processing
        x_graph = x.permute(0, 2, 3, 1)  # [B, T, 30, N]
        x_flat = x_graph.reshape(B*T, K, N)  # [B*T, 30, N]

        # Graph message passing
        x_out = self.graph_attn(x_flat, A_learned)  # [B*T, 30, N]

        # Reshape back
        x_out = x_out.reshape(B, T, K, N).permute(0, 3, 1, 2)  # [B, N, T, 30]

        return x_out

    def visualize_learned_graph(self):
        """
        For paper: Visualize learned harmonic connections.
        Returns adjacency matrix and discovered harmonic pairs.
        """
        A = self.compute_adjacency().detach().cpu().numpy()

        # Identify discovered harmonic relationships
        harmonic_pairs = []
        band_freqs = self._get_band_frequencies()

        for i in range(self.num_bands):
            for j in range(i+1, self.num_bands):
                if A[i, j] > 0.3:  # Strong connection threshold
                    f_i = band_freqs[i]
                    f_j = band_freqs[j]
                    ratio = f_j / f_i if f_j > f_i else f_i / f_j

                    # Check if close to harmonic ratio (2, 3, 4, ...)
                    for k in [2, 3, 4, 5]:
                        if abs(ratio - k) < 0.3:
                            harmonic_pairs.append({
                                'band_i': i,
                                'band_j': j,
                                'freq_i': f_i,
                                'freq_j': f_j,
                                'ratio': ratio,
                                'harmonic': k,
                                'weight': A[i, j]
                            })

        return A, harmonic_pairs


class GraphAttention(nn.Module):
    """
    Graph attention network for inter-band communication.
    """
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x, adj):
        """
        x: [B, N_nodes, D]
        adj: [N_nodes, N_nodes] - adjacency matrix
        """
        B, N, D = x.shape

        # Project
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        # Transpose for attention: [B, num_heads, N, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Mask with adjacency (broadcast to all heads and batches)
        adj_mask = adj.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        scores = scores.masked_fill(adj_mask == 0, float('-inf'))

        # Softmax
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle all-masked rows

        # Apply attention
        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, N, -1)

        # Final projection
        out = self.out_proj(out)

        return out


class DB_Transform(nn.Module):
    """
    Differential Band-Split Transformer with Learnable Harmonic Graph

    Literature grounding:
    1. BandSplit: BSRNN (Yu et al., Interspeech 2023)
    2. Differential Attention: Microsoft (Ye et al., Oct 2024)
    3. Multi-scale: Li et al., Interspeech 2023
    4. Harmonic Graph: Graph audio literature (2023-2024)

    Novel contributions:
    1. First application of differential attention to frequency bands
    2. Frequency-adaptive multi-scale fusion (psychoacoustic grounding)
    3. Learnable harmonic graph for inter-band communication
    """
    def __init__(self, num_channel=128, num_heads=4):
        super().__init__()

        # Stage 1: Band-split (from BSRNN)
        self.band_split = BandSplit(channels=num_channel)

        # Stage 2: Differential band attention (Microsoft 2024 → speech bands)
        self.band_attn = DifferentialBandAttention(num_channel, num_heads, 30)

        # Stage 3: Frequency-adaptive multi-scale (Li et al. 2023 → frequency-dependent)
        self.multi_scale = FrequencyAdaptiveMultiScale(num_channel, 30)

        # Stage 4: Learnable harmonic graph (GNN audio 2023-24 → speech SE)
        self.harmonic_graph = LearnableHarmonicGraph(num_channel, 30)

        # Stage 5: Mask decoder (from BSRNN)
        self.mask_decoder = MaskDecoder(channels=num_channel)

        # Normalization layers
        self.norm1 = nn.GroupNorm(1, num_channel)
        self.norm2 = nn.GroupNorm(1, num_channel)
        self.norm3 = nn.GroupNorm(1, num_channel)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following BSRNN strategy"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, return_analysis=False):
        """
        x: Complex spectrogram [B, 2, 257, T] or [B, 257, T] (complex)
        """
        # Handle both complex and real representations
        if not torch.is_complex(x):
            # If already in real format [B, 2, 257, T]
            if x.shape[1] == 2:
                x_real = x
                x_complex = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
            else:
                x_complex = x
                x_real = torch.view_as_real(x)
        else:
            x_complex = x
            x_real = torch.view_as_real(x)

        # Stage 1: Band-split
        z = self.band_split(x_real).transpose(1, 2)  # [B, N, T, 30]

        # Stage 2: Differential band attention
        z_attn = self.band_attn(z)
        z = self.norm1(z + z_attn)  # Residual

        # Stage 3: Multi-scale
        z_ms = self.multi_scale(z)
        z = self.norm2(z + z_ms)  # Residual

        # Stage 4: Harmonic graph
        z_graph = self.harmonic_graph(z)
        z = self.norm3(z + z_graph)  # Residual

        # Stage 5: Decode
        m = self.mask_decoder(z)
        m = torch.view_as_complex(m)

        # Apply mask (3-tap FIR)
        s = m[:,:,1:-1,0]*x_complex[:,:,:-2] + m[:,:,1:-1,1]*x_complex[:,:,1:-1] + m[:,:,1:-1,2]*x_complex[:,:,2:]
        s_f = m[:,:,0,1]*x_complex[:,:,0] + m[:,:,0,2]*x_complex[:,:,1]
        s_l = m[:,:,-1,0]*x_complex[:,:,-2] + m[:,:,-1,1]*x_complex[:,:,-1]
        s = torch.cat((s_f.unsqueeze(2), s, s_l.unsqueeze(2)), dim=2)

        if return_analysis:
            analysis = {
                'learned_graph': self.harmonic_graph.visualize_learned_graph(),
                'fusion_weights': {f'band_{k}': self.multi_scale.fusion_weights[f'band_{k}'].detach()
                                 for k in range(30)}
            }
            return s, analysis

        return s


if __name__ == '__main__':
    # Simple test
    print("Testing DB-Transform architecture...")

    # Create model
    model = DB_Transform(num_channel=128, num_heads=4)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

    # Test forward pass
    batch_size = 2
    freq_bins = 257
    time_frames = 100

    # Create dummy complex input
    x = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print("\n✅ DB-Transform test passed!")

    # Test with analysis
    print("\nTesting with analysis...")
    with torch.no_grad():
        output, analysis = model(x, return_analysis=True)

    adj_matrix, harmonic_pairs = analysis['learned_graph']
    print(f"Learned adjacency shape: {adj_matrix.shape}")
    print(f"Discovered harmonic pairs: {len(harmonic_pairs)}")

    print("\n✅ All tests passed!")
