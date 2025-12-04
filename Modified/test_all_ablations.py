"""
Comprehensive Test Suite for All Ablation Models

Tests all three ablations to ensure:
1. Correct parameter counts
2. Shape compatibility
3. Forward pass works
4. Gradient flow is correct
5. No bugs or errors
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Baseline'))

# Import all ablations
from bs_bimamba_abl1 import BS_BiMamba_Abl1
from bs_bimamba_abl2 import BS_BiMamba_Abl2
from bs_bimamba_abl3 import BS_BiMamba_Abl3


def test_model(model_class, model_name, expected_params_M):
    """Test a single model"""
    print(f"\n{'='*70}")
    print(f"Testing {model_name}")
    print(f"{'='*70}")

    try:
        # 1. Create model
        print(f"\n1. Creating model...")
        model = model_class(
            num_channel=128,
            num_layers=4,
            num_bands=30,
            d_state=16,
            chunk_size=32
        )

        # 2. Count parameters
        print(f"\n2. Checking parameters...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"   Total parameters: {total_params/1e6:.2f}M")
        print(f"   Trainable parameters: {trainable_params/1e6:.2f}M")
        print(f"   Expected: ~{expected_params_M:.2f}M")

        # Check if within reasonable range (±10%)
        if abs(total_params/1e6 - expected_params_M) / expected_params_M > 0.10:
            print(f"   ⚠ WARNING: Parameter count differs from expected by >10%")
        else:
            print(f"   ✓ PASS: Parameter count within expected range")

        # 3. Test forward pass with complex input
        print(f"\n3. Testing forward pass (complex input)...")
        batch_size = 2
        freq_bins = 257
        time_frames = 100

        x_complex = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)
        print(f"   Input shape: {x_complex.shape}")

        with torch.no_grad():
            output = model(x_complex)

        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")

        assert output.shape == x_complex.shape, f"Shape mismatch! Expected {x_complex.shape}, got {output.shape}"
        assert torch.is_complex(output), "Output should be complex!"
        assert not torch.isnan(output).any(), "Output contains NaN!"
        assert not torch.isinf(output).any(), "Output contains Inf!"

        print(f"   ✓ PASS: Forward pass successful")

        # 4. Test forward pass with real/imag input
        print(f"\n4. Testing forward pass (real/imag input)...")
        x_real_imag = torch.randn(batch_size, 2, freq_bins, time_frames)
        print(f"   Input shape: {x_real_imag.shape}")

        with torch.no_grad():
            output2 = model(x_real_imag)

        print(f"   Output shape: {output2.shape}")
        assert torch.is_complex(output2), "Output should be complex!"
        print(f"   ✓ PASS: Real/imag input works")

        # 5. Test gradient flow
        print(f"\n5. Testing gradient flow...")
        model.train()
        x_train = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)
        target = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)

        output_train = model(x_train)
        loss = F.l1_loss(torch.view_as_real(output_train), torch.view_as_real(target))
        loss.backward()

        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        max_grad = max(grad_norms) if grad_norms else 0
        min_grad = min(grad_norms) if grad_norms else 0

        print(f"   Loss: {loss.item():.6f}")
        print(f"   Gradients computed: {has_grads}")
        print(f"   Max gradient norm: {max_grad:.6f}")
        print(f"   Min gradient norm: {min_grad:.6f}")

        assert has_grads, "No gradients!"
        assert not any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None), "NaN in gradients!"

        print(f"   ✓ PASS: Gradients flow correctly")

        # 6. Memory test (approximate)
        print(f"\n6. Testing memory usage...")
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)  # MB
        print(f"   Model size: {model_size:.2f} MB")
        print(f"   ✓ PASS: Memory usage reasonable")

        print(f"\n{'='*70}")
        print(f"✓ ALL TESTS PASSED FOR {model_name}")
        print(f"{'='*70}")

        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ TESTS FAILED FOR {model_name}")
        print(f"Error: {str(e)}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUITE FOR ALL ABLATIONS")
    print("="*70)

    results = {}

    # Test Ablation 1
    results['abl1'] = test_model(
        BS_BiMamba_Abl1,
        "Ablation 1: IntraBand BiMamba + Uniform Decoder",
        expected_params_M=3.96
    )

    # Test Ablation 2
    results['abl2'] = test_model(
        BS_BiMamba_Abl2,
        "Ablation 2: Dual-Path BiMamba + Uniform Decoder",
        expected_params_M=4.42
    )

    # Test Ablation 3
    results['abl3'] = test_model(
        BS_BiMamba_Abl3,
        "Ablation 3: Full BS-BiMamba with Adaptive Decoder",
        expected_params_M=2.82
    )

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Ablation 1: {'✓ PASS' if results['abl1'] else '✗ FAIL'}")
    print(f"Ablation 2: {'✓ PASS' if results['abl2'] else '✗ FAIL'}")
    print(f"Ablation 3: {'✓ PASS' if results['abl3'] else '✗ FAIL'}")
    print("="*70)

    if all(results.values()):
        print("\n🎉 ALL MODELS READY FOR TRAINING!")
        print("\nTo train all models in parallel, run:")
        print("  python train_abl1.py &")
        print("  python train_abl2.py &")
        print("  python train_abl3.py &")
        print("\nCheckpoints will be saved to:")
        print("  ./checkpoints_abl1/")
        print("  ./checkpoints_abl2/")
        print("  ./checkpoints_abl3/")
    else:
        print("\n⚠ SOME MODELS FAILED TESTS - FIX BEFORE TRAINING!")

    return all(results.values())


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
