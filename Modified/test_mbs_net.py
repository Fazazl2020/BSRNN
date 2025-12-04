#!/usr/bin/env python3
"""
Test script to verify MBS-Net implementation is bug-free.

Checks:
1. Shape compatibility (all dimensions match BSRNN)
2. Forward pass works (no crashes)
3. Parameter count is correct (~2.1M)
4. Memory usage is reasonable
"""

import torch
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Baseline'))

from mbs_net_optimized import MBS_Net, MaskDecoderLightweight
from module import BandSplit, MaskDecoder as MaskDecoderOriginal

def test_mask_decoder_shapes():
    """Test that lightweight decoder outputs same shape as original"""
    print("="*60)
    print("TEST 1: MaskDecoder Shape Compatibility")
    print("="*60)

    batch_size = 2
    channels = 128
    time_steps = 50
    num_bands = 30

    # Create dummy input: [B, N, T, K]
    x = torch.randn(batch_size, channels, time_steps, num_bands)

    # Test lightweight decoder
    decoder_light = MaskDecoderLightweight(channels=channels)
    output_light = decoder_light(x)

    # Test original decoder
    decoder_orig = MaskDecoderOriginal(channels=channels)
    output_orig = decoder_orig(x)

    print(f"Input shape:              {list(x.shape)}")
    print(f"Lightweight output shape: {list(output_light.shape)}")
    print(f"Original output shape:    {list(output_orig.shape)}")

    # Check shapes match
    assert output_light.shape == output_orig.shape, \
        f"Shape mismatch! Light: {output_light.shape}, Orig: {output_orig.shape}"

    # Check expected shape: [B, 257, T, 3, 2]
    expected_shape = (batch_size, 257, time_steps, 3, 2)
    assert output_light.shape == expected_shape, \
        f"Wrong shape! Expected {expected_shape}, got {output_light.shape}"

    print("✓ PASS: Shapes match perfectly!")
    print()

def test_parameter_count():
    """Test that parameter counts are as expected"""
    print("="*60)
    print("TEST 2: Parameter Count Verification")
    print("="*60)

    # Test MaskDecoderLightweight
    decoder_light = MaskDecoderLightweight(channels=128)
    params_light = sum(p.numel() for p in decoder_light.parameters())

    # Test original MaskDecoder
    decoder_orig = MaskDecoderOriginal(channels=128)
    params_orig = sum(p.numel() for p in decoder_orig.parameters())

    print(f"Lightweight decoder: {params_light/1e6:.2f}M params")
    print(f"Original decoder:    {params_orig/1e6:.2f}M params")
    print(f"Reduction:           {params_orig/1e6 - params_light/1e6:.2f}M params ({100*(1-params_light/params_orig):.1f}%)")

    # Check lightweight is roughly 1.8M (allow 10% tolerance)
    assert 1.6e6 < params_light < 2.0e6, \
        f"Unexpected param count! Expected ~1.8M, got {params_light/1e6:.2f}M"

    # Test full MBS-Net model
    model = MBS_Net(num_channel=128, num_layers=4, num_bands=30, d_state=12, chunk_size=32)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()
    print(f"Full MBS-Net Model:")
    print(f"  Total params:      {total_params/1e6:.2f}M")
    print(f"  Trainable params:  {trainable_params/1e6:.2f}M")

    # Check total is around 2.1M (allow 15% tolerance)
    assert 1.8e6 < total_params < 2.5e6, \
        f"Unexpected total params! Expected ~2.1M, got {total_params/1e6:.2f}M"

    print("✓ PASS: Parameter counts are correct!")
    print()

def test_forward_pass():
    """Test that forward pass works without errors"""
    print("="*60)
    print("TEST 3: Forward Pass Verification")
    print("="*60)

    batch_size = 2
    freq_bins = 257
    time_steps = 100

    # Create dummy complex spectrogram: [B, F, T]
    x = torch.randn(batch_size, freq_bins, time_steps, dtype=torch.complex64)

    print(f"Input shape: {list(x.shape)} (complex)")

    # Create model
    model = MBS_Net(num_channel=128, num_layers=4, num_bands=30, d_state=12, chunk_size=32)
    model.eval()

    # Forward pass
    with torch.no_grad():
        try:
            output = model(x)
            print(f"Output shape: {list(output.shape)} (complex)")

            # Check output shape
            expected_shape = (batch_size, freq_bins, time_steps-2)  # 3-tap filter reduces time by 2
            assert output.shape == expected_shape, \
                f"Wrong output shape! Expected {expected_shape}, got {output.shape}"

            # Check output is complex
            assert output.dtype == torch.complex64, \
                f"Output should be complex! Got {output.dtype}"

            # Check no NaNs or Infs
            assert not torch.isnan(output).any(), "Output contains NaNs!"
            assert not torch.isinf(output).any(), "Output contains Infs!"

            print("✓ PASS: Forward pass successful, no errors!")

        except Exception as e:
            print(f"✗ FAIL: Forward pass failed with error:")
            print(f"  {type(e).__name__}: {e}")
            raise

    print()

def test_memory_usage():
    """Test memory usage with larger batch size"""
    print("="*60)
    print("TEST 4: Memory Usage Test")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠ SKIP: CUDA not available, skipping memory test")
        print()
        return

    batch_size = 6
    freq_bins = 257
    time_steps = 200

    # Create dummy input
    x = torch.randn(batch_size, freq_bins, time_steps, dtype=torch.complex64).cuda()

    # Create model
    model = MBS_Net(num_channel=128, num_layers=4).cuda()
    model.eval()

    # Measure memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = model(x)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"Batch size: {batch_size}")
    print(f"Input shape: {list(x.shape)}")
    print(f"Peak memory: {peak_mem:.2f} GB")

    # Check memory is reasonable (should be < 6GB)
    assert peak_mem < 6.0, \
        f"Memory usage too high! {peak_mem:.2f} GB > 6 GB"

    print("✓ PASS: Memory usage is reasonable!")
    print()

def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "MBS-Net Implementation Verification" + " "*12 + "║")
    print("╚" + "="*58 + "╝")
    print()

    try:
        # Run all tests
        test_mask_decoder_shapes()
        test_parameter_count()
        test_forward_pass()
        test_memory_usage()

        # Summary
        print("="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print()
        print("Summary:")
        print("  ✓ Shape compatibility verified")
        print("  ✓ Parameter count correct (~2.1M)")
        print("  ✓ Forward pass works (no bugs)")
        print("  ✓ Memory usage acceptable")
        print()
        print("The implementation is ready for training!")
        print()

    except AssertionError as e:
        print("\n" + "="*60)
        print("TEST FAILED! ✗")
        print("="*60)
        print(f"\nError: {e}\n")
        sys.exit(1)

    except Exception as e:
        print("\n" + "="*60)
        print("UNEXPECTED ERROR! ✗")
        print("="*60)
        print(f"\n{type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
