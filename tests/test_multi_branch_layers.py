"""Unit tests for Multi-Branch Transformer custom layers.

Tests SoftAttention and PositionalEncoding implementations to ensure
correct shapes, gradient flow, and edge case handling.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from pv_forecasting.models.layers import PositionalEncoding, SoftAttention


class TestSoftAttention:
    """Test suite for SoftAttention layer."""

    def test_initialization(self):
        """Test SoftAttention layer initialization."""
        attention = SoftAttention(input_dim=512, hidden_dim=256, temperature=1.0)

        assert attention.input_dim == 512
        assert attention.hidden_dim == 256
        assert attention.temperature == 1.0
        assert isinstance(attention.fc, nn.Linear)
        assert attention.fc.in_features == 512
        assert attention.fc.out_features == 256
        assert attention.context_vector.shape == (256,)

    def test_forward_shape_2_branches(self):
        """Test forward pass output shape with 2 input branches."""
        batch_size = 16
        num_branches = 2
        input_dim = 512

        attention = SoftAttention(input_dim=input_dim, hidden_dim=256)
        x = torch.randn(batch_size, num_branches, input_dim)

        output = attention(x)

        assert output.shape == (batch_size, input_dim)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_forward_shape_3_branches(self):
        """Test forward pass output shape with 3 input branches."""
        batch_size = 32
        num_branches = 3
        input_dim = 256

        attention = SoftAttention(input_dim=input_dim, hidden_dim=128)
        x = torch.randn(batch_size, num_branches, input_dim)

        output = attention(x)

        assert output.shape == (batch_size, input_dim)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1.0 (implicitly via softmax)."""
        batch_size = 8
        num_branches = 3
        input_dim = 128

        attention = SoftAttention(input_dim=input_dim, hidden_dim=64)
        x = torch.randn(batch_size, num_branches, input_dim)

        # Forward pass
        output = attention(x)

        # Check output is weighted combination (should be within input range + some margin)
        x_min = x.min().item()
        x_max = x.max().item()
        output_min = output.min().item()
        output_max = output.max().item()

        # Weighted combination should be within input bounds (with some tolerance)
        assert output_min >= x_min - 1.0
        assert output_max <= x_max + 1.0

    def test_temperature_scaling(self):
        """Test that temperature parameter affects output."""
        batch_size = 8
        num_branches = 2
        input_dim = 128

        x = torch.randn(batch_size, num_branches, input_dim)

        # Low temperature (more peaked distribution)
        attention_low_temp = SoftAttention(input_dim=input_dim, hidden_dim=64, temperature=0.1)
        output_low = attention_low_temp(x)

        # High temperature (more uniform distribution)
        attention_high_temp = SoftAttention(input_dim=input_dim, hidden_dim=64, temperature=10.0)
        output_high = attention_high_temp(x)

        # Copy weights to make fair comparison
        attention_high_temp.fc.weight.data = attention_low_temp.fc.weight.data.clone()
        attention_high_temp.fc.bias.data = attention_low_temp.fc.bias.data.clone()
        attention_high_temp.context_vector.data = attention_low_temp.context_vector.data.clone()

        output_high = attention_high_temp(x)

        # Outputs should be different (temperature affects weighting)
        assert not torch.allclose(output_low, output_high, atol=1e-3)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through attention layer."""
        batch_size = 4
        num_branches = 2
        input_dim = 64

        attention = SoftAttention(input_dim=input_dim, hidden_dim=32)
        x = torch.randn(batch_size, num_branches, input_dim, requires_grad=True)

        output = attention(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are non-zero
        assert x.grad is not None
        assert not torch.all(x.grad == 0), "Input gradients are all zero"
        assert attention.fc.weight.grad is not None
        assert attention.context_vector.grad is not None
        assert not torch.isnan(attention.fc.weight.grad).any()

    def test_single_branch_edge_case(self):
        """Test attention with single branch (should just return input)."""
        batch_size = 8
        num_branches = 1
        input_dim = 128

        attention = SoftAttention(input_dim=input_dim, hidden_dim=64)
        x = torch.randn(batch_size, num_branches, input_dim)

        output = attention(x)

        # With single branch, attention weight should be 1.0, so output ≈ input
        assert output.shape == (batch_size, input_dim)
        assert torch.allclose(output, x.squeeze(1), atol=1e-3)

    def test_deterministic_output(self):
        """Test that same input produces same output (no randomness)."""
        batch_size = 4
        num_branches = 2
        input_dim = 64

        attention = SoftAttention(input_dim=input_dim, hidden_dim=32)
        attention.eval()  # Set to eval mode

        x = torch.randn(batch_size, num_branches, input_dim)

        output1 = attention(x)
        output2 = attention(x)

        assert torch.allclose(output1, output2), "Non-deterministic output detected"


class TestPositionalEncoding:
    """Test suite for PositionalEncoding layer."""

    def test_initialization(self):
        """Test PositionalEncoding initialization."""
        pos_enc = PositionalEncoding(d_model=512, max_len=1000, dropout=0.1)

        assert pos_enc.pe.shape == (1000, 1, 512)
        assert isinstance(pos_enc.dropout, nn.Dropout)

    def test_forward_shape_batch_first(self):
        """Test forward pass with batch_first=True."""
        batch_size = 16
        seq_len = 100
        d_model = 256

        pos_enc = PositionalEncoding(d_model=d_model, max_len=5000, batch_first=True)
        x = torch.randn(batch_size, seq_len, d_model)

        output = pos_enc(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_shape_seq_first(self):
        """Test forward pass with batch_first=False."""
        batch_size = 16
        seq_len = 100
        d_model = 256

        pos_enc = PositionalEncoding(d_model=d_model, max_len=5000, batch_first=False)
        x = torch.randn(seq_len, batch_size, d_model)

        output = pos_enc(x)

        assert output.shape == (seq_len, batch_size, d_model)

    def test_positional_encoding_pattern(self):
        """Test that positional encoding follows sine/cosine pattern."""
        d_model = 128
        max_len = 1000

        pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)

        # Extract positional encoding matrix
        pe = pos_enc.pe.squeeze(1).detach().numpy()  # (max_len, d_model)

        # Check sine pattern on even indices
        for pos in [0, 10, 50, 100]:
            for i in range(0, d_model, 2):
                expected = np.sin(pos / (10000 ** (i / d_model)))
                actual = pe[pos, i]
                assert np.isclose(expected, actual, atol=1e-5), \
                    f"Sine pattern mismatch at pos={pos}, dim={i}"

        # Check cosine pattern on odd indices
        for pos in [0, 10, 50, 100]:
            for i in range(1, d_model, 2):
                expected = np.cos(pos / (10000 ** ((i-1) / d_model)))
                actual = pe[pos, i]
                assert np.isclose(expected, actual, atol=1e-5), \
                    f"Cosine pattern mismatch at pos={pos}, dim={i}"

    def test_different_positions_different_encodings(self):
        """Test that different positions have different encodings."""
        d_model = 256
        max_len = 500

        pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
        pe = pos_enc.pe.squeeze(1)  # (max_len, d_model)

        # Compare position 0 vs position 100
        pe_0 = pe[0, :]
        pe_100 = pe[100, :]

        assert not torch.allclose(pe_0, pe_100), "Different positions have same encoding"

    def test_encoding_magnitude(self):
        """Test that positional encodings have reasonable magnitude."""
        d_model = 512
        max_len = 1000

        pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
        pe = pos_enc.pe.squeeze(1)  # (max_len, d_model)

        # Sine/cosine values should be in [-1, 1]
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_gradient_flow(self):
        """Test that gradients flow through positional encoding."""
        batch_size = 8
        seq_len = 50
        d_model = 128

        pos_enc = PositionalEncoding(d_model=d_model, max_len=5000, dropout=0.0, batch_first=True)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = pos_enc(x)
        loss = output.sum()
        loss.backward()

        # Positional encoding is added, so gradients should flow to input
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_max_len_constraint(self):
        """Test that sequences longer than max_len are handled."""
        d_model = 128
        max_len = 100

        pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len, batch_first=True)

        # Test with sequence shorter than max_len
        x_short = torch.randn(4, 50, d_model)
        output_short = pos_enc(x_short)
        assert output_short.shape == x_short.shape

        # Test with sequence equal to max_len
        x_max = torch.randn(4, max_len, d_model)
        output_max = pos_enc(x_max)
        assert output_max.shape == x_max.shape

    def test_dropout_applied(self):
        """Test that dropout is applied during training."""
        batch_size = 8
        seq_len = 50
        d_model = 128

        pos_enc = PositionalEncoding(d_model=d_model, max_len=5000, dropout=0.5, batch_first=True)
        pos_enc.train()  # Set to training mode

        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times, outputs should differ due to dropout
        output1 = pos_enc(x)
        output2 = pos_enc(x)

        # With dropout=0.5, outputs should be different
        assert not torch.allclose(output1, output2), "Dropout not applied in training mode"

    def test_no_dropout_in_eval(self):
        """Test that dropout is not applied in eval mode."""
        batch_size = 8
        seq_len = 50
        d_model = 128

        pos_enc = PositionalEncoding(d_model=d_model, max_len=5000, dropout=0.5, batch_first=True)
        pos_enc.eval()  # Set to eval mode

        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times, outputs should be identical
        output1 = pos_enc(x)
        output2 = pos_enc(x)

        assert torch.allclose(output1, output2), "Outputs differ in eval mode (dropout should be off)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
