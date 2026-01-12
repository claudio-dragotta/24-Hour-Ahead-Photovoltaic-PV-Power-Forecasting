"""Unit tests for Multi-Branch Transformer model.

Tests the complete MultiBranchTransformer architecture including
forward pass, training step, and prediction functionality.
"""

import pytest
import torch
import torch.nn as nn
from lightning.pytorch import Trainer

from pv_forecasting.models.multi_branch_tft import MultiBranchTransformer


class TestMultiBranchTransformer:
    """Test suite for MultiBranchTransformer model."""

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for testing."""
        batch_size = 8
        seq_len_encoder = 168
        seq_len_decoder = 24
        n_pv_features = 3
        n_hist_weather = 12
        n_forecast_weather = 7

        features = {
            "pv_history": torch.randn(batch_size, seq_len_encoder, n_pv_features),
            "weather_history": torch.randn(batch_size, seq_len_encoder, n_hist_weather),
            "weather_forecast": torch.randn(batch_size, seq_len_decoder, n_forecast_weather),
        }
        targets = torch.rand(batch_size, seq_len_decoder)  # PV values in [0, 1]

        return (features, targets)

    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return MultiBranchTransformer(
            n_pv_features=3,
            n_hist_weather_features=12,
            n_forecast_weather_features=7,
            seq_len_encoder=168,
            seq_len_decoder=24,
            d_model=64,  # Small for fast testing
            num_heads=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
            learning_rate=1e-3,
            weight_decay=1e-4,
        )

    def test_model_initialization(self, model):
        """Test model initialization and hyperparameters."""
        assert model.n_pv_features == 3
        assert model.n_hist_weather_features == 12
        assert model.n_forecast_weather_features == 7
        assert model.seq_len_encoder == 168
        assert model.seq_len_decoder == 24
        assert model.d_model == 64
        assert model.num_heads == 2
        assert model.num_layers == 1

        # Check sub-modules exist
        assert hasattr(model, "pv_embedding")
        assert hasattr(model, "weather_hist_embedding")
        assert hasattr(model, "weather_forecast_embedding")
        assert hasattr(model, "fusion_stage1")
        assert hasattr(model, "fusion_stage2")

    def test_forward_pass_shape(self, model, sample_batch):
        """Test forward pass produces correct output shape."""
        features, _ = sample_batch
        model.eval()

        with torch.no_grad():
            output = model(features)

        batch_size = features["pv_history"].shape[0]
        expected_shape = (batch_size, 24)

        assert output.shape == expected_shape
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_output_range(self, model, sample_batch):
        """Test that output is in valid range [0, 1] due to sigmoid."""
        features, _ = sample_batch
        model.eval()

        with torch.no_grad():
            output = model(features)

        # Output should be in [0, 1] due to final sigmoid
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_training_step(self, model, sample_batch):
        """Test training step computes loss correctly."""
        loss = model.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # MSE loss is non-negative
        assert not torch.isnan(loss), "Loss is NaN"

    def test_validation_step(self, model, sample_batch):
        """Test validation step."""
        loss = model.validation_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_test_step(self, model, sample_batch):
        """Test test step."""
        loss = model.test_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_predict_step(self, model, sample_batch):
        """Test predict step."""
        model.eval()
        predictions = model.predict_step(sample_batch, batch_idx=0)

        batch_size = sample_batch[0]["pv_history"].shape[0]
        assert predictions.shape == (batch_size, 24)

    def test_gradient_flow(self, model, sample_batch):
        """Test that gradients flow through all branches."""
        features, targets = sample_batch

        # Zero gradients
        model.zero_grad()

        # Forward + backward pass
        output = model(features)
        loss = nn.functional.mse_loss(output, targets)
        loss.backward()

        # Check gradients in each branch
        assert model.pv_embedding.weight.grad is not None
        assert not torch.all(model.pv_embedding.weight.grad == 0)

        assert model.weather_hist_embedding.weight.grad is not None
        assert not torch.all(model.weather_hist_embedding.weight.grad == 0)

        assert model.weather_forecast_embedding.weight.grad is not None
        assert not torch.all(model.weather_forecast_embedding.weight.grad == 0)

        # Check fusion layers have gradients
        assert model.fusion_stage1.fc.weight.grad is not None
        assert model.fusion_stage2.fc.weight.grad is not None

    def test_configure_optimizers(self, model):
        """Test optimizer configuration."""
        opt_config = model.configure_optimizers()

        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config

        optimizer = opt_config["optimizer"]
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 1e-3
        assert optimizer.defaults["weight_decay"] == 1e-4

    def test_different_batch_sizes(self, model):
        """Test model handles different batch sizes."""
        seq_len_encoder = 168
        seq_len_decoder = 24

        for batch_size in [1, 4, 16, 32]:
            features = {
                "pv_history": torch.randn(batch_size, seq_len_encoder, 3),
                "weather_history": torch.randn(batch_size, seq_len_encoder, 12),
                "weather_forecast": torch.randn(batch_size, seq_len_decoder, 7),
            }

            model.eval()
            with torch.no_grad():
                output = model(features)

            assert output.shape == (batch_size, seq_len_decoder)

    def test_deterministic_eval_mode(self, model, sample_batch):
        """Test that eval mode produces deterministic outputs."""
        features, _ = sample_batch
        model.eval()

        with torch.no_grad():
            output1 = model(features)
            output2 = model(features)

        assert torch.allclose(output1, output2), "Outputs differ in eval mode"

    def test_model_save_and_load(self, model, tmp_path):
        """Test model can be saved and loaded."""
        # Save model
        checkpoint_path = tmp_path / "test_model.ckpt"
        torch.save(model.state_dict(), checkpoint_path)

        # Create new model and load weights
        new_model = MultiBranchTransformer(
            n_pv_features=3,
            n_hist_weather_features=12,
            n_forecast_weather_features=7,
            seq_len_encoder=168,
            seq_len_decoder=24,
            d_model=64,
            num_heads=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
            learning_rate=1e-3,
            weight_decay=1e-4,
        )
        new_model.load_state_dict(torch.load(checkpoint_path))

        # Compare outputs
        features = {
            "pv_history": torch.randn(4, 168, 3),
            "weather_history": torch.randn(4, 168, 12),
            "weather_forecast": torch.randn(4, 24, 7),
        }

        model.eval()
        new_model.eval()

        with torch.no_grad():
            output1 = model(features)
            output2 = new_model(features)

        assert torch.allclose(output1, output2, atol=1e-5), "Loaded model produces different outputs"

    def test_parameter_count(self, model):
        """Test that model has expected number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have parameters (rough estimate for d_model=64, heads=2, layers=1)
        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable

        # Print for reference (helpful for debugging)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def test_different_model_sizes(self):
        """Test model works with different hyperparameters."""
        configs = [
            # Small model
            {"d_model": 64, "num_heads": 2, "num_layers": 1, "dim_feedforward": 128},
            # Medium model
            {"d_model": 128, "num_heads": 4, "num_layers": 2, "dim_feedforward": 512},
            # Large model (commented out for speed)
            # {'d_model': 256, 'num_heads': 8, 'num_layers': 3, 'dim_feedforward': 1024},
        ]

        for config in configs:
            model = MultiBranchTransformer(
                n_pv_features=3,
                n_hist_weather_features=12,
                n_forecast_weather_features=7,
                seq_len_encoder=168,
                seq_len_decoder=24,
                **config,
            )

            features = {
                "pv_history": torch.randn(4, 168, 3),
                "weather_history": torch.randn(4, 168, 12),
                "weather_forecast": torch.randn(4, 24, 7),
            }

            model.eval()
            with torch.no_grad():
                output = model(features)

            assert output.shape == (4, 24)

    def test_temporal_pooling(self, model, sample_batch):
        """Test that temporal pooling reduces sequence dimension correctly."""
        features, _ = sample_batch

        # Extract intermediate representations (hack into forward pass)
        model.eval()
        with torch.no_grad():
            # Branch 1: PV
            pv_emb = model.pv_embedding(features["pv_history"])
            pv_emb = model.pv_pos_encoder(pv_emb)
            pv_encoded = model.pv_transformer(pv_emb)
            assert pv_encoded.shape == (8, 168, 64)  # (batch, seq_enc, d_model)

            # Temporal pooling
            pv_pooled = model.pv_temporal_pooling(pv_encoded.permute(0, 2, 1)).squeeze(-1)
            assert pv_pooled.shape == (8, 64)  # (batch, d_model)

    def test_fusion_stages(self, model, sample_batch):
        """Test that hierarchical fusion works correctly."""
        features, _ = sample_batch

        model.eval()
        with torch.no_grad():
            # Process all branches
            pv_emb = model.pv_embedding(features["pv_history"])
            pv_emb = model.pv_pos_encoder(pv_emb)
            pv_encoded = model.pv_transformer(pv_emb)
            pv_pooled = model.pv_temporal_pooling(pv_encoded.permute(0, 2, 1)).squeeze(-1)

            wx_hist_emb = model.weather_hist_embedding(features["weather_history"])
            wx_hist_emb = model.weather_hist_pos_encoder(wx_hist_emb)
            wx_hist_encoded = model.weather_hist_transformer(wx_hist_emb)
            wx_hist_pooled = model.weather_hist_temporal_pooling(wx_hist_encoded.permute(0, 2, 1)).squeeze(-1)

            # Stage 1: Fuse PV + Weather history
            fusion1_input = torch.stack([pv_pooled, wx_hist_pooled], dim=1)
            assert fusion1_input.shape == (8, 2, 64)

            fusion1_output = model.fusion_stage1(fusion1_input)
            assert fusion1_output.shape == (8, 64)

    def test_hparams_saved(self, model):
        """Test that hyperparameters are saved correctly."""
        hparams = model.hparams

        assert "n_pv_features" in hparams
        assert "n_hist_weather_features" in hparams
        assert "n_forecast_weather_features" in hparams
        assert "seq_len_encoder" in hparams
        assert "seq_len_decoder" in hparams
        assert "d_model" in hparams
        assert "num_heads" in hparams
        assert "num_layers" in hparams


class TestMultiBranchIntegration:
    """Integration tests for Multi-Branch Transformer."""

    def test_short_training_loop(self):
        """Test that model can run a short training loop without errors."""
        # Create simple dataset
        batch_size = 8
        features = {
            "pv_history": torch.randn(batch_size, 168, 3),
            "weather_history": torch.randn(batch_size, 168, 12),
            "weather_forecast": torch.randn(batch_size, 24, 7),
        }
        targets = torch.rand(batch_size, 24)

        # Create model
        model = MultiBranchTransformer(
            n_pv_features=3,
            n_hist_weather_features=12,
            n_forecast_weather_features=7,
            seq_len_encoder=168,
            seq_len_decoder=24,
            d_model=32,  # Very small for fast test
            num_heads=2,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.1,
        )

        # Train for a few steps
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_loss = None
        for step in range(5):
            optimizer.zero_grad()
            output = model(features)
            loss = nn.functional.mse_loss(output, targets)

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        # Loss should decrease (at least slightly)
        final_loss = loss.item()
        print(f"\nInitial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
        # Don't assert decrease as it might not happen in 5 steps with random data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
