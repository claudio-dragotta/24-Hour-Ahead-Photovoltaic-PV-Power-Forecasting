"""Multi-Branch Transformer for PV Power Forecasting.

This module implements an advanced transformer architecture with separate
processing branches for PV history, weather history, and future weather forecasts.
The architecture uses hierarchical soft attention fusion to combine information
from different data sources adaptively.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule

from pv_forecasting.models.layers import PositionalEncoding, SoftAttention


class MultiBranchTransformer(LightningModule):
    """Multi-Branch Transformer with hierarchical attention fusion.

    Architecture Design Rationale:
    - **Separate branch processing**: PV production, historical weather, and
      forecast weather have fundamentally different characteristics and should be
      processed independently before fusion.
    - **Hierarchical fusion**: First combine PV + historical weather (both look
      backward), then integrate forward-looking weather forecasts.
    - **Adaptive attention weights**: Unlike fixed ensemble weights, soft attention
      learns to weight branches dynamically based on input conditions.

    Args:
        n_pv_features: Number of PV-related lag features (e.g., pv_lag1, pv_lag24).
        n_hist_weather_features: Number of historical weather lag features.
        n_forecast_weather_features: Number of future weather covariates.
        seq_len_encoder: Length of historical context window (hours).
        seq_len_decoder: Length of forecast horizon (default: 24 hours).
        d_model: Model hidden dimension (default: 512).
        num_heads: Number of attention heads in transformers (default: 8).
        num_layers: Number of transformer encoder layers per branch (default: 3).
        dim_feedforward: Feedforward network dimension (default: 2048).
        dropout: Dropout rate (default: 0.1).
        learning_rate: Optimizer learning rate (default: 1e-3).
        weight_decay: L2 regularization factor (default: 1e-4).

    Example:
        >>> model = MultiBranchTransformer(
        ...     n_pv_features=3,  # pv_lag1, pv_lag24, pv_lag168
        ...     n_hist_weather_features=12,  # ghi_lag1, ghi_lag24, etc.
        ...     n_forecast_weather_features=7,  # temp, humidity, ghi, etc.
        ...     seq_len_encoder=168,
        ...     seq_len_decoder=24,
        ...     d_model=256,
        ...     num_heads=4,
        ...     num_layers=2
        ... )
        >>> batch = {
        ...     'pv_history': torch.randn(16, 168, 3),
        ...     'weather_history': torch.randn(16, 168, 12),
        ...     'weather_forecast': torch.randn(16, 24, 7)
        ... }
        >>> output = model(batch)
        >>> output.shape
        torch.Size([16, 24])  # (batch_size, forecast_hours)
    """

    def __init__(
        self,
        n_pv_features: int,
        n_hist_weather_features: int,
        n_forecast_weather_features: int,
        seq_len_encoder: int,
        seq_len_decoder: int = 24,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_pv_features = n_pv_features
        self.n_hist_weather_features = n_hist_weather_features
        self.n_forecast_weather_features = n_forecast_weather_features
        self.seq_len_encoder = seq_len_encoder
        self.seq_len_decoder = seq_len_decoder
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Branch 1: PV Production History
        # Processes historical PV lag features (e.g., lag1, lag24, lag168)
        self.pv_embedding = nn.Linear(n_pv_features, d_model)
        self.pv_pos_encoder = PositionalEncoding(d_model, max_len=seq_len_encoder, dropout=dropout)
        pv_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-normalization for better stability
        )
        self.pv_transformer = nn.TransformerEncoder(pv_encoder_layer, num_layers=num_layers)

        # Branch 2: Historical Weather
        # Processes historical weather lag features (e.g., ghi_lag1, temp_lag24)
        self.weather_hist_embedding = nn.Linear(n_hist_weather_features, d_model)
        self.weather_hist_pos_encoder = PositionalEncoding(d_model, max_len=seq_len_encoder, dropout=dropout)
        weather_hist_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.weather_hist_transformer = nn.TransformerEncoder(weather_hist_encoder_layer, num_layers=num_layers)

        # Branch 3: Future Weather Forecast
        # Processes day-ahead weather predictions (e.g., NWP forecasts)
        self.weather_forecast_embedding = nn.Linear(n_forecast_weather_features, d_model)
        self.weather_forecast_pos_encoder = PositionalEncoding(d_model, max_len=seq_len_decoder, dropout=dropout)
        weather_forecast_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.weather_forecast_transformer = nn.TransformerEncoder(weather_forecast_encoder_layer, num_layers=num_layers)

        # Temporal pooling: sequence -> single representation
        # Uses learnable linear projection instead of fixed global average pooling
        self.pv_temporal_pooling = nn.Linear(seq_len_encoder, 1)
        self.weather_hist_temporal_pooling = nn.Linear(seq_len_encoder, 1)
        self.weather_forecast_temporal_pooling = nn.Linear(seq_len_decoder, 1)

        # Hierarchical Fusion with Soft Attention
        # Stage 1: Fuse PV history + Weather history (both backward-looking)
        self.fusion_stage1 = SoftAttention(input_dim=d_model, hidden_dim=d_model, temperature=1.0)
        self.fc_stage1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        # Stage 2: Fuse (PV + Weather history) + Future weather forecast
        self.fusion_stage2 = SoftAttention(input_dim=d_model, hidden_dim=d_model, temperature=1.0)
        self.fc_output = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, seq_len_decoder)  # 24-hour forecast (no activation for normalized targets)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through multi-branch architecture.

        Args:
            batch: Dictionary containing:
                - 'pv_history': (batch_size, seq_len_encoder, n_pv_features)
                - 'weather_history': (batch_size, seq_len_encoder, n_hist_weather_features)
                - 'weather_forecast': (batch_size, seq_len_decoder, n_forecast_weather_features)

        Returns:
            Tensor of shape (batch_size, seq_len_decoder) with PV power predictions.
        """
        pv_hist = batch['pv_history']  # (B, T_enc, F_pv)
        weather_hist = batch['weather_history']  # (B, T_enc, F_wx_hist)
        weather_fcst = batch['weather_forecast']  # (B, T_dec, F_wx_fcst)

        # Branch 1: Process PV history
        pv_emb = self.pv_embedding(pv_hist)  # (B, T_enc, d_model)
        pv_emb = self.pv_pos_encoder(pv_emb)
        pv_encoded = self.pv_transformer(pv_emb)  # (B, T_enc, d_model)

        # Branch 2: Process weather history
        wx_hist_emb = self.weather_hist_embedding(weather_hist)  # (B, T_enc, d_model)
        wx_hist_emb = self.weather_hist_pos_encoder(wx_hist_emb)
        wx_hist_encoded = self.weather_hist_transformer(wx_hist_emb)  # (B, T_enc, d_model)

        # Branch 3: Process future weather forecast
        wx_fcst_emb = self.weather_forecast_embedding(weather_fcst)  # (B, T_dec, d_model)
        wx_fcst_emb = self.weather_forecast_pos_encoder(wx_fcst_emb)
        wx_fcst_encoded = self.weather_forecast_transformer(wx_fcst_emb)  # (B, T_dec, d_model)

        # Temporal pooling: reduce sequence dimension to single vector
        # pv_encoded: (B, T_enc, d_model) -> (B, d_model, T_enc) -> (B, d_model, 1) -> (B, d_model)
        pv_pooled = self.pv_temporal_pooling(pv_encoded.permute(0, 2, 1)).squeeze(-1)
        wx_hist_pooled = self.weather_hist_temporal_pooling(wx_hist_encoded.permute(0, 2, 1)).squeeze(-1)
        wx_fcst_pooled = self.weather_forecast_temporal_pooling(wx_fcst_encoded.permute(0, 2, 1)).squeeze(-1)

        # Stage 1: Fuse PV + historical weather (both backward-looking signals)
        fusion1_input = torch.stack([pv_pooled, wx_hist_pooled], dim=1)  # (B, 2, d_model)
        fusion1_output = self.fusion_stage1(fusion1_input)  # (B, d_model)
        fusion1_output = self.fc_stage1(fusion1_output)  # (B, d_model)

        # Stage 2: Fuse (PV + Weather history) + Future weather forecast
        fusion2_input = torch.stack([fusion1_output, wx_fcst_pooled], dim=1)  # (B, 2, d_model)
        fusion2_output = self.fusion_stage2(fusion2_input)  # (B, d_model)

        # Final output: 24-hour forecast
        output = self.fc_output(fusion2_output)  # (B, 24)

        return output

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with MSE loss.

        Args:
            batch: Tuple of (features_dict, targets)
                - features_dict: Dictionary with 'pv_history', 'weather_history', 'weather_forecast'
                - targets: Ground truth PV values, shape (batch_size, 24)
            batch_idx: Batch index.

        Returns:
            Training loss value.
        """
        features, targets = batch
        predictions = self(features)
        loss = nn.functional.mse_loss(predictions, targets)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Tuple of (features_dict, targets).
            batch_idx: Batch index.

        Returns:
            Validation loss value.
        """
        features, targets = batch
        predictions = self(features)
        loss = nn.functional.mse_loss(predictions, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Test step.

        Args:
            batch: Tuple of (features_dict, targets).
            batch_idx: Batch index.

        Returns:
            Test loss value.
        """
        features, targets = batch
        predictions = self(features)
        loss = nn.functional.mse_loss(predictions, targets)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure Adam optimizer with ReduceLROnPlateau scheduler.

        Returns:
            Optimizer and scheduler configuration dictionary.
        """
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=10,
            min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def predict_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Prediction step for inference.

        Args:
            batch: Tuple of (features_dict, targets) or just features_dict.
            batch_idx: Batch index.

        Returns:
            Predictions tensor of shape (batch_size, 24).
        """
        if isinstance(batch, tuple):
            features, _ = batch
        else:
            features = batch
        return self(features)
