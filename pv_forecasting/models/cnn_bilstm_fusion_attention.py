"""CNN-BiLSTM with Multi-Level Fusion and Attention for PV forecasting.

This module implements an advanced architecture inspired by the MATNet paper
(Multi-Level Fusion Transformer-Based Model for Day-Ahead PV Generation Forecasting).

Key features:
- Multi-Level Fusion: 3 separate input branches (PV history, Weather history, Weather forecast)
- Multi-Head Attention: Self-attention mechanism for temporal dependencies
- Optimized hyperparameters based on MATNet paper analysis
- Mixed precision training for faster GPU computation

Architecture:
    Branch 1: PV History → CNN → BiLSTM → Attention → Features
    Branch 2: Weather History → CNN → BiLSTM → Attention → Features
    Branch 3: Weather Forecast → CNN → Flatten → Dense → Features

    Fusion Level 1: concat(PV_features, Weather_hist_features) → Dense
    Fusion Level 2: concat(Level1, Weather_forecast_features) → Dense → Output

Reference:
    Tortora et al. (2024). "MATNet: Multi-Level Fusion Transformer-Based Model
    for Day-Ahead PV Generation Forecasting". IEEE Transactions on Smart Grid.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_bilstm_fusion_attention(
    pv_seq_len: int,
    weather_seq_len: int,
    weather_forecast_len: int,
    weather_features: int,
    horizon: int,
    embedding_dim: int = 128,
    num_attention_heads: int = 8,
    dropout_rate: float = 0.2,
) -> keras.Model:
    """Build CNN-BiLSTM model with Multi-Level Fusion and Attention.

    Implements a MATNet-inspired architecture that separates inputs into three
    conceptually different branches and fuses them at multiple levels of abstraction.

    Args:
        pv_seq_len: Length of PV history sequence (e.g., 168 for 7 days)
        weather_seq_len: Length of weather history sequence (e.g., 168)
        weather_forecast_len: Length of weather forecast (e.g., 24 for day-ahead)
        weather_features: Number of weather features (e.g., 7: ghi, dni, dhi, temp, etc.)
        horizon: Forecast horizon in hours (e.g., 24)
        embedding_dim: Dimension for embeddings (default: 128, MATNet uses 512)
        num_attention_heads: Number of attention heads (default: 8, as per MATNet)
        dropout_rate: Dropout rate (default: 0.2, optimized from analysis)

    Returns:
        Compiled Keras Model with 3 inputs and 1 output.
            Inputs:
                - pv_history: (batch, pv_seq_len, 1)
                - weather_history: (batch, weather_seq_len, weather_features)
                - weather_forecast: (batch, weather_forecast_len, weather_features)
            Output:
                - predictions: (batch, horizon)

    Example:
        >>> model = build_cnn_bilstm_fusion_attention(
        ...     pv_seq_len=168,
        ...     weather_seq_len=168,
        ...     weather_forecast_len=24,
        ...     weather_features=7,
        ...     horizon=24
        ... )
        >>> model.summary()
    """

    # =========================================================================
    # BRANCH 1: PV PRODUCTION HISTORY
    # =========================================================================
    pv_input = keras.Input(shape=(pv_seq_len, 1), name="pv_history")

    # CNN feature extraction for PV temporal patterns
    pv_x = layers.Conv1D(
        64, kernel_size=5, padding="same", activation="relu", dtype="float16"
    )(pv_input)
    pv_x = layers.BatchNormalization()(pv_x)

    pv_x = layers.Conv1D(
        128, kernel_size=3, padding="same", activation="relu", dtype="float16"
    )(pv_x)
    pv_x = layers.BatchNormalization()(pv_x)

    # MaxPooling reduces sequence length for efficiency
    pv_x = layers.MaxPooling1D(pool_size=2)(pv_x)

    # Stacked BiLSTM with Attention
    pv_x = layers.Bidirectional(
        layers.LSTM(embedding_dim // 2, return_sequences=True, recurrent_dropout=0.05)
    )(pv_x)

    # Multi-Head Self-Attention (MATNet key component)
    pv_attention = layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=embedding_dim // num_attention_heads,
        dropout=dropout_rate / 2,
    )(pv_x, pv_x)

    # Residual connection + Layer Normalization (Transformer-style)
    pv_x = layers.Add()([pv_x, pv_attention])
    pv_x = layers.LayerNormalization()(pv_x)

    # Final BiLSTM to produce single representation
    pv_x = layers.Bidirectional(
        layers.LSTM(embedding_dim // 2, return_sequences=False, recurrent_dropout=0.05)
    )(pv_x)
    pv_features = layers.Dropout(dropout_rate)(pv_x)  # Shape: (batch, embedding_dim)

    # =========================================================================
    # BRANCH 2: WEATHER HISTORY
    # =========================================================================
    weather_hist_input = keras.Input(
        shape=(weather_seq_len, weather_features), name="weather_history"
    )

    # CNN feature extraction for weather patterns
    wh_x = layers.Conv1D(
        64, kernel_size=5, padding="same", activation="relu", dtype="float16"
    )(weather_hist_input)
    wh_x = layers.BatchNormalization()(wh_x)

    wh_x = layers.Conv1D(
        128, kernel_size=3, padding="same", activation="relu", dtype="float16"
    )(wh_x)
    wh_x = layers.BatchNormalization()(wh_x)

    wh_x = layers.MaxPooling1D(pool_size=2)(wh_x)

    # Stacked BiLSTM with Attention
    wh_x = layers.Bidirectional(
        layers.LSTM(embedding_dim // 2, return_sequences=True, recurrent_dropout=0.05)
    )(wh_x)

    # Multi-Head Self-Attention
    wh_attention = layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=embedding_dim // num_attention_heads,
        dropout=dropout_rate / 2,
    )(wh_x, wh_x)

    # Residual + LayerNorm
    wh_x = layers.Add()([wh_x, wh_attention])
    wh_x = layers.LayerNormalization()(wh_x)

    wh_x = layers.Bidirectional(
        layers.LSTM(embedding_dim // 2, return_sequences=False, recurrent_dropout=0.05)
    )(wh_x)
    weather_hist_features = layers.Dropout(dropout_rate)(wh_x)

    # =========================================================================
    # FUSION LEVEL 1: PV HISTORY + WEATHER HISTORY
    # =========================================================================
    # These two are temporally correlated (both observe the past)
    fusion_1 = layers.Concatenate(name="fusion_level_1")(
        [pv_features, weather_hist_features]
    )  # Shape: (batch, 2*embedding_dim)

    fusion_1 = layers.Dense(embedding_dim, activation="relu")(fusion_1)
    fusion_1 = layers.Dropout(dropout_rate)(fusion_1)

    # =========================================================================
    # BRANCH 3: WEATHER FORECAST (FUTURE)
    # =========================================================================
    weather_forecast_input = keras.Input(
        shape=(weather_forecast_len, weather_features), name="weather_forecast"
    )

    # CNN for forecast patterns (no LSTM needed as it's already future-aligned)
    wf_x = layers.Conv1D(
        64, kernel_size=3, padding="same", activation="relu", dtype="float16"
    )(weather_forecast_input)
    wf_x = layers.BatchNormalization()(wf_x)

    wf_x = layers.Conv1D(
        128, kernel_size=3, padding="same", activation="relu", dtype="float16"
    )(wf_x)
    wf_x = layers.BatchNormalization()(wf_x)

    # Flatten and compress
    wf_x = layers.Flatten()(wf_x)
    wf_x = layers.Dense(embedding_dim, activation="relu")(wf_x)
    weather_forecast_features = layers.Dropout(dropout_rate)(wf_x)

    # =========================================================================
    # FUSION LEVEL 2: (PV+WEATHER_HIST) + WEATHER_FORECAST
    # =========================================================================
    # Forecast is conceptually different (observes future), so fused at higher level
    fusion_2 = layers.Concatenate(name="fusion_level_2")(
        [fusion_1, weather_forecast_features]
    )

    fusion_2 = layers.Dense(embedding_dim, activation="relu")(fusion_2)
    fusion_2 = layers.Dropout(dropout_rate)(fusion_2)

    # Additional dense layer for non-linear transformation
    fusion_2 = layers.Dense(embedding_dim // 2, activation="relu")(fusion_2)
    fusion_2 = layers.Dropout(dropout_rate)(fusion_2)

    # =========================================================================
    # OUTPUT LAYER
    # =========================================================================
    # Multi-horizon output (always in float32 for numerical stability)
    output = layers.Dense(horizon, activation="linear", dtype="float32", name="output")(
        fusion_2
    )

    # =========================================================================
    # MODEL COMPILATION
    # =========================================================================
    model = keras.Model(
        inputs=[pv_input, weather_hist_input, weather_forecast_input],
        outputs=output,
        name="CNN_BiLSTM_Fusion_Attention",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )

    return model
