"""CNN-BiLSTM model architecture for multi-horizon PV power forecasting.

This module defines the deep learning model used for 24-hour ahead photovoltaic
power forecasting. The architecture combines:
- 1D Convolutional layers for feature extraction from time series
- Bidirectional LSTM for capturing temporal dependencies in both directions
- Dense output layer for multi-horizon forecasting

The model is designed to forecast multiple future timesteps simultaneously
(direct multi-horizon strategy) rather than iterative one-step-ahead prediction.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_bilstm(input_shape: Tuple[int, int], horizon: int) -> keras.Model:
    """Build CNN-BiLSTM model for multi-horizon time series forecasting.

    Creates a compiled Keras model that combines Convolutional Neural Networks
    and Bidirectional Long Short-Term Memory networks for forecasting. The
    architecture is specifically designed for multi-variate, multi-horizon
    time series prediction.

    Architecture:
        1. Input layer: (seq_len, n_features)
        2. Conv1D layer: 32 filters, kernel_size=5, ReLU activation
           - Extracts local temporal patterns (5-timestep windows)
        3. Conv1D layer: 64 filters, kernel_size=3, ReLU activation
           - Refines features with smaller receptive field
        4. Batch Normalization: Stabilizes training, faster convergence
        5. Bidirectional LSTM: 64 units (32 forward + 32 backward)
           - Captures long-range temporal dependencies in both directions
           - return_sequences=False: outputs single vector per sequence
        6. Dropout: 0.2 (20% dropout rate)
           - Regularization to prevent overfitting
        7. Dense output: horizon units, linear activation
           - Direct multi-horizon prediction

    Args:
        input_shape: Shape of input sequences as (seq_len, n_features).
            - seq_len: Length of lookback window (e.g., 48 for 48-hour history)
            - n_features: Number of input features (e.g., 45 features)
            Example: (48, 45) for 48-hour lookback with 45 features
        horizon: Number of future timesteps to predict.
            Example: 24 for 24-hour ahead forecasting with hourly data

    Returns:
        Compiled Keras model ready for training.
            - Input shape: (batch_size, seq_len, n_features)
            - Output shape: (batch_size, horizon)
            - Optimizer: Adam with learning rate 1e-3 (0.001)
            - Loss: MSE (Mean Squared Error)

    Example:
        >>> # Build model for 48-hour lookback, 24-hour ahead forecast, 45 features
        >>> model = build_cnn_bilstm(input_shape=(48, 45), horizon=24)
        >>> model.summary()
        Model: "model"
        _________________________________________________________________
         Layer (type)                Output Shape              Param #
        =================================================================
         input_1 (InputLayer)        [(None, 48, 45)]          0
         conv1d (Conv1D)             (None, 48, 32)            7232
         conv1d_1 (Conv1D)           (None, 48, 64)            6208
         batch_normalization         (None, 48, 64)            256
         bidirectional (Bidirect...) (None, 128)               66048
         dropout (Dropout)           (None, 128)               0
         dense (Dense)               (None, 24)                3096
        =================================================================
        Total params: 82,840
        ...
        >>> # Train on windowed data
        >>> history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
        ...                     epochs=50, batch_size=32)

    Note:
        - Model is compiled with Adam optimizer (lr=1e-3) and MSE loss
        - Conv1D with padding='same' preserves sequence length
        - Bidirectional LSTM doubles the number of parameters (64 units → 128 outputs)
        - Dropout is only active during training (automatically disabled during inference)
        - Output uses linear activation (no constraint on predicted values)
        - Total parameters: ~83k (relatively lightweight model)
        - GPU-compatible (uses TensorFlow/Keras operations)

    Training Recommendations:
        - Batch size: 32-128 (depends on available memory)
        - Epochs: 50-200 with early stopping (monitor validation loss)
        - Learning rate: Start with 1e-3, reduce on plateau
        - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        - Validation: Use chronological split (not random)
        - Data preprocessing: Standardize/normalize features before training
        - Regularization: Adjust dropout rate (0.1-0.3) if overfitting occurs

    Architecture Design Choices:
        - CNN layers: Extract local patterns (e.g., morning ramp-up, cloud transients)
        - BiLSTM: Capture both past context (backward) and recent trends (forward)
        - Batch normalization: Improves training stability with deep networks
        - Direct multi-horizon: Predicts all future timesteps simultaneously
          (more efficient than recursive/iterative approaches)
        - MSE loss: Appropriate for regression; penalizes large errors quadratically

    Alternative Architectures:
        - For longer horizons (>24h): Consider attention mechanisms or Transformer
        - For limited data: Reduce model capacity (fewer filters/units)
        - For higher accuracy: Increase depth (more Conv/LSTM layers)
        - For interpretability: Consider LightGBM (simpler, faster, comparable accuracy)

    See Also:
        - window.make_windows: Prepares input data for this model
        - train.py: Training script using this architecture
        - predict.py: Inference script for generating forecasts

    References:
        - Hochreiter & Schmidhuber (1997). "Long Short-Term Memory".
          Neural Computation 9(8):1735-1780.
        - Schuster & Paliwal (1997). "Bidirectional recurrent neural networks".
          IEEE Trans. Signal Processing 45(11):2673-2681.
        - Kim (2014). "Convolutional Neural Networks for Sentence Classification".
          EMNLP 2014. (Inspired 1D CNN for sequences)
    """
    inp = keras.Input(shape=input_shape)

    # CNN feature extraction with increased dropout
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inp)
    x = layers.Dropout(0.2)(x)  # Add dropout after first conv
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.2)(x)  # Add dropout after second conv
    x = layers.BatchNormalization()(x)

    # Bidirectional LSTM with increased capacity and regularization
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False, recurrent_dropout=0.1)
    )(x)
    x = layers.Dropout(0.4)(x)  # Increased dropout before output

    # Output layer
    out = layers.Dense(horizon)(x)

    model = keras.Model(inputs=inp, outputs=out)

    # Lower initial learning rate for better convergence
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="mse"
    )
    return model

