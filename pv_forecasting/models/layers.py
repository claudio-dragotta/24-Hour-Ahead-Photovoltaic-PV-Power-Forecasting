"""Custom neural network layers for PV forecasting models.

This module provides reusable layer implementations for advanced
time series forecasting architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SoftAttention(nn.Module):
    """Soft attention mechanism for adaptive feature fusion.

    Learns to dynamically weight multiple input branches based on their
    relevance to the forecasting task. Unlike fixed ensemble weights,
    attention scores are computed per-sample and can adapt to different
    input conditions.

    Args:
        input_dim: Dimension of input features from each branch.
        hidden_dim: Dimension of the hidden attention layer.
        temperature: Scaling factor for attention logits (default: 1.0).
            Higher temperature -> more uniform weights
            Lower temperature -> more peaked distribution

    Example:
        >>> # Fuse two branches: PV history + Weather history
        >>> attention = SoftAttention(input_dim=512, hidden_dim=512, temperature=1.0)
        >>> pv_features = torch.randn(32, 512)  # batch_size=32
        >>> wx_features = torch.randn(32, 512)
        >>> fused = attention(torch.stack([pv_features, wx_features], dim=1))
        >>> fused.shape
        torch.Size([32, 512])
    """

    def __init__(self, input_dim: int, hidden_dim: int, temperature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Learnable transformation to attention space
        self.fc = nn.Linear(input_dim, hidden_dim, bias=True)

        # Learnable context vector for attention scoring
        self.context_vector = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.context_vector, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted fusion of input branches.

        Args:
            x: Input tensor of shape (batch_size, num_branches, input_dim)
               where num_branches is the number of signals to fuse (e.g., 2 or 3).

        Returns:
            Fused output of shape (batch_size, input_dim), where each sample
            is a weighted combination of the input branches with learned attention.
        """
        # x: (batch_size, num_branches, input_dim)

        # Project to attention space and apply non-linearity
        h = torch.tanh(self.fc(x))  # (batch_size, num_branches, hidden_dim)

        # Compute attention scores via inner product with context vector
        attention_logits = torch.matmul(h, self.context_vector)  # (batch_size, num_branches)

        # Apply temperature scaling (higher temperature => more uniform weights) and softmax normalization
        denom = max(self.temperature, 1e-6)
        attention_weights = torch.softmax(attention_logits / denom, dim=1)

        # Weighted sum of input branches
        # attention_weights: (batch_size, num_branches)
        # x: (batch_size, num_branches, input_dim)
        fused = torch.sum(attention_weights.unsqueeze(-1) * x, dim=1)  # (batch_size, input_dim)

        return fused


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models.

    Adds learnable position information to input embeddings using
    sine and cosine functions of different frequencies.

    Args:
        d_model: Dimension of model embeddings.
        max_len: Maximum sequence length (default: 5000).
        dropout: Dropout probability (default: 0.1).
        batch_first: Whether input has batch dimension first (default: True).

    Reference:
        "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) if batch_first=True,
               otherwise (seq_len, batch_size, d_model).

        Returns:
            Tensor with same shape as input, with positional encodings added.
        """
        if self.batch_first:
            # x: (batch_size, seq_len, d_model)
            x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
            x = x + self.pe[: x.size(0), :]
            x = x.permute(1, 0, 2)  # back to (batch_size, seq_len, d_model)
        else:
            # x: (seq_len, batch_size, d_model)
            x = x + self.pe[: x.size(0), :]

        return self.dropout(x)
