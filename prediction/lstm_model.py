"""
prediction/lstm_model.py — Congestion Prediction LSTM
=====================================================
Single-layer LSTM with a linear head for predicting future queue lengths.

Architecture:
    Input:  (batch, seq_len)     — raw queue length time series
    LSTM:   (batch, seq_len, 1)  → hidden_size
    Output: (batch, 1)           — predicted queue length at t+horizon
"""

import torch
import torch.nn as nn


class CongestionLSTM(nn.Module):
    """
    LSTM-based model for short-term congestion prediction.

    Takes a window of past queue lengths and predicts the queue length
    `horizon` steps into the future. Used to provide predictive input
    to the RL agent and as a local model in the federated learning setup.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_size: Number of features per timestep (1 for univariate).
            hidden_size: LSTM hidden dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout between LSTM layers (only used if num_layers > 1).
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len) or (batch, seq_len, 1).

        Returns:
            Predictions of shape (batch, 1).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len) → (batch, seq_len, 1)

        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last timestep's hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        return self.fc(last_hidden)  # (batch, 1)

    def predict(self, x: torch.Tensor) -> float:
        """
        Single-sample inference (no grad).

        Args:
            x: Tensor of shape (seq_len,) or (1, seq_len).

        Returns:
            Predicted scalar value.
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)  # (seq_len,) → (1, seq_len)
            return self.forward(x).item()
