"""
prediction/train_lstm.py — LSTM Training Script
================================================
Trains the CongestionLSTM on traffic log data produced by sumo_env/logger.py.
Supports both real SUMO data and synthetic data for early testing.

Usage:
    python train_lstm.py                          # uses default paths
    python train_lstm.py --csv path/to/log.csv    # custom data
    python train_lstm.py --synthetic               # generate + train on synthetic data
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from prediction.lstm_model import CongestionLSTM
from prediction.data_prep import make_windows, normalize, train_test_split
from prediction.data_prep import generate_synthetic_data


def train(
    csv_path: str,
    window: int = 15,
    horizon: int = 10,
    hidden_size: int = 32,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "lstm_congestion.pt",
    target_junction: str = None,
) -> dict:
    """
    Full training pipeline: data prep → train → eval → save.

    Args:
        csv_path: Path to traffic log CSV.
        window: Input window size (timesteps).
        horizon: Prediction horizon (timesteps ahead).
        hidden_size: LSTM hidden dimension.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        save_path: Where to save the trained model.
        target_junction: Optional junction to filter data to.

    Returns:
        Dict with final MAE, RMSE, and training loss history.
    """
    # --- Data preparation ---
    print(f"[LSTM] Loading data from {csv_path}...")
    X, y = make_windows(csv_path, window=window, horizon=horizon,
                        target_junction=target_junction)
    print(f"[LSTM] Created {len(X)} samples (window={window}, horizon={horizon})")

    # Normalize
    X_norm, y_norm, min_val, max_val = normalize(X, y)

    # Split
    X_train, y_train, X_test, y_test = train_test_split(X_norm, y_norm)
    print(f"[LSTM] Train: {len(X_train)}, Test: {len(X_test)}")

    # To tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    # --- Model ---
    model = CongestionLSTM(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # --- Training loop ---
    loss_history = []
    print(f"[LSTM] Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} — Loss: {avg_loss:.6f}")

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t).squeeze(-1)
        mae = torch.mean(torch.abs(pred - y_test_t)).item()
        rmse = torch.sqrt(torch.mean((pred - y_test_t) ** 2)).item()

        # De-normalize for interpretable metrics
        range_val = max_val - min_val if max_val != min_val else 1.0
        mae_actual = mae * range_val
        rmse_actual = rmse * range_val

    print(f"\n[LSTM] Evaluation Results:")
    print(f"  MAE  (normalized): {mae:.4f}  |  MAE  (actual): {mae_actual:.2f}")
    print(f"  RMSE (normalized): {rmse:.4f}  |  RMSE (actual): {rmse_actual:.2f}")

    # --- Save model ---
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_size": hidden_size,
            "window": window,
            "horizon": horizon,
            "min_val": min_val,
            "max_val": max_val,
        },
        save_path,
    )
    print(f"[LSTM] Model saved to {save_path}")

    return {
        "mae": mae,
        "rmse": rmse,
        "mae_actual": mae_actual,
        "rmse_actual": rmse_actual,
        "loss_history": loss_history,
    }


def load_model(model_path: str) -> CongestionLSTM:
    """Load a trained LSTM model from checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu")
    model = CongestionLSTM(hidden_size=checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train congestion prediction LSTM")
    parser.add_argument(
        "--csv",
        default="../sumo_env/logs/traffic_log.csv",
        help="Path to traffic log CSV",
    )
    parser.add_argument("--window", type=int, default=15, help="Input window size")
    parser.add_argument(
        "--horizon", type=int, default=10, help="Prediction horizon"
    )
    parser.add_argument(
        "--hidden", type=int, default=32, help="LSTM hidden size"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--save", default="lstm_congestion.pt", help="Model save path"
    )
    parser.add_argument(
        "--junction", default=None, help="Filter to specific junction"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate and train on synthetic data",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if args.synthetic:
        csv_path = generate_synthetic_data()

    train(
        csv_path=csv_path,
        window=args.window,
        horizon=args.horizon,
        hidden_size=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save,
        target_junction=args.junction,
    )
