"""
prediction/data_prep.py — Sliding Window Data Framing
=====================================================
Converts raw traffic CSV logs into supervised learning format for the
congestion prediction LSTM.

Input: CSV with columns from shared.schema.TRAFFIC_LOG_COLUMNS
Output: (X, y) numpy arrays where:
    X[i] = queue_length values over a window of `window` timesteps
    y[i] = queue_length value `horizon` steps ahead
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.schema import TRAFFIC_LOG_COLUMNS


def make_windows(
    csv_path: str,
    window: int = 15,
    horizon: int = 10,
    target_junction: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window (X, y) pairs from a traffic log CSV.

    Args:
        csv_path: Path to the traffic log CSV (produced by sumo_env/logger.py).
        window: Number of past timesteps in each input window.
        horizon: Number of steps ahead to predict.
        target_junction: If specified, filter to this junction only.
                         If None, aggregate queue_length across all junctions
                         per timestamp.

    Returns:
        X: np.ndarray of shape (n_samples, window)
        y: np.ndarray of shape (n_samples,)
    """
    df = pd.read_csv(csv_path)

    # Validate columns
    missing_cols = [c for c in ["timestamp", "queue_length"] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV missing required columns: {missing_cols}. "
            f"Expected columns: {TRAFFIC_LOG_COLUMNS}"
        )

    # Filter to target junction if specified
    if target_junction is not None:
        df = df[df["junction_id"] == target_junction]
        if df.empty:
            raise ValueError(f"No data found for junction '{target_junction}'")

    # Aggregate queue_length per timestamp
    series = df.groupby("timestamp")["queue_length"].sum().sort_index().values

    if len(series) < window + horizon + 1:
        raise ValueError(
            f"Not enough data: {len(series)} timesteps, need at least "
            f"{window + horizon + 1} (window={window}, horizon={horizon})"
        )

    # Build sliding windows
    X, y = [], []
    for i in range(len(series) - window - horizon):
        X.append(series[i : i + window])
        y.append(series[i + window + horizon])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def normalize(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Min-max normalize features and targets.

    Returns:
        X_norm, y_norm, min_val, max_val  (use min/max for inverse transform)
    """
    all_vals = np.concatenate([X.flatten(), y.flatten()])
    min_val = float(all_vals.min())
    max_val = float(all_vals.max())
    range_val = max_val - min_val if max_val != min_val else 1.0

    X_norm = (X - min_val) / range_val
    y_norm = (y - min_val) / range_val

    return X_norm, y_norm, min_val, max_val


def train_test_split(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test sets (sequential, no shuffle)."""
    n = int(len(X) * train_ratio)
    return X[:n], y[:n], X[n:], y[n:]


def generate_synthetic_data(
    n_timesteps: int = 500,
    n_junctions: int = 3,
    output_path: str = "synthetic_traffic_log.csv",
) -> str:
    """
    Generate a synthetic traffic log CSV for testing when SUMO data
    isn't available yet.

    Creates sinusoidal queue patterns with noise to simulate realistic
    traffic flow.

    Returns:
        Path to the generated CSV file.
    """
    np.random.seed(42)
    rows = []
    for t in range(n_timesteps):
        for j in range(n_junctions):
            # Sinusoidal base with noise
            base = 10 * np.sin(2 * np.pi * t / 100 + j * np.pi / n_junctions) + 12
            queue = max(0, int(base + np.random.normal(0, 2)))
            wait = max(0.0, base * 2.5 + np.random.normal(0, 5))
            rows.append({
                "timestamp": float(t),
                "junction_id": f"J{j}",
                "queue_length": queue,
                "waiting_time": round(wait, 2),
                "current_phase": t % 4,
                "phase_duration": round((t % 30) * 1.0, 2),
                "vehicle_count": queue + np.random.randint(0, 5),
            })

    df = pd.DataFrame(rows, columns=TRAFFIC_LOG_COLUMNS)
    df.to_csv(output_path, index=False)
    print(f"[DATA] Synthetic traffic log saved to {output_path} ({len(df)} rows)")
    return output_path
