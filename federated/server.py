"""
federated/server.py — Flower Federated Learning Server
======================================================
Runs the FedProx aggregation server. Each round:
    1. Sends global model to all clients.
    2. Clients train locally and return updated weights.
    3. Server aggregates with FedProx (proximal term prevents drift).

The PriorityCoordinator (priority_trigger.py) runs alongside this
server and shares access to the global model parameters.

Usage:
    python server.py                     # default settings
    python server.py --rounds 50 --mu 0.2  # custom FedProx config
"""

import os
import sys
import argparse
import pickle
from typing import List, Optional, Tuple, Dict

import numpy as np

try:
    import flwr as fl
    from flwr.server.strategy import FedProx
    from flwr.common import Parameters, FitRes, ndarrays_to_parameters
except ImportError:
    print("WARNING: flwr not found. Install with: pip install flwr")
    fl = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class MaestroFedProxStrategy(FedProx):
    """
    Extended FedProx strategy that saves global model parameters after
    each round, making them available to the PriorityCoordinator for
    out-of-cycle emergency pushes.
    """

    def __init__(
        self,
        proximal_mu: float = 0.1,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        params_save_path: str = "global_params.pkl",
        **kwargs,
    ):
        """
        Args:
            proximal_mu: FedProx proximal term weight. Higher = more resistance
                         to drift from global model. Start with 0.1, tune up
                         if clients diverge.
            fraction_fit: Fraction of clients to train per round.
            min_fit_clients: Minimum number of clients required per round.
            min_available_clients: Minimum clients that must be connected.
            params_save_path: Where to save global params (read by
                              PriorityCoordinator).
        """
        super().__init__(
            proximal_mu=proximal_mu,
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
        self.params_save_path = params_save_path
        self._latest_params: Optional[List[np.ndarray]] = None

    def aggregate_fit(self, server_round, results, failures):
        """Override to save global params after aggregation."""
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is not None:
            parameters, metrics = aggregated
            # Save for PriorityCoordinator access
            self._save_global_params(parameters)
            print(
                f"[SERVER] Round {server_round} complete — "
                f"global model saved to {self.params_save_path}"
            )

        return aggregated

    def _save_global_params(self, parameters: Parameters) -> None:
        """Serialize global parameters to disk for side-channel access."""
        try:
            # Convert Parameters to list of numpy arrays
            ndarrays = fl.common.parameters_to_ndarrays(parameters)
            self._latest_params = ndarrays

            with open(self.params_save_path, "wb") as f:
                pickle.dump(ndarrays, f)
        except Exception as e:
            print(f"[SERVER] Warning: could not save global params: {e}")

    def get_latest_params(self) -> Optional[List[np.ndarray]]:
        """
        Get the latest global model parameters (in-memory).
        Used by PriorityCoordinator for immediate out-of-cycle pushes.
        """
        return self._latest_params


def load_global_params(params_path: str = "global_params.pkl") -> Optional[List[np.ndarray]]:
    """
    Load global parameters from disk.
    Called by PriorityCoordinator when in-memory access isn't available.
    """
    try:
        with open(params_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"[SERVER] No global params found at {params_path}")
        return None


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 20,
    proximal_mu: float = 0.1,
    min_clients: int = 2,
    params_save_path: str = "global_params.pkl",
) -> None:
    """
    Start the Flower federated learning server.

    Args:
        server_address: Address to bind (host:port).
        num_rounds: Number of FL training rounds.
        proximal_mu: FedProx proximal term weight.
        min_clients: Minimum clients to start a round.
        params_save_path: Where to save global model after each round.
    """
    strategy = MaestroFedProxStrategy(
        proximal_mu=proximal_mu,
        fraction_fit=1.0,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        params_save_path=params_save_path,
    )

    print(f"[SERVER] Starting Flower server at {server_address}")
    print(f"[SERVER] FedProx μ={proximal_mu}, rounds={num_rounds}, min_clients={min_clients}")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAESTRO-FL Federated Server")
    parser.add_argument(
        "--address", default="0.0.0.0:8080", help="Server address"
    )
    parser.add_argument(
        "--rounds", type=int, default=20, help="Number of FL rounds"
    )
    parser.add_argument(
        "--mu", type=float, default=0.1, help="FedProx proximal_mu"
    )
    parser.add_argument(
        "--min-clients", type=int, default=2, help="Min clients per round"
    )
    parser.add_argument(
        "--params-path",
        default="global_params.pkl",
        help="Path to save global params",
    )
    args = parser.parse_args()

    start_server(
        server_address=args.address,
        num_rounds=args.rounds,
        proximal_mu=args.mu,
        min_clients=args.min_clients,
        params_save_path=args.params_path,
    )
