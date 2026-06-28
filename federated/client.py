"""
federated/client.py — Flower Federated Learning Client
=======================================================
Each intersection runs one FlowerClient wrapping the PPO policy network.
The client participates in FedProx rounds and can receive out-of-cycle
model pushes from the PriorityCoordinator.

Integration flow:
    1. Server broadcasts global model → client.set_parameters()
    2. Client runs local PPO training → client.fit()
    3. Client sends updated weights → server aggregates (FedProx)
    4. (Novel) PriorityCoordinator can push weights out-of-cycle
"""

import os
import sys
from typing import List, Dict, Any, Tuple
from collections import OrderedDict

import numpy as np
import torch

try:
    import flwr as fl
except ImportError:
    print("WARNING: flwr not found. Install with: pip install flwr")
    fl = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TrafficClient(fl.client.NumPyClient):
    """
    Flower client for a single junction's PPO policy network.

    The client extracts the MLP policy weights from the SB3 PPO model,
    participates in federated rounds, and supports emergency model pushes.
    """

    def __init__(
        self,
        junction_id: str,
        policy_net: torch.nn.Module,
        train_fn=None,
        eval_fn=None,
    ):
        """
        Args:
            junction_id: SUMO junction this client controls.
            policy_net: The PyTorch neural network (PPO's policy MLP).
            train_fn: Callable that runs local training and returns
                      (num_examples, metrics_dict). If None, fit() is a no-op.
            eval_fn: Callable that runs local evaluation and returns
                     (loss, num_examples, metrics_dict). If None, evaluate()
                     returns dummy values.
        """
        super().__init__()
        self.junction_id = junction_id
        self.model = policy_net
        self._train_fn = train_fn
        self._eval_fn = eval_fn

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Extract model parameters as a list of numpy arrays."""
        return [
            val.cpu().numpy()
            for val in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load model parameters from a list of numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Receive global parameters, run local training, return updated weights.

        Args:
            parameters: Global model parameters from server.
            config: Training config from server (e.g., epochs, lr).

        Returns:
            (updated_parameters, num_examples, metrics)
        """
        self.set_parameters(parameters)
        print(f"[CLIENT {self.junction_id}] Received global model, starting local training...")

        num_examples = 0
        metrics = {}

        if self._train_fn is not None:
            num_examples, metrics = self._train_fn()
        else:
            # Placeholder: in production, run PPO.learn() for N steps here
            num_examples = 1000
            print(f"[CLIENT {self.junction_id}] (placeholder) Local training complete.")

        return self.get_parameters(config={}), num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Receive global parameters, evaluate locally, return metrics.

        Args:
            parameters: Global model parameters from server.
            config: Evaluation config.

        Returns:
            (loss, num_examples, metrics)
        """
        self.set_parameters(parameters)

        if self._eval_fn is not None:
            loss, num_examples, metrics = self._eval_fn()
        else:
            # Placeholder evaluation
            loss = 0.0
            num_examples = 100
            metrics = {"junction_id": self.junction_id}
            print(f"[CLIENT {self.junction_id}] (placeholder) Evaluation complete.")

        return loss, num_examples, metrics

    def receive_emergency_push(self, parameters: List[np.ndarray]) -> None:
        """
        Handle an out-of-cycle model push from the PriorityCoordinator.

        This is the novel mechanism: when an emergency is detected,
        the coordinator pushes the latest global model immediately,
        bypassing the normal FL round schedule.
        """
        self.set_parameters(parameters)
        print(
            f"[CLIENT {self.junction_id}] Emergency model push received — "
            f"weights updated out-of-cycle."
        )


def start_client(
    junction_id: str,
    policy_net: torch.nn.Module,
    server_address: str = "127.0.0.1:8080",
    train_fn=None,
    eval_fn=None,
) -> None:
    """
    Convenience function to start a Flower client.

    Args:
        junction_id: SUMO junction ID.
        policy_net: PyTorch policy network.
        server_address: Flower server address.
        train_fn: Local training function.
        eval_fn: Local evaluation function.
    """
    client = TrafficClient(
        junction_id=junction_id,
        policy_net=policy_net,
        train_fn=train_fn,
        eval_fn=eval_fn,
    )
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )

if __name__ == "__main__":
    import argparse
    from stable_baselines3 import PPO
    from rl_agent.traffic_env import TrafficEnv

    parser = argparse.ArgumentParser(description="MAESTRO-FL Federated Client")
    parser.add_argument("--junction", required=True, help="SUMO junction ID")
    parser.add_argument("--server", default="127.0.0.1:8080", help="Server address")
    parser.add_argument("--sumo-cfg", default="sumo_env/network/osm.sumocfg", help="SUMO config file")
    args = parser.parse_args()

    print(f"[CLIENT] Starting client for junction {args.junction} connecting to {args.server}")
    env = TrafficEnv(junction_id=args.junction, sumo_cfg=args.sumo_cfg)
    
    # Initialize a new dummy PPO model to extract the PyTorch policy net for federated weights
    model = PPO("MlpPolicy", env, verbose=0)
    
    start_client(
        junction_id=args.junction,
        policy_net=model.policy,
        server_address=args.server,
    )
