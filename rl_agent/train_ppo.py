"""
rl_agent/train_ppo.py — PPO Training Script
============================================
Trains a PPO agent on a single junction using Stable-Baselines3.
Run standalone or import `train_ppo()` from federated/client.py for
local training rounds within the FL loop.
"""

import os
import sys
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl_agent.traffic_env import TrafficEnv


def make_env(
    junction_id: str,
    sumo_cfg: str,
    max_steps: int = 1000,
    use_gui: bool = False,
) -> TrafficEnv:
    """Create and validate a TrafficEnv instance."""
    env = TrafficEnv(
        junction_id=junction_id,
        sumo_cfg=sumo_cfg,
        max_steps=max_steps,
        use_gui=use_gui,
    )
    return env


def train_ppo(
    junction_id: str = "J1",
    sumo_cfg: str = "../sumo_env/network/osm.sumocfg",
    total_timesteps: int = 500_000,
    learning_rate: float = 3e-4,
    save_path: str = "models",
    use_gui: bool = False,
) -> PPO:
    """
    Train a PPO model on a single junction.

    Args:
        junction_id: SUMO junction to control.
        sumo_cfg: Path to SUMO configuration file.
        total_timesteps: Total training timesteps.
        learning_rate: PPO learning rate.
        save_path: Directory to save model checkpoints.
        use_gui: Whether to launch SUMO with GUI.

    Returns:
        Trained PPO model.
    """
    # Auto-detect junction if not explicitly provided
    if junction_id == "J1" or junction_id == "auto":
        import sumolib
        net_file = sumo_cfg.replace(".sumocfg", ".net.xml.gz")
        if not os.path.exists(net_file):
            net_file = sumo_cfg.replace(".sumocfg", ".net.xml")
        net = sumolib.net.readNet(net_file)
        tls_list = net.getTrafficLights()
        if not tls_list:
            raise ValueError("No traffic lights found in the network!")
        junction_id = tls_list[0].getID()
        print(f"[TRAIN] Auto-detected TLS ID: {junction_id}")

    env = make_env(junction_id, sumo_cfg, use_gui=use_gui)

    # Validate environment
    print(f"[TRAIN] Validating environment for junction {junction_id}...")
    check_env(env, warn=True)

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    # Checkpoint callback — save every 50k steps
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=save_path,
        name_prefix=f"ppo_traffic_{junction_id}",
    )

    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        # tensorboard_log=os.path.join(save_path, "tb_logs"),  # uncomment if tensorboard is installed
    )

    print(f"[TRAIN] Starting PPO training — {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(save_path, f"ppo_traffic_{junction_id}_final")
    model.save(final_path)
    print(f"[TRAIN] Model saved to {final_path}")

    env.close()
    return model


def load_model(model_path: str, env: TrafficEnv = None) -> PPO:
    """Load a previously saved PPO model."""
    return PPO.load(model_path, env=env)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO traffic agent")
    parser.add_argument("--junction", default="J1", help="Junction ID")
    default_cfg = "sumo_env/network/osm.sumocfg" if os.path.exists("sumo_env/network/osm.sumocfg") else "../sumo_env/network/osm.sumocfg"
    
    parser.add_argument(
        "--sumo-cfg",
        default=default_cfg,
        help="Path to SUMO .sumocfg",
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000, help="Training timesteps"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI")
    parser.add_argument(
        "--save-path", default="models", help="Model save directory"
    )
    args = parser.parse_args()

    train_ppo(
        junction_id=args.junction,
        sumo_cfg=args.sumo_cfg,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        save_path=args.save_path,
        use_gui=args.gui,
    )
