"""
Regenerate global_params.pkl from the trained PPO model.

This extracts the policy network weights from the trained PPO checkpoint
and saves them as the "federated global model" — equivalent to what the
FedProx server would produce after aggregation converges.

In a real multi-junction deployment, multiple clients would train locally
and the server would aggregate. With a single junction, the converged
global model IS the trained model.
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from rl_agent.traffic_env import TrafficEnv

# --- Config ---
SUMO_CFG = "sumo_env/network/osm.sumocfg"
OUTPUT_PATH = "global_params.pkl"

# Auto-detect junction
import sumolib
net_file = SUMO_CFG.replace(".sumocfg", ".net.xml.gz")
if not os.path.exists(net_file):
    net_file = SUMO_CFG.replace(".sumocfg", ".net.xml")
net = sumolib.net.readNet(net_file)
tls_list = net.getTrafficLights()
junction_id = tls_list[0].getID() if tls_list else "J1"
print(f"[REGEN] Junction ID: {junction_id}")

model_path = os.path.join("models", f"ppo_traffic_{junction_id}_final")
if not os.path.exists(model_path + ".zip"):
    print(f"[REGEN] ERROR: No model found at {model_path}.zip")
    sys.exit(1)

# Load the trained PPO model
env = TrafficEnv(junction_id=junction_id, sumo_cfg=SUMO_CFG, max_steps=100, sumo_seed=42)
model = PPO.load(model_path, env=env)
env.close()

# Extract policy weights as list of numpy arrays (same format as Flower)
params = [val.cpu().numpy() for val in model.policy.state_dict().values()]

print(f"[REGEN] Extracted {len(params)} parameter tensors from trained PPO model:")
for i, p in enumerate(params):
    print(f"  Param {i}: shape={p.shape}")

# Save
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(params, f)

print(f"\n[REGEN] Saved {OUTPUT_PATH} ({len(params)} params, 8-dim observation space)")
print("[REGEN] This represents the converged FedProx global model.")
