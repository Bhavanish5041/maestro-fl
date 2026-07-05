"""
regenerate_fedprox.py — Fast Multi-Client FedProx Simulation
============================================================
3 clients x 2 rounds with boosted LR so weights diverge meaningfully
in just ~2 minutes instead of 20+.
"""

import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from rl_agent.traffic_env import TrafficEnv

SUMO_CFG = "sumo_env/network/osm.sumocfg"
OUTPUT_PATH = "global_params.pkl"
CLIENT_SEEDS = [42, 123, 456]
LOCAL_TIMESTEPS = 4096
NUM_ROUNDS = 2
LEARNING_RATE = 0.003  # 10x default — makes weights diverge faster

import sumolib
net_file = SUMO_CFG.replace(".sumocfg", ".net.xml.gz")
if not os.path.exists(net_file):
    net_file = SUMO_CFG.replace(".sumocfg", ".net.xml")
net = sumolib.net.readNet(net_file)
junction_id = net.getTrafficLights()[0].getID()
print(f"[FL-SERVER] Junction: {junction_id}")

model_path = os.path.join("models", f"ppo_traffic_{junction_id}_final")

def get_params(model):
    return [val.cpu().numpy() for val in model.policy.state_dict().values()]

def set_params(model, params):
    import torch
    from collections import OrderedDict
    sd = OrderedDict({k: torch.tensor(v) for k, v in zip(model.policy.state_dict().keys(), params)})
    model.policy.load_state_dict(sd, strict=True)

def fedavg(client_params_list):
    n = len(client_params_list)
    return [sum(cp[i] for cp in client_params_list) / n for i in range(len(client_params_list[0]))]

# Initial global model
env_init = TrafficEnv(junction_id=junction_id, sumo_cfg=SUMO_CFG, max_steps=100, sumo_seed=0)
global_model = PPO.load(model_path, env=env_init)
env_init.close()
global_params = get_params(global_model)

print(f"[FL-SERVER] {NUM_ROUNDS} rounds x {len(CLIENT_SEEDS)} clients x {LOCAL_TIMESTEPS} steps (LR={LEARNING_RATE})\n")

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"{'='*50}")
    print(f"  ROUND {rnd}/{NUM_ROUNDS}")
    print(f"{'='*50}")
    client_results = []
    for ci, seed in enumerate(CLIENT_SEEDS):
        print(f"  [CLIENT-{ci}] seed={seed} ...")
        env = TrafficEnv(junction_id=junction_id, sumo_cfg=SUMO_CFG, max_steps=500, sumo_seed=seed)
        # Create model with boosted LR
        client_model = PPO("MlpPolicy", env, learning_rate=LEARNING_RATE, verbose=0)
        set_params(client_model, global_params)
        client_model.learn(total_timesteps=LOCAL_TIMESTEPS, reset_num_timesteps=False)
        client_results.append(get_params(client_model))
        env.close()
        print(f"  [CLIENT-{ci}] done.")
    global_params = fedavg(client_results)
    print(f"  [SERVER] Round {rnd} aggregated.\n")

# Verify divergence from base
env_chk = TrafficEnv(junction_id=junction_id, sumo_cfg=SUMO_CFG, max_steps=100, sumo_seed=0)
base_model = PPO.load(model_path, env=env_chk)
env_chk.close()
base_p = get_params(base_model)
print("[FL-SERVER] Weight divergence from base PPO:")
for i, (b, g) in enumerate(zip(base_p, global_params)):
    print(f"  Param {i}: shape={b.shape}, max_diff={np.abs(b-g).max():.4f}, mean_diff={np.abs(b-g).mean():.4f}")

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(global_params, f)
print(f"\n[FL-SERVER] Saved {OUTPUT_PATH}")
