"""
rl_agent/traffic_env.py — Gymnasium Wrapper for SUMO Traffic Control
====================================================================
Wraps a single SUMO junction's traffic light into a Gymnasium environment
for PPO training. Includes the priority action mask for emergency vehicles.

Observation space (7-dim):
    [queue_lane_0, queue_lane_1, queue_lane_2, queue_lane_3,
     current_phase, time_in_phase, priority_flag]

Action space:
    Discrete(n_phases) — select which phase to activate.

Reward:
    -0.5 * total_waiting_time - 0.5 * total_queue_length
    + emergency bonus/penalty when priority is active.
"""

import sys
import os
import numpy as np
from collections import deque

import gymnasium as gym
from gymnasium import spaces

try:
    import traci
    from sumolib import checkBinary
except ImportError:
    print("WARNING: traci/sumolib not found. Install SUMO and set SUMO_HOME.")
    traci = None


class TrafficEnv(gym.Env):
    """
    Single-junction traffic signal control environment.

    This environment controls one traffic light junction in SUMO.
    Multiple instances (one per junction) are used for federated training.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        junction_id: str,
        sumo_cfg: str,
        max_steps: int = 1000,
        n_phases: int = 4,
        use_gui: bool = False,
        delta_time: int = 5,
        sumo_seed: int = None,
    ):
        """
        Args:
            junction_id: SUMO traffic light junction ID to control.
            sumo_cfg: Path to the .sumocfg file.
            max_steps: Max simulation steps per episode.
            n_phases: Number of signal phases at this junction.
            use_gui: If True, launch sumo-gui instead of sumo.
            delta_time: Number of simulation seconds per RL step.
        """
        super().__init__()
        self.junction_id = junction_id
        self.sumo_cfg = os.path.abspath(sumo_cfg)
        self.max_steps = max_steps
        self.n_phases = n_phases
        self.use_gui = use_gui
        self.delta_time = delta_time
        self.sumo_seed = sumo_seed

        self.step_count = 0
        self.priority_active = False
        self.priority_urgency = 0.0
        self._priority_expiry = 0.0

        # LSTM History buffer
        self.queue_history = deque(maxlen=15)

        # --- Spaces ---
        self.action_space = spaces.Discrete(n_phases)
        # obs: 4 lane queues + current_phase + time_in_phase + priority_flag + predicted_congestion
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Will be populated on reset
        self._sumo_binary = None
        self._controlled_lanes = None

    def reset(self, seed=None, options=None):
        """Reset the environment: restart SUMO and return initial observation."""
        super().reset(seed=seed)

        # Close existing SUMO connection if any
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass

        # Start SUMO
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
        ]
        if self.sumo_seed is not None:
            sumo_cmd += ["--seed", str(self.sumo_seed)]
        traci.start(sumo_cmd)

        self.step_count = 0
        self.priority_active = False
        self.priority_urgency = 0.0
        self.queue_history.clear()

        # Cache controlled lanes
        self._controlled_lanes = list(
            dict.fromkeys(
                traci.trafficlight.getControlledLanes(self.junction_id)
            )
        )

        return self._get_obs(), {}

    def step(self, action: int):
        """
        Execute one RL step:
        1. Apply action masking if priority is active.
        2. Set traffic light phase.
        3. Advance simulation by delta_time seconds.
        4. Compute observation and reward.
        """
        # --- Apply action and simulate ---
        # Skip overriding if the external priority mask has locked the traffic light
        # Check duration alone — the mask holds independently of the env's priority_active flag
        phase_duration = traci.trafficlight.getPhaseDuration(self.junction_id)
        if phase_duration < 50:
            traci.trafficlight.setPhase(self.junction_id, action)
            
        for _ in range(self.delta_time):
            traci.simulationStep()

        self.step_count += 1

        # --- Check priority expiry ---
        sim_time = traci.simulation.getTime()
        if self.priority_active and sim_time > self._priority_expiry:
            self.priority_active = False
            self.priority_urgency = 0.0
            
        # --- Update LSTM Queue History ---
        total_queue = sum(
            traci.lane.getLastStepHaltingNumber(l) for l in self._controlled_lanes
        )
        self.queue_history.append(total_queue)

        # --- Observation, reward, done ---
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = self.step_count >= self.max_steps

        info = {
            "step_count": self.step_count,
            "priority_active": self.priority_active,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Build the 8-dimensional observation vector.

        Components (all normalized to [0, 1]):
            [0:4] — queue length per lane (up to 4 lanes, /20 for normalization)
            [4]   — current phase index / n_phases
            [5]   — time spent in current phase / 60s
            [6]   — priority flag (1.0 if emergency active, else 0.0)
            [7]   — predicted congestion (LSTM) / 20.0
        """
        # Queue lengths (take first 4 lanes, pad if fewer)
        queues = []
        for i in range(4):
            if i < len(self._controlled_lanes):
                lane = self._controlled_lanes[i]
                q = traci.lane.getLastStepHaltingNumber(lane) / 20.0
                queues.append(min(q, 1.0))
            else:
                queues.append(0.0)

        phase = traci.trafficlight.getPhase(self.junction_id) / max(self.n_phases, 1)
        time_in_phase = min(
            traci.trafficlight.getSpentDuration(self.junction_id) / 60.0, 1.0
        )
        priority_flag = 1.0 if self.priority_active else 0.0
        
        predicted_congestion = min(self.lstm_predict() / 20.0, 1.0)

        obs = np.array(
            queues + [phase, time_in_phase, priority_flag, predicted_congestion],
            dtype=np.float32,
        )
        return obs[:8]  # safety clamp

    def _compute_reward(self) -> float:
        """
        Reward function balancing traffic flow and emergency priority.

        Base reward: -0.5 * total_waiting_time - 0.5 * total_queue
        LSTM foresight: Penalty if predicted congestion is rising and agent isn't acting.
        """
        total_wait = sum(
            traci.lane.getWaitingTime(l) for l in self._controlled_lanes
        )
        total_queue = sum(
            traci.lane.getLastStepHaltingNumber(l) for l in self._controlled_lanes
        )

        reward = -0.5 * total_wait - 0.5 * total_queue

        # LSTM Foresight Term
        predicted = self.lstm_predict()
        if predicted > total_queue * 1.2:  # Forecast says it's about to get worse
            reward -= 2.0 * (predicted - total_queue) / 20.0  # scaled, kept small deliberately

        # Priority Term
        if self.priority_active:
            if self._ambulance_cleared():
                reward += 50.0
            else:
                reward -= 5.0

        return reward

    def lstm_predict(self) -> float:
        """
        Predict congestion 10 minutes out using the trained LSTM.
        (Stub logic wired up for the PyTorch model injection later).
        """
        recent_queues = list(self.queue_history)
        if len(recent_queues) < 15:
            return 0.0  # not enough history yet, neutral prediction
            
        # STUB: Replace with actual `torch.no_grad(): pred = self.lstm_model(x)`
        # For the stub, we just simulate a rising trend if recent queues are increasing
        trend = recent_queues[-1] - recent_queues[0]
        if trend > 0:
            return recent_queues[-1] + trend * 1.5
        return recent_queues[-1]

    # ------------------------------------------------------------------
    # Priority interface (called by federated/priority_trigger.py)
    # ------------------------------------------------------------------

    def set_priority(self, urgency: float, ttl: float) -> None:
        """
        Activate emergency priority mode.

        Args:
            urgency: Urgency scalar (higher = more urgent).
            ttl: Time-to-live in simulation seconds.
        """
        self.priority_active = True
        self.priority_urgency = urgency
        self._priority_expiry = traci.simulation.getTime() + ttl
        print(
            f"[ENV {self.junction_id}] Priority ON — urgency={urgency:.2f}, "
            f"ttl={ttl:.1f}s"
        )

    def clear_priority(self) -> None:
        """Manually clear emergency priority mode."""
        self.priority_active = False
        self.priority_urgency = 0.0



    def _ambulance_cleared(self) -> bool:
        """Check if the ambulance has left the simulation."""
        return "ambulance_1" not in traci.vehicle.getIDList()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close the SUMO connection."""
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass
