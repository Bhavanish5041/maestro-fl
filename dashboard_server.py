import os
import sys
import asyncio
import json
import logging
from typing import Dict, List, Optional
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

import traci
from sumolib import checkBinary

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_agent.traffic_env import TrafficEnv

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

app = FastAPI(title="MAESTRO-FL Interactive Dashboard")

# Global simulation state
class SimulationRunner:
    def __init__(self):
        self.active_condition = 1
        self.running = False
        self.env: Optional[TrafficEnv] = None
        self.model = None
        self.junction_id = None
        self.sumo_cfg = "sumo_env/network/osm.sumocfg"
        self.seed = 42
        
        self.step_count = 0
        self.ambulance_injected = False
        self.ambulance_start_time = None
        self.ambulance_travel_time = None
        self.ambulance_waiting_time = 0.0
        
        self.metrics_history = {
            "waiting_time": [],
            "queue_length": []
        }
        self.log_messages = []
        self.tls_center = (741.06, 687.10)
        
        self.AMB_ROUTE_EDGES = [
            "40633855#3", "40633855#4", "27673609#1", "27673609#2",
            "27673609#3", "27673609#4", "1222891448#0",
            "1222891447#2", "1222891447#3"
        ]

    def add_log(self, msg: str):
        logger.info(msg)
        self.log_messages.append(msg)
        if len(self.log_messages) > 100:
            self.log_messages.pop(0)

    async def start(self, condition: int):
        if self.running:
            self.stop()
            await asyncio.sleep(0.5)

        self.active_condition = condition
        self.running = True
        self.step_count = 0
        self.ambulance_injected = False
        self.ambulance_start_time = None
        self.ambulance_travel_time = None
        self.ambulance_waiting_time = 0.0
        self.metrics_history = {"waiting_time": [], "queue_length": []}
        self.log_messages.clear()
        
        self.add_log(f"Starting simulation in Condition {condition}...")

        # Initialize Environment
        # Auto-detect junction first if needed
        try:
            import sumolib
            net_file = self.sumo_cfg.replace(".sumocfg", ".net.xml.gz")
            if not os.path.exists(net_file):
                net_file = self.sumo_cfg.replace(".sumocfg", ".net.xml")
            net = sumolib.net.readNet(net_file)
            tls_list = net.getTrafficLights()
            if tls_list:
                self.junction_id = tls_list[0].getID()
            else:
                self.junction_id = "J1"
        except Exception as e:
            self.junction_id = "J1"
            self.add_log(f"Junction detection error: {e}")

        self.add_log(f"Junction ID: {self.junction_id}")

        # Try to load PPO model if needed
        self.model = None
        if condition in [2, 3, 4]:
            try:
                from stable_baselines3 import PPO
                import pickle
                
                model_path = os.path.join("models", f"ppo_traffic_{self.junction_id}_final")
                if os.path.exists(model_path + ".zip"):
                    # Temporarily load env to satisfy PPO requirements
                    temp_env = TrafficEnv(junction_id=self.junction_id, sumo_cfg=self.sumo_cfg, max_steps=5000, sumo_seed=self.seed)
                    self.model = PPO.load(model_path, env=temp_env)
                    temp_env.close()
                    
                    # If Condition 3 or 4, try loading federated weights
                    if condition in [3, 4]:
                        fedprox_params_path = "global_params.pkl"
                        if os.path.exists(fedprox_params_path):
                            with open(fedprox_params_path, "rb") as f:
                                global_params = pickle.load(f)
                            processed_params = []
                            for i, p in enumerate(global_params):
                                if i in [0, 4] and p.shape == (64, 7):
                                    p = np.hstack([p, np.zeros((64, 1), dtype=p.dtype)])
                                    self.add_log(f"[FL-FEDPROX] Padded parameter {i} from (64, 7) to (64, 8) with zeros")
                                processed_params.append(p)
                            from collections import OrderedDict
                            import torch
                            params_dict = zip(self.model.policy.state_dict().keys(), processed_params)
                            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                            self.model.policy.load_state_dict(state_dict, strict=False)
                            self.add_log("[FL-FEDPROX] Loaded aggregated global model weights out-of-cycle")
                        else:
                            self.add_log("[FL-FEDPROX] No global_params.pkl found. Falling back to base PPO weights.")
                    self.add_log("RL PPO Model loaded successfully.")
                else:
                    self.add_log(f"WARNING: No trained PPO model found at {model_path}. Running with random actions.")
            except Exception as e:
                self.add_log(f"Error loading PPO model: {e}")

        # Start environment
        try:
            self.env = TrafficEnv(
                junction_id=self.junction_id,
                sumo_cfg=self.sumo_cfg,
                max_steps=5000,
                sumo_seed=self.seed,
                use_gui=False
            )
            self.obs, _ = self.env.reset()
            self.add_log("SUMO simulation engine initialized.")
            
            # Read TLS location
            controlled = traci.trafficlight.getControlledLanes(self.junction_id)
            if controlled:
                xs, ys = [], []
                for lane in controlled:
                    shape = traci.lane.getShape(lane)
                    if shape:
                        xs.append(shape[-1][0])
                        ys.append(shape[-1][1])
                if xs:
                    self.tls_center = (sum(xs)/len(xs), sum(ys)/len(ys))
        except Exception as e:
            self.running = False
            self.add_log(f"Error initializing SUMO: {e}")

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.add_log("Stopping simulation...")
        if self.env:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = None
        self.model = None
        self.add_log("Simulation stopped.")

    def inject_ambulance(self) -> bool:
        if not self.running or self.ambulance_injected:
            return False
        
        try:
            # Register route if needed
            if "amb_route" not in traci.route.getIDList():
                traci.route.add("amb_route", self.AMB_ROUTE_EDGES)
            
            traci.vehicle.add(
                vehID="ambulance_1",
                routeID="amb_route",
                typeID="emergency",
                depart="now",
                departPos="0"
            )
            traci.vehicle.setVehicleClass("ambulance_1", "emergency")
            
            if self.active_condition == 4:
                # MAESTRO-FL Priority Trigger
                traci.vehicle.setSpeedMode("ambulance_1", 7) # ignore right-of-way
                self.env.set_priority(urgency=1.0, ttl=40.0)
                self.add_log("[MAESTRO-FL] 🚨 Out-of-Cycle Priority Triggered!")
                self.add_log("[MAESTRO-FL] Model params pushed to downstream client J1 immediately")
            else:
                self.add_log("[BASELINE] Ambulance injected without priority override.")

            self.ambulance_injected = True
            self.ambulance_start_time = traci.simulation.getTime()
            return True
        except Exception as e:
            self.add_log(f"Error injecting ambulance: {e}")
            return False

    def step(self) -> Optional[dict]:
        if not self.running or not self.env:
            return None

        sim_time = traci.simulation.getTime()
        
        # Determine Action
        action = 0
        if self.model is not None:
            action, _ = self.model.predict(self.obs, deterministic=True)
            action = int(action)
        else:
            # Fallback to fixed-timer rotation or random actions
            if self.active_condition == 1:
                action = (int(sim_time) // 30) % 4
            else:
                action = int(np.random.randint(0, 4))

        # Advance simulation
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Apply TraCI priority mask in MAESTRO-FL mode (forces green along route)
        if self.active_condition == 4 and self.ambulance_injected:
            if "ambulance_1" in traci.vehicle.getIDList():
                from rl_agent.priority_mask import force_green_along_route, release_green_lock
                force_green_along_route("ambulance_1", lookahead=2)
                release_green_lock("ambulance_1")

        # Collect metrics
        controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.junction_id)))
        total_waiting_time = float(sum(traci.lane.getWaitingTime(l) for l in controlled_lanes))
        total_queue_length = float(sum(traci.lane.getLastStepHaltingNumber(l) for l in controlled_lanes))
        
        self.metrics_history["waiting_time"].append(total_waiting_time)
        self.metrics_history["queue_length"].append(total_queue_length)
        if len(self.metrics_history["waiting_time"]) > 200:
            self.metrics_history["waiting_time"].pop(0)
            self.metrics_history["queue_length"].pop(0)

        # Track ambulance metrics
        ambulance_active = False
        ambulance_x, ambulance_y = 0.0, 0.0
        ambulance_speed = 0.0
        
        if self.ambulance_injected:
            if "ambulance_1" in traci.vehicle.getIDList():
                ambulance_active = True
                ambulance_x, ambulance_y = traci.vehicle.getPosition("ambulance_1")
                ambulance_speed = traci.vehicle.getSpeed("ambulance_1")
                if ambulance_speed < 0.1:
                    self.ambulance_waiting_time += 1.0
            else:
                # Completed
                if self.ambulance_travel_time is None:
                    self.ambulance_travel_time = sim_time - self.ambulance_start_time
                    self.add_log(f"[SIM] Ambulance completed route! Travel time: {self.ambulance_travel_time:.1f}s, Waiting: {self.ambulance_waiting_time:.1f}s")
                    if self.active_condition == 4:
                        self.env.clear_priority()
                        self.add_log("[MAESTRO-FL] Priority green lock released. Resuming normal RL control.")

        # Read TLS State
        tls_state = traci.trafficlight.getRedYellowGreenState(self.junction_id)
        
        # Read vehicle positions (within 300m of center)
        vehicles = []
        for veh_id in traci.vehicle.getIDList():
            try:
                x, y = traci.vehicle.getPosition(veh_id)
                dist = np.sqrt((x - self.tls_center[0])**2 + (y - self.tls_center[1])**2)
                if dist <= 300.0:
                    vclass = traci.vehicle.getVehicleClass(veh_id)
                    speed = traci.vehicle.getSpeed(veh_id)
                    angle = traci.vehicle.getAngle(veh_id)
                    vehicles.append({
                        "id": veh_id,
                        "x": float(x),
                        "y": float(y),
                        "angle": float(angle),
                        "type": "emergency" if vclass == "emergency" else "normal",
                        "speed": float(speed)
                    })
            except Exception:
                continue

        state_packet = {
            "time": float(sim_time),
            "condition": self.active_condition,
            "metrics": {
                "waiting_time": total_waiting_time,
                "queue_length": total_queue_length,
                "waiting_time_history": self.metrics_history["waiting_time"],
                "queue_length_history": self.metrics_history["queue_length"]
            },
            "ambulance": {
                "injected": self.ambulance_injected,
                "active": ambulance_active,
                "x": float(ambulance_x),
                "y": float(ambulance_y),
                "speed": float(ambulance_speed),
                "travel_time": self.ambulance_travel_time,
                "waiting_time": self.ambulance_waiting_time
            },
            "tls": {
                "id": self.junction_id,
                "state": tls_state,
                "center": [float(self.tls_center[0]), float(self.tls_center[1])]
            },
            "vehicles": vehicles,
            "logs": list(self.log_messages)
        }
        
        # Clear log list for next frame
        self.log_messages.clear()
        
        # Termination conditions
        if terminated or truncated:
            self.stop()
            state_packet["terminated"] = True
            
        return state_packet

runner = SimulationRunner()

# CORS & static files
os.makedirs("dashboard", exist_ok=True)

@app.post("/api/start")
async def start_sim(data: dict):
    condition = data.get("condition", 1)
    await runner.start(condition)
    return JSONResponse({"status": "success", "message": f"Simulation started in mode {condition}"})

@app.post("/api/stop")
async def stop_sim():
    runner.stop()
    return JSONResponse({"status": "success", "message": "Simulation stopped"})

@app.post("/api/inject")
async def inject_amb():
    success = runner.inject_ambulance()
    return JSONResponse({"status": "success" if success else "error", "message": "Ambulance injected" if success else "Failed to inject"})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    try:
        while True:
            # Advance simulation if active
            if runner.running:
                packet = runner.step()
                if packet:
                    await websocket.send_json(packet)
                else:
                    await websocket.send_json({"running": False})
            else:
                await websocket.send_json({"running": False})
            # Control simulation speed (~10 steps per second)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# Mount static files after endpoints to allow index fallback if needed
app.mount("/", StaticFiles(directory="dashboard", html=True), name="static")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start Dashboard Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    args = parser.parse_args()
    
    uvicorn.run("dashboard_server:app", host="0.0.0.0", port=args.port, reload=False)
