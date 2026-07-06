import sys
import os
import traci
import threading
import asyncio
import time
import numpy as np
from collections import deque
import pickle
import traceback

try:
    import pyproj
except ImportError:
    print("[SumoRunner] pyproj not found. Attempting to install it into the current python environment...")
    import subprocess
    import importlib
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyproj"])
        importlib.invalidate_caches()
        import pyproj
        print("[SumoRunner] pyproj successfully installed and imported.")
    except Exception as e:
        print("[SumoRunner] Failed to auto-install pyproj:", e)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "rl_agent"))

from rl_agent.priority_mask import force_green_along_route, release_green_lock
from backend.coordinate_utils import sumo_to_latlon, get_junction_positions

SUMO_CFG = os.path.join(REPO_ROOT, "sumo_env", "network", "osm.sumocfg")
TLS_ID = "GS_cluster_11303526465_13072877373_13072877377_13072877378_#2more"

# Ambulance route endpoints — crosses BOTH TLS junctions
# TLS1 (GS_cluster_11303526465...) input: 1044521114#0
# TLS1 output -> TLS2 (GS_cluster_10123822790...) input: 1222891448#0
# Route: start -> TLS1 -> TLS2 -> destination
AMB_FROM_EDGE = "1044521114#0"
AMB_TO_EDGE = "-1102792424"

class SumoRunner:
    def __init__(self, state_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.state_queue = state_queue
        self.loop = loop
        
        self.thread = None
        self.running = False
        self.paused = False
        self.condition = "fixed_timer" # "fixed_timer", "ppo_only", "maestro_fl"
        self.inject_requested = False
        self.reset_requested = False

        self.model = None
        self.env_sim = None
        self.junction_positions = {}

        # Priority wave tracking
        self._synced_junctions = set()
        self._route_junction_ids = []  # ordered list of TLS IDs along the ambulance route
        self._sync_schedule = {}  # junction_id -> step at which to fire sync event
        
    def start_simulation(self, condition):
        if self.running:
            return
        self.condition = condition
        self.running = True
        self.paused = False
        self.inject_requested = False
        self.reset_requested = False
        self.thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.thread.start()

    def stop_simulation(self):
        self.running = False
        self.reset_requested = True
        if self.thread:
            self.thread.join(timeout=2.0)

    def pause_simulation(self):
        self.paused = True

    def resume_simulation(self):
        self.paused = False

    def request_inject(self):
        self.inject_requested = True

    def _sim_loop(self):
        try:
            self._run_traci()
        except traci.exceptions.FatalTraCIError:
            print("[SumoRunner] SUMO window closed or simulation ended.")
            self._push_state({"warning": "Simulation ended."})
        except Exception as e:
            err_msg = f"SUMO crashed: {e}"
            print(f"[SumoRunner] Error: {err_msg}")
            traceback.print_exc()
            self._push_state({"error": err_msg})
        finally:
            self.running = False
            try:
                traci.close()
            except:
                pass

    def _load_ppo_model(self):
        model_path = os.path.join(REPO_ROOT, "models", f"ppo_traffic_{TLS_ID}_final")
        
        if not os.path.exists(model_path + ".zip"):
            print(f"[SumoRunner] PPO model not found at {model_path}. Falling back to fixed-timer.")
            return None, None
            
        try:
            from stable_baselines3 import PPO
            from rl_agent.traffic_env import TrafficEnv
            
            # Instantiate environment just to hold observation logic & LSTM
            env = TrafficEnv(junction_id=TLS_ID, sumo_cfg=SUMO_CFG, max_steps=5000)
            model = PPO.load(model_path, env=env)
            
            if self.condition == "maestro_fl" or self.condition == "ppo_fedprox":
                # Try to load fedprox params
                fedprox_params_path = os.path.join(REPO_ROOT, "global_params.pkl")
                if os.path.exists(fedprox_params_path):
                    with open(fedprox_params_path, "rb") as f:
                        global_params = pickle.load(f)
                    
                    processed_params = []
                    for i, p in enumerate(global_params):
                        if i in [0, 4] and p.shape == (64, 7):
                            p = np.hstack([p, np.zeros((64, 1), dtype=p.dtype)])
                        processed_params.append(p)
                        
                    from collections import OrderedDict
                    import torch
                    params_dict = zip(model.policy.state_dict().keys(), processed_params)
                    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                    model.policy.load_state_dict(state_dict, strict=True)
                    print("[SumoRunner] Loaded FedProx weights.")
            
            return model, env
        except Exception as e:
            print(f"[SumoRunner] Error loading PPO model: {e}")
            traceback.print_exc()
            return None, None

    def _push_state(self, state_dict):
        # Fire and forget into asyncio loop
        self.loop.call_soon_threadsafe(self.state_queue.put_nowait, state_dict)

    def _run_traci(self):
        traci.start([
            "sumo-gui",
            "-c", SUMO_CFG,
            "--start",
            "--delay", "150",
            "--no-step-log", "true",
            "--quit-on-end", "false",
        ])
        
        self.junction_positions = {}
        for tls in traci.trafficlight.getIDList():
            # Get a node associated with the tls
            # For simplicity, we just use the first controlled link's junction coordinate
            links = traci.trafficlight.getControlledLinks(tls)
            if links and links[0]:
                in_lane = links[0][0][0]
                edge_id = traci.lane.getEdgeID(in_lane)
                # get edge to-node position (which is the junction)
                to_node = traci.edge.getToJunction(edge_id)
                x, y = traci.junction.getPosition(to_node)
                lat, lon = sumo_to_latlon(x, y)
                self.junction_positions[tls] = {"lat": lat, "lon": lon}
            else:
                self.junction_positions[tls] = {"lat": 12.9165, "lon": 77.5831} # fallback
                
        step = 0
        ambulance_active = False
        ambulance_start_time = 0
        priority_active = False
        self._synced_junctions = set()
        self._route_junction_ids = []
        self._sync_schedule = {}
        
        model = None
        env_helper = None
        
        if self.condition in ["ppo_only", "maestro_fl", "ppo_fedprox"]:
            model, env_helper = self._load_ppo_model()
            if model is None:
                self._push_state({"warning": "PPO model not found, running fixed-timer fallback."})
                self.condition = "fixed_timer"

        # FL state dummy data
        fl_state = {
            "round": 0,
            "last_aggregation_step": 0,
            "clients_active": 3
        }

        while self.running:
            if self.reset_requested:
                break
                
            if self.paused:
                time.sleep(0.1)
                continue
                
            # Handle Ambulance Injection
            if self.inject_requested and not ambulance_active:
                self.inject_requested = False
                try:
                    # Dynamically find route using SUMO's router
                    route_result = traci.simulation.findRoute(AMB_FROM_EDGE, AMB_TO_EDGE)
                    if not route_result.edges:
                        print(f"[SumoRunner] findRoute({AMB_FROM_EDGE} -> {AMB_TO_EDGE}) returned empty. Ambulance not injected.")
                    else:
                        route_edges = list(route_result.edges)
                        print(f"[SumoRunner] Ambulance route: {len(route_edges)} edges")

                        if "amb_demo_route" not in traci.route.getIDList():
                            traci.route.add("amb_demo_route", route_edges)

                        traci.vehicle.add(
                            vehID="ambulance_1",
                            routeID="amb_demo_route",
                            typeID="emergency",
                            depart="now"
                        )
                        traci.vehicle.setColor("ambulance_1", (255, 0, 0, 255))
                        traci.vehicle.setVehicleClass("ambulance_1", "emergency")

                        if self.condition == "maestro_fl":
                            traci.vehicle.setSpeedMode("ambulance_1", 7)
                            priority_active = True
                            if env_helper:
                                env_helper.priority_active = True
                                env_helper.priority_urgency = 1.0

                        # Track ambulance in SUMO-GUI so it's visible
                        try:
                            traci.gui.trackVehicle("View #0", "ambulance_1")
                            traci.gui.setZoom("View #0", 3000)
                        except traci.exceptions.TraCIException:
                            print("[SumoRunner] Warning: camera tracking failed (non-fatal)")

                        ambulance_active = True
                        ambulance_start_time = step

                        fl_state["last_aggregation_step"] = step
                        fl_state["round"] += 1

                        # --- Emit ambulance_detected event ---
                        route_coords = []
                        for edge_id in route_edges:
                            try:
                                shape = traci.edge.getShape(edge_id)
                                for pt in shape:
                                    lat, lon = sumo_to_latlon(pt[0], pt[1])
                                    route_coords.append([lat, lon])
                            except Exception:
                                pass

                        # Compute destination lat/lon from last edge
                        dest_lat, dest_lon = route_coords[-1] if route_coords else [12.9165, 77.5831]

                        self._push_state({
                            "event": "ambulance_detected",
                            "route_edges": route_edges,
                            "route_coords": route_coords,
                            "from": AMB_FROM_EDGE,
                            "to": AMB_TO_EDGE,
                            "dest_lat": dest_lat,
                            "dest_lon": dest_lon,
                        })

                        # --- Build staggered priority_sync schedule ---
                        self._synced_junctions = set()
                        self._route_junction_ids = []
                        all_tls = set(self.junction_positions.keys())
                        for edge_id in route_edges:
                            try:
                                to_node = traci.edge.getToJunction(edge_id)
                                # Check if to_node matches a TLS ID or is part of a TLS cluster
                                for tls_id in all_tls:
                                    if to_node in tls_id or tls_id == to_node:
                                        if tls_id not in self._route_junction_ids:
                                            self._route_junction_ids.append(tls_id)
                            except Exception:
                                pass

                        # Schedule syncs: stagger by ~15 steps apart from injection
                        self._sync_schedule = {}
                        for i, jid in enumerate(self._route_junction_ids):
                            self._sync_schedule[jid] = step + 10 + (i * 15)
                            print(f"[SumoRunner] Scheduled priority_sync for {jid[:20]}... at step {self._sync_schedule[jid]}")
                except Exception as e:
                    print(f"[SumoRunner] Could not inject ambulance: {e}")

            # PPO Actions
            if model and env_helper and step % 5 == 0:
                # Re-construct observation manually
                env_helper._controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(TLS_ID)))
                
                # Update queue history
                total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in env_helper._controlled_lanes)
                env_helper.queue_history.append(total_queue)
                
                obs = env_helper._get_obs()
                action, _ = model.predict(obs, deterministic=True)
                
                phase_duration = traci.trafficlight.getPhaseDuration(TLS_ID)
                if phase_duration < 50: # Avoid overriding priority mask locks
                    traci.trafficlight.setPhase(TLS_ID, action)
            
            # Priority Override
            if ambulance_active and "ambulance_1" in traci.vehicle.getIDList():
                if self.condition == "maestro_fl":
                    force_green_along_route("ambulance_1", lookahead=5)
                    release_green_lock("ambulance_1")

                # --- Fire staggered priority_sync events ---
                for jid, fire_step in list(self._sync_schedule.items()):
                    if step >= fire_step and jid not in self._synced_junctions:
                        self._synced_junctions.add(jid)
                        # Compute queue_before for this junction
                        q_before = 0
                        try:
                            lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(jid)))
                            q_before = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
                        except Exception:
                            pass
                        self._push_state({
                            "event": "priority_sync",
                            "junction_id": jid,
                            "timestamp": step,
                            "queue_before": q_before,
                        })
                        print(f"[SumoRunner] priority_sync fired for {jid[:20]}... (queue={q_before})")

            elif ambulance_active:
                ambulance_active = False
                priority_active = False
                self._synced_junctions = set()
                self._sync_schedule = {}
                if env_helper:
                    env_helper.priority_active = False

            traci.simulationStep()
            step += 1
            
            # Build State Payload
            state_payload = {
                "step": step,
                "condition": self.condition,
                "priority_active": priority_active,
                "ambulance": {"active": False},
                "junctions": [],
                "metrics": {},
                "fl": fl_state
            }
            
            if ambulance_active and "ambulance_1" in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition("ambulance_1")
                lat, lon = sumo_to_latlon(x, y)
                speed = traci.vehicle.getSpeed("ambulance_1")
                wait_time = traci.vehicle.getWaitingTime("ambulance_1")
                state_payload["ambulance"] = {
                    "active": True,
                    "lat": lat,
                    "lon": lon,
                    "speed": speed,
                    "travel_time": step - ambulance_start_time,
                    "waiting_time": wait_time
                }
                
            total_wait = 0
            total_queue = 0
            num_lanes = 0
            
            for tls in self.junction_positions.keys():
                lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls)))
                j_wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
                j_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
                phase = traci.trafficlight.getPhase(tls)
                
                total_wait += j_wait
                total_queue += j_queue
                num_lanes += len(lanes)
                
                state_payload["junctions"].append({
                    "id": tls,
                    "lat": self.junction_positions[tls]["lat"],
                    "lon": self.junction_positions[tls]["lon"],
                    "queue_length": j_queue,
                    "waiting_time": j_wait,
                    "current_phase": phase,
                    "priority_override": priority_active and tls == TLS_ID
                })
                
            state_payload["metrics"] = {
                "avg_waiting_time": round(total_wait / max(num_lanes, 1), 2),
                "avg_queue_length": round(total_queue / max(num_lanes, 1), 2),
                "throughput": traci.simulation.getArrivedNumber()
            }

            self._push_state(state_payload)
            
            time.sleep(0.15) # ~7 FPS, slow enough for humans to follow
