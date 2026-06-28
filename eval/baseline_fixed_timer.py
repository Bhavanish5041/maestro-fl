"""
eval/baseline_fixed_timer.py — Fixed-Timer Baseline
====================================================
Runs SUMO with a fixed-cycle traffic light (no intelligence) to establish
the baseline metrics for comparison against PPO, PPO+FedProx, and the
full MAESTRO-FL system.

The fixed timer ignores traffic state entirely — it just cycles through
phases at a constant rate. This is the "what happens without our system"
comparison that makes the results section compelling.
"""

import os
import sys
from typing import Dict, List, Optional

try:
    import traci
except ImportError:
    traci = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_fixed_timer_baseline(
    sumo_cfg: str,
    junction_id: str,
    cycle_time: int = 30,
    n_phases: int = 4,
    max_steps: int = 5000,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Run a fixed-timer baseline simulation.

    The traffic light cycles through phases at a constant rate,
    completely ignoring actual traffic conditions.

    Args:
        sumo_cfg: Path to SUMO configuration file.
        junction_id: Junction to control with fixed timer.
        cycle_time: Seconds per phase before switching.
        n_phases: Number of signal phases.
        max_steps: Maximum simulation steps.
        seed: Random seed for reproducibility.

    Returns:
        Dict with lists of per-step metrics:
            - waiting_time: total waiting time across all lanes
            - queue_length: total halting vehicles across all lanes
            - vehicle_count: total vehicles on controlled lanes
            - phase: active phase at each step
            - throughput: vehicles that completed their trip so far
    """
    sumo_cmd = [
        "sumo",
        "-c", os.path.abspath(sumo_cfg),
        "--seed", str(seed),
        "--no-step-log", "true",
    ]
    traci.start(sumo_cmd)

    metrics = {
        "waiting_time": [],
        "queue_length": [],
        "vehicle_count": [],
        "phase": [],
        "throughput": [],
    }

    step = 0
    try:
        while (
            traci.simulation.getMinExpectedNumber() > 0
            and step < max_steps
        ):
            # Fixed cycle: ignore real traffic state entirely
            phase = (step // cycle_time) % n_phases
            traci.trafficlight.setPhase(junction_id, phase)
            traci.simulationStep()

            # Collect metrics
            lanes = list(
                dict.fromkeys(
                    traci.trafficlight.getControlledLanes(junction_id)
                )
            )

            total_wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
            total_queue = sum(
                traci.lane.getLastStepHaltingNumber(l) for l in lanes
            )
            total_vehicles = sum(
                traci.lane.getLastStepVehicleNumber(l) for l in lanes
            )
            arrived = traci.simulation.getArrivedNumber()

            metrics["waiting_time"].append(total_wait)
            metrics["queue_length"].append(total_queue)
            metrics["vehicle_count"].append(total_vehicles)
            metrics["phase"].append(phase)
            metrics["throughput"].append(arrived)

            step += 1

    finally:
        traci.close()

    print(
        f"[BASELINE] Fixed-timer simulation complete — {step} steps, "
        f"avg wait: {sum(metrics['waiting_time'])/max(step,1):.2f}, "
        f"avg queue: {sum(metrics['queue_length'])/max(step,1):.2f}"
    )

    return metrics


def run_emergency_baseline(
    sumo_cfg: str,
    junction_id: str,
    ambulance_route_id: str,
    ambulance_depart: float = 100.0,
    cycle_time: int = 30,
    seed: int = 42,
) -> Dict[str, any]:
    """
    Run fixed-timer baseline WITH an ambulance injected.

    The fixed timer does NOT give the ambulance any priority — this
    demonstrates the problem our system solves.

    Returns:
        Dict with per-step metrics PLUS ambulance-specific metrics:
            - ambulance_travel_time: total time ambulance spent in simulation
            - ambulance_waiting_time: time ambulance spent stopped
    """
    sumo_cmd = [
        "sumo",
        "-c", os.path.abspath(sumo_cfg),
        "--seed", str(seed),
        "--no-step-log", "true",
    ]
    traci.start(sumo_cmd)

    metrics = {
        "waiting_time": [],
        "queue_length": [],
        "ambulance_travel_time": None,
        "ambulance_waiting_time": 0.0,
    }

    ambulance_injected = False
    ambulance_start_time = None
    step = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            sim_time = traci.simulation.getTime()

            if not ambulance_injected and sim_time >= ambulance_depart:
                try:
                    if "amb_route" not in traci.route.getIDList():
                        import random
                        edges = traci.edge.getIDList()
                        route_edges = []
                        controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)
                        tls_edges = list(set(l.rsplit('_', 1)[0] for l in controlled_lanes))
                        
                        if tls_edges:
                            target = tls_edges[0]
                            for _ in range(50):
                                start = random.choice(edges)
                                route = traci.simulation.findRoute(start, target)
                                if route.edges and len(route.edges) > 3:
                                    route_edges = list(route.edges)
                                    break
                        
                        if not route_edges:
                            entry_edge = "1259589338#3"
                            exit_edge = "1222891448#0"
                            route_edges = [entry_edge, exit_edge]
                            
                        traci.route.add("amb_route", route_edges)
                        
                    traci.vehicle.add(
                        vehID="ambulance_1",
                        routeID="amb_route",
                        typeID="emergency",
                        depart="now",
                        departPos="0"
                    )
                    traci.vehicle.setVehicleClass("ambulance_1", "emergency")
                    ambulance_injected = True
                    ambulance_start_time = sim_time
                    print(f"[BASELINE] Ambulance injected at t={sim_time}s on route {traci.route.getEdges('amb_route')}")
                except Exception as e:
                    print(f"[BASELINE] Could not inject ambulance: {e}")

            # Fixed timer
            phase = (step // cycle_time) % 4
            traci.trafficlight.setPhase(junction_id, phase)
            traci.simulationStep()

            # Collect metrics
            lanes = list(dict.fromkeys(
                traci.trafficlight.getControlledLanes(junction_id)
            ))
            metrics["waiting_time"].append(
                sum(traci.lane.getWaitingTime(l) for l in lanes)
            )
            metrics["queue_length"].append(
                sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            )

            # Track ambulance
            if ambulance_injected and "ambulance_1" in traci.vehicle.getIDList():
                speed = traci.vehicle.getSpeed("ambulance_1")
                if speed < 0.1:
                    metrics["ambulance_waiting_time"] += 1.0
            elif ambulance_injected and ambulance_start_time is not None:
                if metrics["ambulance_travel_time"] is None:
                    metrics["ambulance_travel_time"] = sim_time - ambulance_start_time
                    print(
                        f"[BASELINE] Ambulance completed — "
                        f"travel time: {metrics['ambulance_travel_time']:.1f}s"
                    )

            step += 1

    finally:
        traci.close()

    return metrics
