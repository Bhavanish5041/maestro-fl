"""
MAESTRO-FL Live Demo
---------------------
Runs a watchable SUMO-GUI simulation:
  - Normal traffic runs first so the audience sees baseline congestion.
  - An ambulance is injected on a dynamically discovered route (crosses
    the first traffic light junction) after a short delay.
  - The priority override forces green lights along its upcoming path
    and releases them once it has passed each junction.

Run from repo root:
    python3 live_demo.py
"""

import sys
import os
import traci

# --- Path setup: make rl_agent importable regardless of where this is run from ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(REPO_ROOT, "rl_agent"))

from priority_mask import (  # noqa: E402
    force_green_along_route,
    release_green_lock,
    maybe_use_wrong_side,
    restore_normal_driving,
)

import argparse

# --- Config Defaults ---
SUMO_CFG = os.path.join(REPO_ROOT, "sumo_env", "network", "osm.sumocfg")
AMBULANCE_DEPART_STEP = 50      # let normal traffic build up first
STEP_DELAY_MS = 100             # GUI playback delay; raise for slower/more watchable
ZOOM_LEVEL = 3000
LOOKAHEAD_JUNCTIONS = 4         # how many junctions ahead to pre-green
WRONG_SIDE_WAIT = 8.0           # seconds blocked before trying wrong side


def discover_ambulance_route(start_edge=None, end_edge=None):
    """
    Find a valid ambulance route. If start_edge and end_edge are provided,
    calculates a route between them. Otherwise, dynamically finds a route through the junction.
    """
    # Custom start and end destinations provided by the user
    if start_edge and end_edge:
        try:
            route = traci.simulation.findRoute(start_edge, end_edge)
            if route.edges:
                print(f"[DEMO] Successfully routed from custom start '{start_edge}' to end '{end_edge}' ({len(route.edges)} edges)")
                return list(route.edges)
            else:
                print(f"[DEMO] Warning: findRoute returned empty route between '{start_edge}' and '{end_edge}'. Falling back to discovery.")
        except Exception as e:
            print(f"[DEMO] Error routing between '{start_edge}' and '{end_edge}': {e}. Falling back to discovery.")

    tls_list = traci.trafficlight.getIDList()
    if not tls_list:
        print("[DEMO] No traffic lights found!")
        return None

    tls_id = tls_list[0]
    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
    tls_edges = list(set(lane.rsplit("_", 1)[0] for lane in controlled_lanes))

    if not tls_edges:
        print("[DEMO] No edges feed the traffic light!")
        return None

    # Strategy 1: Copy a route from an existing vehicle that passes through the TLS
    for veh_id in traci.vehicle.getIDList():
        try:
            route = traci.vehicle.getRoute(veh_id)
            for i, edge in enumerate(route):
                if edge in tls_edges and i >= 2 and i + 3 < len(route):
                    return route[max(0, i - 3): min(len(route), i + 4)]
        except Exception:
            continue

    # Strategy 2: Use SUMO's built-in router
    import random
    all_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    target_edge = tls_edges[0]
    for _ in range(100):
        start = random.choice(all_edges)
        route = traci.simulation.findRoute(start, target_edge)
        if route.edges and len(route.edges) > 3:
            return list(route.edges)

    print("[DEMO] Could not find a valid route through the junction!")
    return None


def inject_ambulance(start_edge=None, end_edge=None):
    route_edges = discover_ambulance_route(start_edge, end_edge)
    if not route_edges:
        return False

    traci.route.add("amb_demo_route", route_edges)
    traci.vehicle.add(
        vehID="ambulance_1",
        routeID="amb_demo_route",
        typeID="emergency",
        depart=traci.simulation.getTime(),
    )
    traci.vehicle.setColor("ambulance_1", (255, 0, 0, 255))
    traci.vehicle.setVehicleClass("ambulance_1", "emergency")
    print(f"[DEMO] Ambulance injected. Route: {len(route_edges)} edges")
    return True


def run_live_demo(start_edge=None, end_edge=None, max_steps=1000):
    traci.start([
        "sumo-gui",
        "-c", SUMO_CFG,
        "--start",
        "--delay", str(STEP_DELAY_MS),
        "--quit-on-end", "false",
    ])

    ambulance_injected = False
    ambulance_spawned = False
    ambulance_done = False
    step = 0

    try:
        while step < max_steps:
            if traci.simulation.getMinExpectedNumber() <= 0 and ambulance_injected:
                # Nothing left to simulate and ambulance already ran its course
                break

            if not ambulance_injected and step >= AMBULANCE_DEPART_STEP:
                if inject_ambulance(start_edge, end_edge):
                    ambulance_injected = True
                    try:
                        traci.gui.trackVehicle("View #0", "ambulance_1")
                        traci.gui.setZoom("View #0", ZOOM_LEVEL)
                    except traci.exceptions.TraCIException:
                        print("[DEMO] Warning: camera tracking failed (non-fatal)")
                else:
                    print("[DEMO] Could not inject ambulance — continuing without it")
                    ambulance_injected = True  # don't retry every step

            if "ambulance_1" in traci.vehicle.getIDList():
                ambulance_spawned = True
                # 1. Force green lights ahead of ambulance
                force_green_along_route("ambulance_1", lookahead=LOOKAHEAD_JUNCTIONS)
                release_green_lock("ambulance_1")
                # 2. If stuck at red / blocked traffic → try wrong side of road
                maybe_use_wrong_side("ambulance_1", wait_threshold=WRONG_SIDE_WAIT)
                # 3. Restore normal driving once it's moving again
                restore_normal_driving("ambulance_1")
            elif ambulance_spawned and not ambulance_done:
                print(f"[DEMO] Ambulance completed its route at step {step}")
                ambulance_done = True

            traci.simulationStep()
            step += 1

    finally:
        traci.close()
        print(f"[DEMO] Finished after {step} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAESTRO-FL Live Demo Options")
    parser.add_argument("--start", type=str, default=None, help="Initial edge ID for the ambulance route (e.g. '28087618#0')")
    parser.add_argument("--end", type=str, default=None, help="Final edge ID for the ambulance route (e.g. '172930638#4')")
    parser.add_argument("--steps", type=int, default=1500, help="Maximum simulation steps to run (default: 1500)")
    args = parser.parse_args()

    run_live_demo(start_edge=args.start, end_edge=args.end, max_steps=args.steps)
