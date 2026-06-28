"""
MAESTRO-FL Live Demo
---------------------
Runs a watchable SUMO-GUI simulation:
  - Normal traffic runs first so the audience sees baseline congestion.
  - An ambulance is injected on a confirmed route (crosses the fixed
    roundabout junction) after a short delay.
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

from priority_mask import force_green_along_route, release_green_lock  # noqa: E402

# --- Config ---
SUMO_CFG = os.path.join(REPO_ROOT, "sumo_env", "network", "osm.sumocfg")  # must point at the FIXED network
ROUTE_FROM_EDGE = "40633859#8"
ROUTE_TO_EDGE = "799788864#5"
AMBULANCE_DEPART_STEP = 50      # let normal traffic build up first
STEP_DELAY_MS = 100             # GUI playback delay; raise for slower/more watchable
ZOOM_LEVEL = 800
LOOKAHEAD_JUNCTIONS = 5
MAX_STEPS = 1000                # safety cap so the demo can't run forever


def inject_ambulance():
    route = traci.simulation.findRoute(fromEdge=ROUTE_FROM_EDGE, toEdge=ROUTE_TO_EDGE)
    traci.route.add("amb_demo_route", route.edges)
    traci.vehicle.add(
        vehID="ambulance_1",
        routeID="amb_demo_route",
        typeID="emergency",
        depart=traci.simulation.getTime(),
    )
    print(f"[DEMO] Ambulance injected. Route: {len(route.edges)} edges, "
          f"expected travel time ~{route.travelTime:.0f}s")


def run_live_demo():
    traci.start([
        "sumo-gui",
        "-c", SUMO_CFG,
        "--start",
        "--delay", str(STEP_DELAY_MS),
        "--quit-on-end", "false",
    ])

    ambulance_injected = False
    ambulance_done = False
    step = 0

    try:
        while step < MAX_STEPS:
            if traci.simulation.getMinExpectedNumber() <= 0 and ambulance_injected:
                # Nothing left to simulate and ambulance already ran its course
                break

            if not ambulance_injected and step >= AMBULANCE_DEPART_STEP:
                inject_ambulance()
                ambulance_injected = True
                try:
                    traci.gui.trackVehicle("View #0", "ambulance_1")
                    traci.gui.setZoom("View #0", ZOOM_LEVEL)
                except traci.exceptions.TraCIException:
                    print("[DEMO] Warning: camera tracking failed (non-fatal)")

            if ambulance_injected and "ambulance_1" in traci.vehicle.getIDList():
                force_green_along_route("ambulance_1", lookahead=LOOKAHEAD_JUNCTIONS)
                release_green_lock("ambulance_1")
            elif ambulance_injected and not ambulance_done:
                print(f"[DEMO] Ambulance completed its route at step {step}")
                ambulance_done = True

            traci.simulationStep()
            step += 1

    finally:
        traci.close()
        print(f"[DEMO] Finished after {step} steps")


if __name__ == "__main__":
    run_live_demo()
