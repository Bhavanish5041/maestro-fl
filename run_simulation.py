"""
run_simulation.py — Run SUMO with logging + ambulance injection
===============================================================
Produces real traffic logs for LSTM training and demonstrates
the ambulance injection + priority broadcast pipeline.

Usage:
    python run_simulation.py
    python run_simulation.py --gui    # watch it visually
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))

import traci
from sumo_env.logger import TrafficLogger
from sumo_env.ambulance import inject_ambulance, check_emergency_events, build_priority_broadcast

SUMO_CFG = "sumo_env/network/osm.sumocfg"
JUNCTION_ID = "GS_cluster_11197334454_11197334455_11197334456_11197334457_#9more"
LOG_PATH = "sumo_env/logs/traffic_log.csv"


def run(use_gui=False, inject_ambulance_at=100.0):
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", os.path.abspath(SUMO_CFG),
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
        "--seed", "42",
    ]

    traci.start(sumo_cmd)
    print(f"[SIM] SUMO started — logging to {LOG_PATH}")
    if use_gui:
        print(f"[SIM] ⏳ Set Delay slider to ~50ms, then press Play in the GUI")
        print(f"[SIM] 🚑 Ambulance will appear at t={inject_ambulance_at:.0f}s — watch for bright RED vehicle!")

    logger = TrafficLogger(LOG_PATH, junction_ids=[JUNCTION_ID])

    ambulance_injected = False
    ambulance_done = False
    step = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            sim_time = traci.simulation.getTime()

            # --- Inject ambulance ---
            if not ambulance_injected and sim_time >= inject_ambulance_at:
                vehicles = traci.vehicle.getIDList()
                # Pick the vehicle with the longest remaining route
                best_veh = None
                best_remaining = 0
                for v in vehicles:
                    try:
                        route = traci.vehicle.getRoute(v)
                        idx = traci.vehicle.getRouteIndex(v)
                        remaining = len(route) - idx
                        if remaining > best_remaining:
                            best_remaining = remaining
                            best_veh = v
                    except Exception:
                        continue

                if best_veh:
                    route_edges = traci.vehicle.getRoute(best_veh)
                    try:
                        traci.route.add("ambulance_route", route_edges)
                        # Add the vehicle manually with emergency type
                        traci.vehicle.add(
                            vehID="ambulance_1",
                            routeID="ambulance_route",
                            typeID="emergency",
                            depart="now",
                        )
                        # Force bright red color and emergency class
                        traci.vehicle.setColor("ambulance_1", (255, 0, 0, 255))
                        traci.vehicle.setVehicleClass("ambulance_1", "emergency")
                        traci.vehicle.setSpeedMode("ambulance_1", 0)  # ignore right-of-way
                        ambulance_injected = True
                        print(f"\n[SIM] 🚑🚑🚑 AMBULANCE INJECTED at t={sim_time:.0f}s!")
                        print(f"[SIM]   Route: {len(route_edges)} edges (copied from {best_veh})")
                        print(f"[SIM]   Look for the BRIGHT RED vehicle on the map!\n")
                    except Exception as e:
                        print(f"[SIM] Route injection failed: {e}")

            # --- Check for emergency events ---
            if ambulance_injected and not ambulance_done:
                # Keep forcing the color every few steps (SUMO sometimes resets it)
                if step % 10 == 0 and "ambulance_1" in traci.vehicle.getIDList():
                    traci.vehicle.setColor("ambulance_1", (255, 0, 0, 255))

                events = check_emergency_events(junction_radius=100)
                for event_type, junc_id, veh_id in events:
                    msg = build_priority_broadcast(veh_id, sim_time)
                    if msg:
                        print(f"[SIM] 🚨 PRIORITY BROADCAST: {msg['junction_ids']} "
                              f"urgency={msg['urgency']:.2f}")

                # Check if ambulance finished its route
                if "ambulance_1" not in traci.vehicle.getIDList():
                    print(f"[SIM] ✅ Ambulance completed at t={sim_time:.1f}s "
                          f"(travel time: {sim_time - inject_ambulance_at:.1f}s)")
                    ambulance_done = True

            # --- Log traffic data ---
            logger.log_step(sim_time)
            traci.simulationStep()
            step += 1

            # Progress update every 500 steps
            if step % 500 == 0:
                status = ""
                if ambulance_injected and not ambulance_done:
                    status = " | 🚑 AMBULANCE ACTIVE"
                print(f"[SIM] Step {step}, t={sim_time:.0f}s, "
                      f"vehicles={traci.simulation.getMinExpectedNumber()}{status}")

    except KeyboardInterrupt:
        print("\n[SIM] Interrupted by user.")
    finally:
        logger.close()
        traci.close()

    print(f"\n[SIM] Done! {step} steps simulated.")
    print(f"[SIM] Traffic log saved to: {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SUMO simulation with logging")
    parser.add_argument("--gui", action="store_true", help="Open SUMO GUI")
    parser.add_argument("--ambulance-at", type=float, default=100.0,
                        help="Inject ambulance at this simulation time (seconds)")
    args = parser.parse_args()
    run(use_gui=args.gui, inject_ambulance_at=args.ambulance_at)

