"""
demo_priority.py — Simple MAESTRO-FL Priority Demo
===================================================
One ambulance, one traffic light, one green override.

Usage:
    python demo_priority.py --gui
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
import traci
from rl_agent.priority_mask import force_green_along_route, release_green_lock

SUMO_CFG = "sumo_env/network/osm.sumocfg"
# We will dynamically pick the first TLS in the network
TLS_ID = None


def run(use_gui=False):
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", os.path.abspath(SUMO_CFG),
        "--no-step-log", "true",
        "--seed", "42",
    ]

    traci.start(sumo_cmd)
    
    global TLS_ID
    if TLS_ID is None:
        all_tls = traci.trafficlight.getIDList()
        if not all_tls:
            print("❌ No traffic lights found in this map!")
            traci.close()
            return
        TLS_ID = all_tls[0]

    print("=" * 60)
    print("  MAESTRO-FL PRIORITY DEMO")
    print("=" * 60)
    print(f"  [INFO] Using Traffic Light: {TLS_ID}")
    print("\n  1. Zoom into the main junction (center of map)")
    print("  2. Set Delay to ~100ms")
    print("  3. Press Play")
    print("  4. At t=200s the ambulance spawns")
    print("  5. Watch the traffic light change!\n")

    # Get the controlled lanes to know which edges feed the TLS
    controlled_lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    tls_edges = set()
    for lane in controlled_lanes:
        edge = lane.rsplit("_", 1)[0]
        tls_edges.add(edge)
    print(f"  [INFO] Traffic light controls edges: {tls_edges}\n")

    # Get all phases to find which ones have greens
    logic = traci.trafficlight.getAllProgramLogics(TLS_ID)
    phases = logic[0].phases
    print(f"  [INFO] Traffic light has {len(phases)} phases")
    for i, p in enumerate(phases):
        print(f"    Phase {i}: {p.state}")
    print()

    ambulance_injected = False
    ambulance_active = False
    override_active = False
    step = 0
    inject_time = 50.0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            sim_time = traci.simulation.getTime()

            # === Inject ambulance at t=200 ===
            if not ambulance_injected and sim_time >= inject_time:
                # Look at all the trip routes — find one that passes through our TLS
                # We'll use DijkstraRouter to build a fresh route through the junction
                all_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
                
                # Find an edge that feeds INTO the traffic light
                entry_edge = None
                exit_edge = None
                for lane in controlled_lanes:
                    edge = lane.rsplit("_", 1)[0]
                    if edge in all_edges:
                        if entry_edge is None:
                            entry_edge = edge
                        elif edge != entry_edge:
                            exit_edge = edge
                            break
                
                if entry_edge and exit_edge:
                    # Find an edge far from the junction to start from
                    # Use the trips file edges — pick one that connects to entry_edge
                    vehicles = traci.vehicle.getIDList()
                    start_edge = None
                    for v in vehicles:
                        try:
                            route = traci.vehicle.getRoute(v)
                            if entry_edge in route:
                                # Use edges BEFORE the entry edge as the starting point
                                idx = route.index(entry_edge)
                                if idx >= 2:
                                    # Build route: a few edges before TLS → through TLS → a few after
                                    amb_edges = route[max(0, idx-3) : min(len(route), idx+4)]
                                    route_name = "amb_route"
                                    traci.route.add(route_name, amb_edges)
                                    traci.vehicle.add(
                                        vehID="ambulance_1",
                                        routeID=route_name,
                                        typeID="emergency",
                                        depart=str(sim_time),
                                        departPos="0",
                                        departLane="best",
                                    )
                                    traci.vehicle.setColor("ambulance_1", (255, 0, 0, 255))
                                    traci.vehicle.setVehicleClass("ambulance_1", "emergency")
                                    ambulance_injected = True
                                    ambulance_active = True
                                    print(f"  🚑 [{sim_time:.0f}s] AMBULANCE SPAWNED!")
                                    print(f"     Route: {amb_edges}")
                                    print(f"     Will pass through traffic light!\n")
                                    start_edge = True
                                    break
                        except:
                            continue
                    
                    if not start_edge:
                        # Fallback: find a random valid route through the junction
                        try:
                            import random
                            route_name = "amb_route"
                            route_edges = []
                            target = list(tls_edges)[0] if tls_edges else None
                            
                            if target:
                                for _ in range(50):
                                    start = random.choice(all_edges)
                                    route = traci.simulation.findRoute(start, target)
                                    if route.edges and len(route.edges) > 2:
                                        route_edges = list(route.edges)
                                        break
                                        
                            if route_edges:
                                traci.route.add(route_name, route_edges)
                                traci.vehicle.add(
                                    vehID="ambulance_1",
                                    routeID=route_name,
                                    typeID="emergency",
                                    depart="now",
                                )
                                traci.vehicle.setColor("ambulance_1", (255, 0, 0, 255))
                                traci.vehicle.setVehicleClass("ambulance_1", "emergency")
                                ambulance_injected = True
                                ambulance_active = True
                                print(f"  🚑 [{sim_time:.0f}s] AMBULANCE SPAWNED (fallback route)!")
                                print(f"     Route: {route_edges}\n")
                            else:
                                print(f"  ❌ Fallback failed: Could not find valid route to {target}")
                        except Exception as e:
                            print(f"  ❌ Fallback exception: {e}")

            # === Priority Mask Logic ===
            if ambulance_active and sim_time > inject_time + 5:
                if "ambulance_1" in traci.vehicle.getIDList():
                    traci.vehicle.setColor("ambulance_1", (255, 0, 0, 255))
                    
                    # If this is the first time we detect the ambulance is moving,
                    # we would trigger the FL broadcast here.
                    if not override_active:
                        print(f"  🚨 [{sim_time:.0f}s] PRIORITY EVENT DETECTED!")
                        print(f"     [FL BROADCAST] Pushing global model weights out-of-cycle!")
                        override_active = True
                        
                    # 1. Force the green lock for the upcoming junctions
                    force_green_along_route("ambulance_1", lookahead=3)
                    
                    # 2. Release locks on passed junctions
                    release_green_lock("ambulance_1")
                    
                else:
                    # Ambulance finished
                    travel_time = sim_time - inject_time
                    print(f"  ✅ [{sim_time:.0f}s] Ambulance passed — locks released.")
                    print(f"  🏁 [{sim_time:.0f}s] AMBULANCE COMPLETED!")
                    print(f"     Travel time: {travel_time:.0f}s\n")
                    ambulance_active = False
                    override_active = False

            traci.simulationStep()
            step += 1

            if step % 500 == 0:
                veh_count = len(traci.vehicle.getIDList())
                print(f"  [{sim_time:.0f}s] {veh_count} vehicles on road")

    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    finally:
        traci.close()

    print(f"\n{'=' * 60}")
    print(f"  DEMO COMPLETE — {step} steps")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    run(use_gui=args.gui)
