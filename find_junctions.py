"""Quick script to find all traffic light IDs in the SUMO network."""
import traci
import sys

sumo_cfg = "sumo_env/network/osm.sumocfg"
traci.start(["sumo", "-c", sumo_cfg, "--no-step-log", "true"])

tls_ids = traci.trafficlight.getIDList()
print(f"\nFound {len(tls_ids)} traffic light(s):\n")
for tls_id in tls_ids:
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    n_phases = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
    print(f"  ID: {tls_id}")
    print(f"      Lanes controlled: {len(set(lanes))}")
    print(f"      Phases: {n_phases}")
    print()

traci.close()
