import sys
import traci
from rl_agent.traffic_env import TrafficEnv

env = TrafficEnv(
    junction_id="GS_cluster_10123822790_11303526453_11303526454_248766831_#2more",
    sumo_cfg="sumo_env/network/osm.sumocfg",
    use_gui=False,
    delta_time=1
)
env.reset()

entry_edge = "1259589338#3"
exit_edge = "1222891448#0"
traci.route.add("amb_route", [entry_edge, exit_edge])
traci.vehicle.add(vehID="ambulance_1", routeID="amb_route", typeID="emergency", depart="now")

traci.simulationStep()
traci.simulationStep()
amb_lane = traci.vehicle.getLaneID("ambulance_1")
print(f"Ambulance is in lane: {amb_lane}")

best_phase = env._priority_override_action()
print(f"Calculated best phase: {best_phase}")
env.close()
