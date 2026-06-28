import time
import sys
import traci
from rl_agent.traffic_env import TrafficEnv

def main():
    # Detect TLS ID from our demo script to use here
    with open("demo_priority.py", "r") as f:
        # We know we can just grab one manually or start SUMO to find one
        pass
        
    print("Testing TrafficEnv override...")
    env = TrafficEnv(
        junction_id="GS_cluster_10123822790_11303526453_11303526454_248766831_#2more",
        sumo_cfg="sumo_env/network/osm.sumocfg",
        use_gui=False,
        delta_time=1
    )
    
    obs, info = env.reset()
    
    # Inject ambulance
    # We use the direct route from demo
    entry_edge = "1259589338#3"
    exit_edge = "1222891448#0"
    
    try:
        traci.route.add("amb_route", [entry_edge, exit_edge])
        traci.vehicle.add(
            vehID="ambulance_1",
            routeID="amb_route",
            typeID="emergency",
            depart="now",
        )
    except Exception as e:
        print(f"Failed to add ambulance: {e}")
        
    print(f"Ambulance injected. Activating priority.")
    env.set_priority(urgency=1.0, ttl=30.0)
    
    # Run a step with action=1 (which should be overridden)
    print("Agent requests phase 1")
    obs, reward, terminated, truncated, info = env.step(1)
    
    current_phase = traci.trafficlight.getPhase(env.junction_id)
    print(f"Actual phase applied by environment: {current_phase}")
    print(f"Priority active: {env.priority_active}")
    
    env.close()

if __name__ == "__main__":
    main()
