import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import traci
except ImportError:
    traci = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.baseline_fixed_timer import run_emergency_baseline
from rl_agent.traffic_env import TrafficEnv

def run_priority_simulation(sumo_cfg: str, junction_id: str, ambulance_depart: float = 50.0):
    """Run the environment with the priority override active."""
    print("\n" + "=" * 50)
    print("RUNNING MAESTRO-FL PRIORITY OVERRIDE")
    print("=" * 50)
    
    env = TrafficEnv(
        junction_id=junction_id,
        sumo_cfg=sumo_cfg,
        use_gui=False,
        delta_time=1,
        max_steps=2000
    )
    
    env.reset()
    
    ambulance_injected = False
    ambulance_start_time = None
    ambulance_travel_time = None
    ambulance_wait_time = 0.0
    
    # We will use a fixed cycle as the "agent" action, but the environment will override it
    step_count = 0
    cycle_time = 30
    
    while True:
        sim_time = traci.simulation.getTime()
        
        # Inject ambulance
        if not ambulance_injected and sim_time >= ambulance_depart:
            try:
                if "amb_route" not in traci.route.getIDList():
                    import random
                    edges = traci.edge.getIDList()
                    route_edges = []
                    # Try to find a route that passes through the junction
                    controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)
                    tls_edges = list(set(l.rsplit('_', 1)[0] for l in controlled_lanes))
                    
                    if tls_edges:
                        target = tls_edges[0]
                        # Just find a route from a random edge to the target edge
                        for _ in range(50):
                            start = random.choice(edges)
                            route = traci.simulation.findRoute(start, target)
                            if route.edges and len(route.edges) > 3:
                                route_edges = list(route.edges)
                                break
                    
                    if not route_edges:
                        # Fallback
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
                print(f"[MAESTRO] Ambulance injected at t={sim_time}s on route {traci.route.getEdges('amb_route')}")
                
                # Activate priority in the environment!
                env.set_priority(urgency=1.0, ttl=300.0)
            except Exception as e:
                print(f"[MAESTRO] Failed to inject: {e}")

        # The agent just tries to run a fixed cycle, completely oblivious
        action = (step_count // cycle_time) % 4
        
        # The environment will override the action if priority is active
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track metrics
        if ambulance_injected and "ambulance_1" in traci.vehicle.getIDList():
            if traci.vehicle.getSpeed("ambulance_1") < 0.1:
                ambulance_wait_time += 1.0
        elif ambulance_injected and ambulance_start_time is not None and ambulance_travel_time is None:
            ambulance_travel_time = sim_time - ambulance_start_time
            print(f"[MAESTRO] Ambulance completed! Travel time: {ambulance_travel_time:.1f}s")
            # Clear priority
            env.clear_priority()
            
        step_count += 1
        
        if terminated or truncated or (ambulance_travel_time is not None and step_count > ambulance_start_time + ambulance_travel_time + 10):
            break
            
    env.close()
    
    return {
        "travel_time": ambulance_travel_time or 0.0,
        "wait_time": ambulance_wait_time
    }

def main():
    sumo_cfg = "sumo_env/network/osm.sumocfg"
    junction_id = "GS_cluster_10123822790_11303526453_11303526454_248766831_#2more"
    
    print("\n" + "=" * 50)
    print("RUNNING BASELINE (NO PRIORITY)")
    print("=" * 50)
    baseline_metrics = run_emergency_baseline(
        sumo_cfg=sumo_cfg,
        junction_id=junction_id,
        ambulance_route_id="amb_route",
        ambulance_depart=50.0,
        cycle_time=30
    )
    
    maestro_metrics = run_priority_simulation(
        sumo_cfg=sumo_cfg,
        junction_id=junction_id,
        ambulance_depart=50.0
    )
    
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    
    base_tt = baseline_metrics.get("ambulance_travel_time") or 0.0
    base_wait = baseline_metrics.get("ambulance_waiting_time") or 0.0
    
    maestro_tt = maestro_metrics["travel_time"]
    maestro_wait = maestro_metrics["wait_time"]
    
    print(f"BASELINE: Travel Time = {base_tt:.1f}s, Wait Time = {base_wait:.1f}s")
    print(f"MAESTRO : Travel Time = {maestro_tt:.1f}s, Wait Time = {maestro_wait:.1f}s")
    
    if base_tt > 0:
        improvement = ((base_tt - maestro_tt) / base_tt) * 100
        print(f"IMPROVEMENT: {improvement:.1f}% reduction in travel time!")
    
    # Plotting
    if plt:
        os.makedirs("eval/results/plots", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = ['Travel Time', 'Waiting Time']
        base_vals = [base_tt, base_wait]
        maestro_vals = [maestro_tt, maestro_wait]
        
        x = np.arange(len(labels))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline (No Priority)', color='#e74c3c')
        rects2 = ax.bar(x + width/2, maestro_vals, width, label='MAESTRO-FL', color='#2ecc71')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Emergency Vehicle Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}s',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
                            
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        plt.savefig("eval/results/plots/priority_comparison.png", dpi=150)
        print("\n[SUCCESS] Plot saved to eval/results/plots/priority_comparison.png")

if __name__ == "__main__":
    main()
