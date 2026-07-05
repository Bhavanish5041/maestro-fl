import traci
import os

SUMO_CFG = "sumo_env/network/osm.sumocfg"
sumo_cmd = ["sumo", "-c", os.path.abspath(SUMO_CFG), "--no-step-log", "true"]
traci.start(sumo_cmd)

tls_ids = traci.trafficlight.getIDList()
if tls_ids:
    tls_id = tls_ids[0]
    controlled = traci.trafficlight.getControlledLanes(tls_id)
    xs, ys = [], []
    for lane in controlled:
        shape = traci.lane.getShape(lane)
        if shape:
            xs.append(shape[-1][0])
            ys.append(shape[-1][1])
    jx, jy = sum(xs)/len(xs), sum(ys)/len(ys)
    print(f"Junction {tls_id} position: ({jx:.2f}, {jy:.2f})")
    
    # Get overall bounds
    (xmin, ymin), (xmax, ymax) = traci.simulation.getNetBoundary()
    print(f"Boundary: xmin={xmin:.2f}, ymin={ymin:.2f}, xmax={xmax:.2f}, ymax={ymax:.2f}")
    
traci.close()
