import traci
import sys

traci.start(["sumo", "-c", "sumo_env/network/osm.sumocfg", "--no-step-log", "true"])
edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
valid_route = None

for e1 in edges:
    outgoing1 = [e for e in edges if traci.edge.getFromJunction(e) == traci.edge.getToJunction(e1)]
    if outgoing1:
        e2 = outgoing1[0]
        outgoing2 = [e for e in edges if traci.edge.getFromJunction(e) == traci.edge.getToJunction(e2)]
        if outgoing2:
            e3 = outgoing2[0]
            valid_route = [e1, e2, e3]
            break

print(f"Valid route: {valid_route}")
traci.close()
