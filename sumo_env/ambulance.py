"""
sumo_env/ambulance.py — Ambulance Injection & Route Lookahead
=============================================================
Handles emergency vehicle lifecycle in SUMO via TraCI:
  - Injecting an ambulance with the 'emergency' vType
  - Route lookahead to find upcoming junctions (feeds priority broadcast)
  - Proximity-based detection of ambulance near junctions
"""

import sys
import math
from typing import List, Tuple, Optional

try:
    import traci
except ImportError:
    print("WARNING: traci not found. Install SUMO and ensure traci is on PYTHONPATH.")
    traci = None

# Add project root to path for shared imports
sys.path.insert(0, "..")
from shared.schema import make_priority_message


# ---------------------------------------------------------------------------
# Ambulance injection
# ---------------------------------------------------------------------------

def inject_ambulance(
    route_id: str,
    depart_time: float,
    veh_id: str = "ambulance_1",
) -> None:
    """
    Inject an emergency vehicle into the running SUMO simulation.

    Prerequisites:
        The route file (.rou.xml) must define a vType with id="emergency":
        <vType id="emergency" vClass="emergency" color="1,0,0"
               guiShape="emergency" maxSpeed="25"/>

    Args:
        route_id: ID of a route already defined in the route file.
        depart_time: Simulation time (seconds) when the vehicle should depart.
        veh_id: Unique vehicle ID (default "ambulance_1").
    """
    traci.vehicle.add(
        vehID=veh_id,
        routeID=route_id,
        typeID="emergency",
        depart=depart_time,
    )
    traci.vehicle.setVehicleClass(veh_id, "emergency")
    print(f"[AMBULANCE] Injected {veh_id} on route {route_id} at t={depart_time}")


# ---------------------------------------------------------------------------
# Route lookahead — feeds the priority broadcast
# ---------------------------------------------------------------------------

def get_upcoming_tls(veh_id: str, n_ahead: int = 4) -> List[Tuple[str, str]]:
    """
    Look ahead along the vehicle's route and return the next N Traffic Light junctions
    and the specific target edge the ambulance is approaching them from.

    Args:
        veh_id: SUMO vehicle ID (must be currently in simulation).
        n_ahead: Number of edges to look ahead.

    Returns:
        List of (junction_id, target_edge) tuples.
    """
    try:
        route = traci.vehicle.getRoute(veh_id)
        current_road_idx = traci.vehicle.getRouteIndex(veh_id)
        upcoming_edges = route[current_road_idx : current_road_idx + n_ahead]
        
        tls_targets = []
        # Check all traffic lights
        for tls_id in traci.trafficlight.getIDList():
            controlled = traci.trafficlight.getControlledLanes(tls_id)
            if not controlled:
                continue
                
            tls_edges = set(lane.rsplit("_", 1)[0] for lane in controlled)
            
            # Find the first edge in upcoming_edges that feeds this TLS
            for edge in upcoming_edges:
                if edge in tls_edges:
                    # Avoid duplicates if multiple edges feed the same TLS
                    if not any(j == tls_id for j, _ in tls_targets):
                        tls_targets.append((tls_id, edge))
                    break
        return tls_targets
    except traci.TraCIException:
        return []


def build_priority_broadcast(
    veh_id: str,
    sim_time: float,
    n_ahead: int = 4,
    ttl_seconds: float = 30.0,
) -> Optional[dict]:
    """
    Build a PRIORITY_MESSAGE for the given emergency vehicle.

    Args:
        veh_id: SUMO vehicle ID of the emergency vehicle.
        sim_time: Current simulation time.
        n_ahead: Number of edges to look ahead.
        ttl_seconds: How long the priority flag should stay active.

    Returns:
        A PRIORITY_MESSAGE dict, or None if no upcoming TLS junctions found.
    """
    tls_targets = get_upcoming_tls(veh_id, n_ahead)
    if not tls_targets:
        return None

    # We just send the junction IDs in the standard schema for now.
    # The environment will separately query the target edge when applying the mask.
    junctions = [tls_id for tls_id, _ in tls_targets]

    # Urgency: inverse of speed (slower = more urgent, likely stuck in traffic)
    speed = traci.vehicle.getSpeed(veh_id)
    urgency = 1.0 / max(speed, 0.1)

    return make_priority_message(
        junction_ids=junctions,
        urgency=urgency,
        timestamp=sim_time,
        ttl_seconds=ttl_seconds,
    )
