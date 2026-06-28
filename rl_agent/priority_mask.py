"""
rl_agent/priority_mask.py
=========================
Rule-based logic to force green lights along an emergency vehicle's route.
This serves as the robust baseline override, directly integrated with the 
FL out-of-cycle push broadcast.
"""

import traci

def get_green_phase_for_edge(junction_id: str, incoming_edge: str, outgoing_edge: str = None) -> int:
    """
    Finds the phase that gives a green light to the specific connection 
    from incoming_edge to outgoing_edge, avoiding shared-lane deadlocks.
    """
    controlled_links = traci.trafficlight.getControlledLinks(junction_id)
    program = traci.trafficlight.getAllProgramLogics(junction_id)[0]

    best_phase = None
    best_score = -1

    for phase_idx, phase in enumerate(program.phases):
        state = phase.state
        score = 0
        
        for li, link in enumerate(controlled_links):
            if not link:
                continue
            
            in_lane = link[0][0]
            out_lane = link[0][1]
            in_edge = in_lane.rsplit("_", 1)[0]
            out_edge = out_lane.rsplit("_", 1)[0]
            
            if li < len(state) and state[li] in ("G", "g"):
                if in_edge == incoming_edge:
                    if outgoing_edge and out_edge == outgoing_edge:
                        score += 100  # Exact match for the route connection
                    else:
                        score += 1    # Belongs to the edge but maybe wrong turn

        if score > best_score:
            best_score = score
            best_phase = phase_idx

    return best_phase if best_score > 0 else None


def force_green_along_route(amb_id="ambulance_1", lookahead=3):
    """
    Forces every traffic light on the ambulance's upcoming route to
    immediately switch to the phase that gives green to its approach.
    """
    if amb_id not in traci.vehicle.getIDList():
        return
        
    route = traci.vehicle.getRoute(amb_id)
    current_idx = traci.vehicle.getRouteIndex(amb_id)
    
    # We only care about the *upcoming* edges
    upcoming_edges = route[current_idx:current_idx + lookahead]

    for i, edge in enumerate(upcoming_edges):
        # Determine the next edge on the route to resolve split-phase connections
        outgoing_edge = None
        # Check if there is a next edge in the upcoming slice, else check the full route
        route_pos = current_idx + i + 1
        if route_pos < len(route):
            outgoing_edge = route[route_pos]
            
        # Check if the edge leads to a junction
        junction = traci.edge.getToJunction(edge)
        
        tls_id = None
        for t_id in traci.trafficlight.getIDList():
            if t_id == junction or t_id == f"GS_{junction}" or t_id == f"joinedS_{junction}":
                tls_id = t_id
                break
                
        if tls_id is None:
            continue

        # Find which phase gives green to vehicles coming FROM this edge
        green_phase = get_green_phase_for_edge(tls_id, edge, outgoing_edge)
        if green_phase is not None:
            # Check if it's already on this phase
            current_phase = traci.trafficlight.getPhase(tls_id)
            if current_phase != green_phase:
                traci.trafficlight.setPhase(tls_id, green_phase)
            
            # Hold green for up to 60s — long enough for ambulance to clear,
            # short enough to auto-recover and prevent catastrophic congestion buildup
            traci.trafficlight.setPhaseDuration(tls_id, 60)


def release_green_lock(amb_id="ambulance_1"):
    """
    Restores normal control to junctions the ambulance has already passed.
    Call this each step.
    """
    if amb_id not in traci.vehicle.getIDList():
        return

    route = traci.vehicle.getRoute(amb_id)
    current_idx = traci.vehicle.getRouteIndex(amb_id)
    passed_edges = route[:current_idx]

    for edge in passed_edges:
        junction = traci.edge.getToJunction(edge)
        tls_id = None
        for t_id in traci.trafficlight.getIDList():
            if t_id == junction or t_id == f"GS_{junction}" or t_id == f"joinedS_{junction}":
                tls_id = t_id
                break
                
        if tls_id is not None:
            # Check if it's locked to 999
            if traci.trafficlight.getPhaseDuration(tls_id) > 50:
                # Reset to default program '0'
                traci.trafficlight.setProgram(tls_id, "0")
