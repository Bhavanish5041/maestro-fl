"""
rl_agent/priority_mask.py
=========================
Rule-based logic to force green lights along an emergency vehicle's route
and optionally reroute it onto the wrong side of the road when blocked.

HOW TRAFFIC LIGHTS WORK IN TRACI
---------------------------------
Each TLS (Traffic Light System) has:
  - A list of phases (e.g. "GGGrrr", "yyyrrr", "rrrGGG", ...)
    G = green, g = green (lower priority / yield), y = yellow, r = red
  - Each character in the phase string maps to one *link* (connection)
    as returned by traci.trafficlight.getControlledLinks(tls_id)
  - You can force a phase instantly: traci.trafficlight.setPhase(tls_id, idx)
  - You can lock it open:             traci.trafficlight.setPhaseDuration(tls_id, 60)
  - You can write a raw state string: traci.trafficlight.setRedYellowGreenState(tls_id, "GGGrrr...")

LAYER ARCHITECTURE
------------------
  Layer 1 (this file) — deterministic rule: ambulance approaching → force green
  Layer 2 (traffic_env.py) — PPO picks phase each step normally
  Layer 3 (federated/) — FedProx aggregates PPO weights across junctions

When an ambulance is active, Layer 1 overrides Layer 2 unconditionally.
The FL system (Layer 3) only changes the weights that Layer 2 uses — it
never sees Layer 1's override, which is intentional (out-of-cycle push).
"""

import traci

# ------------------------------------------------------------------
# Internal state: which junctions are currently locked green/red
# ------------------------------------------------------------------
_green_locked: dict[str, int] = {}   # tls_id -> phase index we forced
_red_locked:   set[str] = set()      # tls_ids where we forced cross-traffic red
_wrong_side_active: bool = False      # whether ambulance is on wrong side
_last_wrong_side_time: float = -999.0 # last time (in s) wrong-side was activated


# ------------------------------------------------------------------
# TLS lookup helpers
# ------------------------------------------------------------------

def _build_junction_to_tls_map() -> dict[str, str]:
    """
    Build a mapping from junction node ID → TLS ID.

    SUMO network quirk: TLS IDs for cluster junctions look like
    'GS_cluster_11303526465_13072877373_...' but the junction node IDs
    are just integers like '308729257'. We resolve this by checking
    which junction each TLS's controlled links actually feed into.
    """
    mapping = {}
    for tls_id in traci.trafficlight.getIDList():
        links = traci.trafficlight.getControlledLinks(tls_id)
        for link_group in links:
            if link_group:
                in_lane = link_group[0][0]
                try:
                    edge_id = traci.lane.getEdgeID(in_lane)
                    to_junc = traci.edge.getToJunction(edge_id)
                    mapping[to_junc] = tls_id
                except Exception:
                    pass
    return mapping


_junction_to_tls: dict[str, str] = {}   # populated lazily on first call


def _get_tls_for_edge(edge_id: str) -> str | None:
    """Return the TLS ID controlling the junction at the end of edge_id."""
    global _junction_to_tls
    if not _junction_to_tls:
        _junction_to_tls = _build_junction_to_tls_map()

    try:
        junc = traci.edge.getToJunction(edge_id)
        return _junction_to_tls.get(junc)
    except Exception:
        return None


# ------------------------------------------------------------------
# Phase-finding: which phase gives green to vehicles from incoming_edge?
# ------------------------------------------------------------------

def get_green_phase_for_edge(tls_id: str, incoming_edge: str,
                              outgoing_edge: str = None) -> int | None:
    """
    Scan all phases of a TLS and return the index of the one that gives
    a protected green (G, not just g) to vehicles coming from incoming_edge
    turning toward outgoing_edge (if specified).

    Returns None if no matching phase is found.
    """
    try:
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    except Exception:
        return None

    best_phase = None
    best_score  = -1

    for phase_idx, phase in enumerate(program.phases):
        state = phase.state
        # Skip transition phases (all-yellow, all-red)
        if all(c in ("y", "r", "u") for c in state):
            continue

        score = 0
        for li, link_group in enumerate(controlled_links):
            if not link_group:
                continue
            in_lane, out_lane, *_ = link_group[0]
            in_edge  = traci.lane.getEdgeID(in_lane)
            out_edge = traci.lane.getEdgeID(out_lane)

            if li < len(state) and state[li] in ("G", "g"):
                if in_edge == incoming_edge:
                    if outgoing_edge and out_edge == outgoing_edge:
                        score += 100   # exact turn match
                    elif not outgoing_edge:
                        score += 10    # any green for this approach
                    else:
                        score += 1     # right approach, wrong turn

        if score > best_score:
            best_score  = score
            best_phase  = phase_idx

    return best_phase if best_score > 0 else None


def get_all_red_state(tls_id: str) -> str:
    """Return an all-red phase state string of the correct length."""
    try:
        links = traci.trafficlight.getControlledLinks(tls_id)
        return "r" * len(links)
    except Exception:
        return "rrrrrrrr"


# ------------------------------------------------------------------
# Main API: force green along route
# ------------------------------------------------------------------

def force_green_along_route(amb_id: str = "ambulance_1", lookahead: int = 4):
    """
    For every TLS on the ambulance's upcoming route (up to `lookahead` edges):
      1. Force cross-traffic to red first (safety clearance)
      2. Switch to the green phase for the ambulance's approach
      3. Hold it for 60 s so the ambulance can clear

    Call this every simulation step while the ambulance is active.
    """
    global _junction_to_tls

    if amb_id not in traci.vehicle.getIDList():
        return

    route       = traci.vehicle.getRoute(amb_id)
    current_idx = traci.vehicle.getRouteIndex(amb_id)
    upcoming    = route[current_idx: current_idx + lookahead]

    for i, edge in enumerate(upcoming):
        tls_id = _get_tls_for_edge(edge)
        if tls_id is None:
            continue

        # Determine the next edge (which turn the ambulance intends)
        next_route_pos = current_idx + i + 1
        outgoing_edge  = route[next_route_pos] if next_route_pos < len(route) else None

        # (Removed all-red clearance step since it breaks SUMO phase indices 
        # when setRedYellowGreenState is used immediately before setPhase)

        # Step 2: find and set the correct green phase
        green_phase = get_green_phase_for_edge(tls_id, edge, outgoing_edge)
        if green_phase is not None:
            try:
                current_phase = traci.trafficlight.getPhase(tls_id)
                if current_phase != green_phase:
                    traci.trafficlight.setPhase(tls_id, green_phase)
                # Hold green for 60 s so the ambulance can pass
                traci.trafficlight.setPhaseDuration(tls_id, 60)
                _green_locked[tls_id] = green_phase
                _red_locked.discard(tls_id)
            except Exception:
                pass


def release_green_lock(amb_id: str = "ambulance_1"):
    """
    Restore normal TLS control for junctions the ambulance has already
    passed. Call this every simulation step after force_green_along_route().
    """
    if amb_id not in traci.vehicle.getIDList():
        # Ambulance left simulation — release everything
        for tls_id in list(_green_locked.keys()):
            try:
                traci.trafficlight.setProgram(tls_id, "0")
            except Exception:
                pass
        _green_locked.clear()
        _red_locked.clear()
        return

    route       = traci.vehicle.getRoute(amb_id)
    current_idx = traci.vehicle.getRouteIndex(amb_id)
    passed      = route[:max(0, current_idx - 1)]   # edges fully behind ambulance

    for edge in passed:
        tls_id = _get_tls_for_edge(edge)
        if tls_id and tls_id in _green_locked:
            try:
                traci.trafficlight.setProgram(tls_id, "0")
                _green_locked.pop(tls_id, None)
            except Exception:
                pass


# ------------------------------------------------------------------
# Wrong-side-of-road rerouting
# ------------------------------------------------------------------

def _get_opposite_edge(edge_id: str) -> str | None:
    """
    In OSM-derived SUMO networks, the reverse direction of edge 'foo#N'
    is typically '-foo#N'. Returns None if the reverse doesn't exist.
    """
    if edge_id.startswith("-"):
        candidate = edge_id[1:]          # -foo → foo
    else:
        candidate = "-" + edge_id        # foo → -foo

    try:
        all_edges = traci.edge.getIDList()
        return candidate if candidate in all_edges else None
    except Exception:
        return None


def _ambulance_is_blocked(amb_id: str, speed_threshold: float = 0.5,
                           wait_threshold: float = 5.0) -> bool:
    """
    Returns True if the ambulance has been nearly stationary for at least
    wait_threshold seconds — i.e. it is blocked by traffic or a red light.
    """
    try:
        speed    = traci.vehicle.getSpeed(amb_id)
        waiting  = traci.vehicle.getWaitingTime(amb_id)
        return speed < speed_threshold and waiting >= wait_threshold
    except Exception:
        return False


def maybe_use_wrong_side(amb_id: str = "ambulance_1",
                          wait_threshold: float = 8.0,
                          speed_threshold: float = 0.5,
                          cooldown: float = 5.0) -> bool:
    """
    If the ambulance has been blocked for `wait_threshold` seconds,
    reroute it onto the opposite (oncoming) lane of the current edge.

    HOW IT WORKS
    ------------
    1. Detect that the ambulance is stationary (waiting > threshold)
    2. Find the reverse edge of its *next* route edge (the oncoming lane)
    3. Build a new route: [reverse_edge] + rest_of_original_route
    4. Set the vehicle's route and allow it to change lanes freely
    5. Give it max-speed and ignore right-of-way

    SUMO MECHANICS
    --------------
    - traci.vehicle.setRoute() replaces the entire planned route
    - traci.vehicle.changeLane() moves it to a specific lane index
    - traci.vehicle.setLaneChangeMode(0b000000000000) = no restrictions
    - traci.vehicle.setSpeedMode(0) = ignore all traffic rules
    - The vehicle still physically occupies a lane — SUMO will handle
      collision detection (we disable it for the ambulance below)

    Returns True if a wrong-side reroute was triggered.
    """
    global _wrong_side_active, _last_wrong_side_time

    if amb_id not in traci.vehicle.getIDList():
        _wrong_side_active = False
        return False

    # Check cooldown
    current_time = traci.simulation.getTime()
    if current_time - _last_wrong_side_time < cooldown:
        return False  # Cooldown still active

    if not _ambulance_is_blocked(amb_id, speed_threshold, wait_threshold):
        return False

    route       = traci.vehicle.getRoute(amb_id)
    current_idx = traci.vehicle.getRouteIndex(amb_id)

    # We want to go wrong-way on the NEXT edge (the one we're blocked trying to enter)
    if current_idx + 1 >= len(route):
        return False   # no next edge

    next_edge    = route[current_idx + 1]
    reverse_edge = _get_opposite_edge(next_edge)

    if reverse_edge is None:
        print(f"[PRIORITY] No reverse edge for {next_edge} — cannot use wrong side")
        return False

    # Check if the opposite lane is actually less congested
    try:
        next_count = traci.edge.getLastStepVehicleNumber(next_edge)
        rev_count = traci.edge.getLastStepVehicleNumber(reverse_edge)
        if rev_count >= next_count and next_count > 0:
            return False
    except Exception:
        pass

    # Build new route: current_edge → reverse_edge → remainder
    remainder = list(route[current_idx + 2:])    # edges after the blocked one
    new_route  = [route[current_idx], reverse_edge] + remainder

    try:
        traci.vehicle.setRoute(amb_id, new_route)

        # Allow unrestricted lane changes and ignore traffic rules
        traci.vehicle.setLaneChangeMode(amb_id, 0b000000000000)
        traci.vehicle.setSpeedMode(amb_id, 0)          # ignore all: speed, TLS, right-of-way
        traci.vehicle.setMaxSpeed(amb_id, 30.0)        # ~108 km/h
        traci.vehicle.setColor(amb_id, (255, 165, 0, 255))  # orange = wrong-side mode

        _wrong_side_active = True
        _last_wrong_side_time = current_time
        print(f"[PRIORITY] 🚨 WRONG-SIDE ACTIVATED: rerouting via {reverse_edge} "
              f"(was blocked on {next_edge} for {traci.vehicle.getWaitingTime(amb_id):.0f}s)")
        return True

    except Exception as e:
        print(f"[PRIORITY] Wrong-side reroute failed: {e}")
        return False


def restore_normal_driving(amb_id: str = "ambulance_1"):
    """
    After the ambulance clears a wrong-side stretch, restore normal
    driving — red light compliance (but not speed limits).
    Call each step when _wrong_side_active is True and ambulance is moving.
    """
    global _wrong_side_active
    if not _wrong_side_active:
        return
    if amb_id not in traci.vehicle.getIDList():
        _wrong_side_active = False
        return

    speed = traci.vehicle.getSpeed(amb_id)
    if speed > 2.0:   # moving again — restore partial compliance
        try:
            # SpeedMode 7 = ignore right-of-way at junctions, obey TLS partially
            traci.vehicle.setSpeedMode(amb_id, 7)
            traci.vehicle.setColor(amb_id, (255, 0, 0, 255))   # back to red
            _wrong_side_active = False
            print(f"[PRIORITY] ✅ Wrong-side cleared — ambulance moving normally again")
        except Exception:
            pass
