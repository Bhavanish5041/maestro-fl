"""
shared/schema.py — MAESTRO-FL Data Schema
==========================================
Locked on day one. All modules produce/consume data using these formats.
Person A produces traffic logs; Persons B and C consume them.
Person D uses PRIORITY_MESSAGE for the novel out-of-cycle FL trigger.
"""

from typing import List

# ---------------------------------------------------------------------------
# Traffic log row format (Person A produces → B/C consume)
# ---------------------------------------------------------------------------
TRAFFIC_LOG_COLUMNS = [
    "timestamp",       # simulation step (float, seconds)
    "junction_id",     # SUMO junction ID string
    "queue_length",    # total halting vehicles across controlled lanes
    "waiting_time",    # cumulative waiting time (seconds) across lanes
    "current_phase",   # integer phase index at this junction
    "phase_duration",  # seconds spent in current phase so far
    "vehicle_count",   # total vehicles on controlled lanes
]

# ---------------------------------------------------------------------------
# Priority broadcast message format (the novel FL trigger)
# Sent when an ambulance is detected approaching a corridor of junctions.
# ---------------------------------------------------------------------------
PRIORITY_MESSAGE_KEYS = [
    "junction_ids",    # list[str]  — downstream junctions on ambulance path
    "urgency",         # float      — inverse of estimated time-to-arrival
    "timestamp",       # float      — simulation time when event was created
    "ttl_seconds",     # float      — how long this priority flag stays active
]


def make_priority_message(
    junction_ids: List[str],
    urgency: float,
    timestamp: float,
    ttl_seconds: float = 30.0,
) -> dict:
    """Create a validated priority broadcast message."""
    if not junction_ids:
        raise ValueError("junction_ids must be a non-empty list")
    if urgency < 0:
        raise ValueError("urgency must be non-negative")
    return {
        "junction_ids": list(junction_ids),
        "urgency": float(urgency),
        "timestamp": float(timestamp),
        "ttl_seconds": float(ttl_seconds),
    }


def validate_priority_message(msg: dict) -> bool:
    """Check that a dict conforms to the PRIORITY_MESSAGE schema."""
    for key in PRIORITY_MESSAGE_KEYS:
        if key not in msg:
            return False
    if not isinstance(msg["junction_ids"], list) or len(msg["junction_ids"]) == 0:
        return False
    if not isinstance(msg["urgency"], (int, float)):
        return False
    return True


def validate_log_row(row: dict) -> bool:
    """Check that a traffic log row has all required columns."""
    return all(col in row for col in TRAFFIC_LOG_COLUMNS)
