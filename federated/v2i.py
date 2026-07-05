"""
federated/v2i.py
================
Minimal V2I communication layer for emergency priority broadcasts.

This module models vehicle-to-infrastructure messaging without adding a
network broker dependency. It validates priority messages, stores them in a
bounded bus, and lets infrastructure clients consume messages addressed to
their junction.
"""

from collections import deque
from typing import Deque, Dict, List

from shared.schema import validate_priority_message


class V2IMessageBus:
    """In-memory V2I bus for simulation, tests, and demos."""

    def __init__(self, max_messages: int = 256):
        self.messages: Deque[dict] = deque(maxlen=max_messages)

    def publish_priority(self, message: dict) -> None:
        """Publish a validated priority message."""
        if not validate_priority_message(message):
            raise ValueError("Invalid priority message")
        self.messages.append(dict(message))

    def consume_for_junction(self, junction_id: str, sim_time: float) -> List[dict]:
        """Return non-expired messages addressed to a junction."""
        matches = []
        retained = deque(maxlen=self.messages.maxlen)
        for message in self.messages:
            expires_at = message["timestamp"] + message["ttl_seconds"]
            if expires_at < sim_time:
                continue
            if junction_id in message["junction_ids"]:
                matches.append(message)
            retained.append(message)
        self.messages = retained
        return matches


def apply_v2i_priority(env, bus: V2IMessageBus, sim_time: float) -> bool:
    """
    Apply the most urgent current V2I priority message to a TrafficEnv.

    Returns True when a matching message was applied.
    """
    messages = bus.consume_for_junction(env.junction_id, sim_time)
    if not messages:
        return False

    message = max(messages, key=lambda item: item["urgency"])
    env.set_priority(
        urgency=message["urgency"],
        ttl=message["ttl_seconds"],
    )
    return True


def priority_message_size_bytes(message: Dict) -> int:
    """Estimate compact V2I payload size for communication-cost reporting."""
    junction_bytes = sum(len(str(junction_id)) for junction_id in message["junction_ids"])
    eta_bytes = 8 * len(message.get("eta_seconds", {}))
    fixed_fields = 24
    return fixed_fields + junction_bytes + eta_bytes
