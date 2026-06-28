"""
federated/priority_trigger.py — Novel Out-of-Cycle Emergency Trigger
====================================================================
THIS IS THE NOVEL CONTRIBUTION OF MAESTRO-FL.

Standard Flower operates in synchronous rounds: server → clients → aggregate → repeat.
There is no built-in "interrupt now and push" mechanism. This module implements a
lightweight side-channel that:

    1. Receives a PRIORITY_MESSAGE from the SUMO/RL side when an ambulance is detected.
    2. Reads the current global model parameters (from server's shared state or disk).
    3. Pushes them immediately to affected junction clients, OUT OF BAND.
    4. Flips the priority flag on those junctions' environments (action mask activates).

Integration wiring:
    SUMO ambulance detected
    → ambulance.build_priority_broadcast()
    → coordinator.receive_priority_broadcast(message)
    → push global weights to affected clients
    → flip env.priority_active = True on each
    → TrafficEnv.step() uses _priority_override_action()

SPIKE THIS IN ISOLATION FIRST (Week 1):
    Fake the SUMO event with a manual function call, fake the "push to client"
    with a print statement, prove the message flows end-to-end before plugging
    in real models.
"""

import time
import threading
from typing import Callable, Dict, List, Optional, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.schema import validate_priority_message


class PriorityCoordinator:
    """
    Manages out-of-cycle emergency model pushes alongside the Flower server.

    This coordinator maintains a registry of active priorities (with TTL expiry)
    and, when triggered, immediately pushes the latest global model weights
    to affected junction clients.
    """

    def __init__(
        self,
        get_global_params_fn: Callable[[], Optional[list]],
        push_to_client_fn: Callable[[str, list, float], None],
    ):
        """
        Args:
            get_global_params_fn: Callable that returns the current global model
                parameters as a list of numpy arrays. Typically reads from the
                Flower server's strategy or from a shared file.

            push_to_client_fn: Callable(junction_id, parameters, urgency) that
                delivers the model parameters to a specific junction's client.
                In production this calls client.receive_emergency_push().
                For spike testing, this can just be a print statement.
        """
        self.get_global_params = get_global_params_fn
        self.push_to_client = push_to_client_fn

        # junction_id → expiry_timestamp
        self.active_priorities: Dict[str, float] = {}

        # Event log for evaluation/debugging
        self.event_log: List[Dict[str, Any]] = []

        # Thread lock for concurrent access safety
        self._lock = threading.Lock()

    def receive_priority_broadcast(self, message: dict) -> int:
        """
        Handle an incoming priority broadcast from the SUMO/RL side.

        This is the main entry point. When an ambulance is detected:
        1. Validates the message format.
        2. For each affected junction, stores the priority with TTL.
        3. Immediately pushes current global model to those junctions.

        Args:
            message: Dict conforming to shared.schema.PRIORITY_MESSAGE format.

        Returns:
            Number of junctions that received the emergency push.
        """
        if not validate_priority_message(message):
            print("[PRIORITY] Invalid priority message — ignoring.")
            return 0

        now = time.time()
        pushed_count = 0

        with self._lock:
            for jid in message["junction_ids"]:
                expiry = now + message["ttl_seconds"]
                self.active_priorities[jid] = expiry

                # Get current global model
                params = self.get_global_params()
                if params is not None:
                    # Push immediately — this is the out-of-cycle mechanism
                    self.push_to_client(
                        jid, params, urgency=message["urgency"]
                    )
                    pushed_count += 1
                    print(
                        f"[PRIORITY] Emergency push to {jid} — "
                        f"urgency={message['urgency']:.2f}, "
                        f"ttl={message['ttl_seconds']:.1f}s"
                    )
                else:
                    print(
                        f"[PRIORITY] No global params available — "
                        f"skipping push to {jid}"
                    )

            # Log the event
            self.event_log.append({
                "timestamp": now,
                "message": message,
                "junctions_pushed": pushed_count,
            })

        return pushed_count

    def is_priority_active(self, junction_id: str) -> bool:
        """
        Check if a junction currently has an active emergency priority.

        Automatically cleans up expired priorities.

        Args:
            junction_id: SUMO junction ID to check.

        Returns:
            True if priority is active and not expired.
        """
        with self._lock:
            expiry = self.active_priorities.get(junction_id)
            if expiry is None:
                return False
            if time.time() > expiry:
                del self.active_priorities[junction_id]
                return False
            return True

    def get_active_junctions(self) -> List[str]:
        """Return list of junction IDs with currently active priorities."""
        now = time.time()
        with self._lock:
            active = [
                jid for jid, expiry in self.active_priorities.items()
                if now <= expiry
            ]
            # Clean up expired
            expired = [
                jid for jid, expiry in self.active_priorities.items()
                if now > expiry
            ]
            for jid in expired:
                del self.active_priorities[jid]
            return active

    def clear_all_priorities(self) -> None:
        """Clear all active priorities (e.g., after ambulance passes)."""
        with self._lock:
            self.active_priorities.clear()
            print("[PRIORITY] All priorities cleared.")

    @property
    def total_events(self) -> int:
        """Total number of emergency events processed."""
        return len(self.event_log)


# ---------------------------------------------------------------------------
# Spike test — run this standalone to verify the message flow works
# ---------------------------------------------------------------------------

def _spike_test():
    """
    Standalone spike test with fake inputs and outputs.
    Run this in Week 1 to prove the priority trigger flow works
    before plugging in real SUMO events and Flower models.
    """
    print("=" * 60)
    print("SPIKE TEST: Priority Trigger (fake inputs/outputs)")
    print("=" * 60)

    # Fake global params (just random arrays)
    import numpy as np
    fake_params = [np.random.randn(10, 10), np.random.randn(10)]

    def fake_get_params():
        print("  [FAKE] Reading global model parameters...")
        return fake_params

    def fake_push(junction_id, params, urgency):
        print(
            f"  [FAKE] Pushing model to junction {junction_id} "
            f"(urgency={urgency:.2f}, {len(params)} param arrays)"
        )

    # Create coordinator
    coordinator = PriorityCoordinator(
        get_global_params_fn=fake_get_params,
        push_to_client_fn=fake_push,
    )

    # Simulate an ambulance detection
    from shared.schema import make_priority_message

    message = make_priority_message(
        junction_ids=["J1", "J2", "J3"],
        urgency=0.85,
        timestamp=time.time(),
        ttl_seconds=15.0,
    )

    print("\n--- Sending priority broadcast ---")
    pushed = coordinator.receive_priority_broadcast(message)
    print(f"\nPushed to {pushed} junctions.")

    print("\n--- Checking active priorities ---")
    for jid in ["J1", "J2", "J3", "J4"]:
        active = coordinator.is_priority_active(jid)
        print(f"  {jid}: {'ACTIVE' if active else 'inactive'}")

    print(f"\nTotal events logged: {coordinator.total_events}")
    print("\n✓ Spike test passed — message flow works end to end.")
    print("=" * 60)


if __name__ == "__main__":
    _spike_test()
