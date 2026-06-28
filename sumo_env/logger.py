"""
sumo_env/logger.py — Traffic Data Logger
=========================================
Writes per-step queue/wait/phase data to CSV using the shared schema.
Person A produces these logs; Persons B (PPO) and C (LSTM) consume them.
"""

import os
import csv
import sys
from typing import Optional

try:
    import traci
except ImportError:
    print("WARNING: traci not found. Install SUMO and ensure traci is on PYTHONPATH.")
    traci = None

sys.path.insert(0, "..")
from shared.schema import TRAFFIC_LOG_COLUMNS


class TrafficLogger:
    """
    Logs traffic state at each simulation step to a CSV file.

    Usage:
        logger = TrafficLogger("logs/traffic_log.csv", junction_ids=["J1", "J2"])
        while simulation_running:
            logger.log_step(sim_time)
        logger.close()
    """

    def __init__(
        self,
        output_path: str,
        junction_ids: Optional[list] = None,
    ):
        """
        Args:
            output_path: Path to the output CSV file.
            junction_ids: List of junction IDs to log. If None, logs all
                          traffic-light-controlled junctions.
        """
        self.output_path = output_path
        self.junction_ids = junction_ids
        self._rows_written = 0

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        self._file = open(output_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=TRAFFIC_LOG_COLUMNS)
        self._writer.writeheader()

    def _get_junction_ids(self) -> list:
        """Return junction IDs to log (user-specified or all TLS junctions)."""
        if self.junction_ids is not None:
            return self.junction_ids
        return traci.trafficlight.getIDList()

    def log_step(self, sim_time: float) -> int:
        """
        Log the current traffic state for all tracked junctions.

        Args:
            sim_time: Current simulation timestamp.

        Returns:
            Number of rows written in this step.
        """
        rows_this_step = 0
        for junc_id in self._get_junction_ids():
            try:
                lanes = traci.trafficlight.getControlledLanes(junc_id)
                # De-duplicate lanes (a lane can appear multiple times for
                # different signal groups)
                unique_lanes = list(dict.fromkeys(lanes))

                queue_length = sum(
                    traci.lane.getLastStepHaltingNumber(l) for l in unique_lanes
                )
                waiting_time = sum(
                    traci.lane.getWaitingTime(l) for l in unique_lanes
                )
                vehicle_count = sum(
                    traci.lane.getLastStepVehicleNumber(l) for l in unique_lanes
                )
                current_phase = traci.trafficlight.getPhase(junc_id)
                phase_duration = traci.trafficlight.getSpentDuration(junc_id)

                row = {
                    "timestamp": sim_time,
                    "junction_id": junc_id,
                    "queue_length": queue_length,
                    "waiting_time": round(waiting_time, 2),
                    "current_phase": current_phase,
                    "phase_duration": round(phase_duration, 2),
                    "vehicle_count": vehicle_count,
                }
                self._writer.writerow(row)
                rows_this_step += 1
                self._rows_written += 1

            except traci.TraCIException as e:
                print(f"[LOGGER] Warning: could not log junction {junc_id}: {e}")

        return rows_this_step

    @property
    def total_rows(self) -> int:
        """Total rows written so far."""
        return self._rows_written

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._file.close()
        print(
            f"[LOGGER] Closed {self.output_path} — {self._rows_written} rows written."
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
