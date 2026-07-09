#!/usr/bin/env python3
"""
run_pygame_demo.py — Launch MAESTRO-FL with the Pygame visual renderer.

SUMO runs headless (-nogui). Pygame reads state each step and renders it.
The existing FastAPI/WebSocket dashboard continues running in parallel.

Usage:
    python run_pygame_demo.py                       # default: fixed_timer
    python run_pygame_demo.py --condition maestro_fl
    python run_pygame_demo.py --compare             # run fixed_timer then maestro_fl, show results
"""

import argparse
import asyncio
import os
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "rl_agent"))

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MAESTRO-FL Pygame Demo")
    parser.add_argument(
        "--condition",
        choices=["fixed_timer", "ppo_only", "maestro_fl"],
        default="fixed_timer",
        help="Traffic control condition to simulate (default: fixed_timer)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run fixed_timer then maestro_fl sequentially, show comparison results",
    )
    parser.add_argument("--width", type=int, default=1400, help="Canvas width")
    parser.add_argument("--height", type=int, default=900, help="Canvas height")
    parser.add_argument(
        "--inject-step",
        type=int,
        default=60,
        help="Auto-inject ambulance at this simulation step (0 = manual only)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Stop simulation after N steps (0 = unlimited)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the parallel FastAPI dashboard (default: 8000)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dashboard server (background)
# ---------------------------------------------------------------------------

def start_dashboard_server(port: int):
    """Start the FastAPI/WebSocket dashboard in a background thread."""
    try:
        import uvicorn
        from backend.server import app
        config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        print(f"[Demo] Dashboard server started on http://localhost:{port}")
        return server
    except Exception as e:
        print(f"[Demo] Could not start dashboard server: {e}")
        return None


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

def run_single(condition: str, inject_step: int, max_steps: int):
    """
    Run one simulation with Pygame rendering.

    Returns a dict of final metrics when the simulation ends.
    """
    from backend.sumo_runner import SumoRunner

    loop = asyncio.new_event_loop()
    queue = asyncio.Queue()

    runner = SumoRunner(queue, loop, enable_pygame=True)
    runner.start_simulation(condition)

    # Wait for the simulation thread to spin up and create the renderer
    deadline = time.time() + 10
    while runner.renderer is None and runner.running and time.time() < deadline:
        time.sleep(0.1)

    if runner.renderer is None:
        print("[Demo] Renderer did not initialize.  Running dashboard-only.")

    # Auto-inject ambulance at the requested step
    injected = inject_step <= 0  # already "done" if manual

    last_state = {}
    try:
        while runner.running:
            # Peek at state via the asyncio queue (non-blocking)
            try:
                while not queue.empty():
                    last_state = queue.get_nowait()
            except Exception:
                pass

            step = last_state.get("step", 0)

            # Auto-inject
            if not injected and step >= inject_step:
                runner.request_inject()
                injected = True
                print(f"[Demo] Auto-injected ambulance at step {step}")

            # Max-step guard
            if max_steps > 0 and step >= max_steps:
                print(f"[Demo] Reached max steps ({max_steps}). Stopping.")
                break

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[Demo] Interrupted.")
    finally:
        runner.stop_simulation()

    # Extract final metrics for comparison mode
    metrics = last_state.get("metrics", {})
    ambulance = last_state.get("ambulance", {})
    return {
        "ambulance_travel_time": ambulance.get("travel_time", 0),
        "ambulance_waiting_time": ambulance.get("waiting_time", 0),
        "avg_queue_length": metrics.get("avg_queue_length", 0),
        "avg_waiting_time": metrics.get("avg_waiting_time", 0),
        "throughput": metrics.get("throughput", 0),
    }


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def run_comparison(inject_step: int, max_steps: int):
    """Run fixed_timer then maestro_fl, display results side-by-side."""
    print("\n" + "=" * 60)
    print("  COMPARISON MODE: Fixed Timer vs MAESTRO-FL")
    print("=" * 60 + "\n")

    print("[Compare] Phase 1/2: Running FIXED TIMER...")
    fixed_results = run_single("fixed_timer", inject_step, max_steps)
    print(f"[Compare] Fixed Timer results: {fixed_results}\n")

    time.sleep(1)

    print("[Compare] Phase 2/2: Running MAESTRO-FL...")
    maestro_results = run_single("maestro_fl", inject_step, max_steps)
    print(f"[Compare] MAESTRO-FL results: {maestro_results}\n")

    # Show comparison screen
    try:
        from visualization.pygame_renderer import MAESTRORenderer
        import pygame

        net_file = os.path.join(REPO_ROOT, "sumo_env", "network", "osm_fixed.net.xml")
        renderer = MAESTRORenderer(
            net_file=net_file,
            start_thread=False,
        )
        renderer.show_comparison_screen(fixed_results, maestro_results)
        renderer.stop()
    except Exception as e:
        print(f"[Compare] Could not show comparison screen: {e}")
        print("\n--- RESULTS ---")
        for key in fixed_results:
            print(f"  {key}: Fixed={fixed_results[key]}  MAESTRO={maestro_results[key]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Start the dashboard server in the background
    start_dashboard_server(args.port)

    if args.compare:
        run_comparison(args.inject_step, args.max_steps)
    else:
        results = run_single(args.condition, args.inject_step, args.max_steps)
        print(f"\n[Demo] Final metrics: {results}")


if __name__ == "__main__":
    main()
