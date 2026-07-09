"""
Run the MAESTRO-FL dashboard server.

Usage:
    python -m backend.server
    python -m backend.server --port 8000
"""

import argparse
import os
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start MAESTRO-FL Dashboard Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--no-pygame", action="store_true", help="Run SUMO/dashboard without the Pygame visual renderer")
    args = parser.parse_args()

    if args.no_pygame:
        os.environ["MAESTRO_PYGAME"] = "0"

    uvicorn.run("backend.server:app", host=args.host, port=args.port, reload=False)
