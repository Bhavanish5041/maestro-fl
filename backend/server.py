import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

from backend.sumo_runner import SumoRunner

# Determine paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

# State Queue
state_queue = asyncio.Queue()

# Active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()
runner = None

async def broadcast_loop():
    while True:
        state = await state_queue.get()
        await manager.broadcast(state)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner
    runner = SumoRunner(state_queue, asyncio.get_running_loop())
    asyncio.create_task(broadcast_loop())
    yield

app = FastAPI(title="MAESTRO-FL Dashboard", lifespan=lifespan)

@app.post("/start")
async def start_sim(condition: str = "fixed_timer"):
    if runner:
        runner.start_simulation(condition)
    return {"status": "started", "condition": condition}

@app.post("/inject")
async def inject_amb():
    if runner:
        runner.request_inject()
    return {"status": "inject_requested"}

@app.post("/pause")
async def pause_sim():
    if runner:
        runner.pause_simulation()
    return {"status": "paused"}

@app.post("/resume")
async def resume_sim():
    if runner:
        runner.resume_simulation()
    return {"status": "resumed"}

@app.post("/reset")
async def reset_sim():
    if runner:
        runner.stop_simulation()
    return {"status": "reset"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming commands over WS if any
            # For now, commands are primarily handled via POST routes
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Serve the dashboard directory (index.html, style.css, app.js, etc.)
# Must be mounted AFTER route definitions so API routes take priority
app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="static")
