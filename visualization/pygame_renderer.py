"""
Pygame visual renderer for the MAESTRO-FL SUMO simulation.

SUMO remains the simulator. This module consumes backend state payloads and
renders them; it never opens a TraCI connection and never changes simulation
state directly.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pygame
    import pygame.gfxdraw
except Exception:  # pragma: no cover - availability depends on demo machine
    pygame = None

try:
    import sumolib
except Exception:  # pragma: no cover - SUMO env may be absent in CI
    sumolib = None


Point = Tuple[float, float]
Color = Tuple[int, int, int]


@dataclass
class VehicleState:
    prev_pos: Optional[Point] = None
    curr_pos: Optional[Point] = None
    prev_angle: float = 0.0
    curr_angle: float = 0.0
    last_update_time: float = 0.0

    def update(self, x: float, y: float, angle: float, step_time: float) -> None:
        self.prev_pos = self.curr_pos or (x, y)
        self.curr_pos = (x, y)
        self.prev_angle = self.curr_angle
        self.curr_angle = angle
        self.last_update_time = step_time

    def interpolated_pos(self, current_time: float, step_duration: float = 1.0) -> Optional[Point]:
        if self.curr_pos is None:
            return None
        if self.prev_pos is None:
            return self.curr_pos
        t = min(max((current_time - self.last_update_time) / step_duration, 0.0), 1.0)
        x = self.prev_pos[0] + t * (self.curr_pos[0] - self.prev_pos[0])
        y = self.prev_pos[1] + t * (self.curr_pos[1] - self.prev_pos[1])
        return x, y

    def interpolated_angle(self, current_time: float, step_duration: float = 1.0) -> float:
        t = min(max((current_time - self.last_update_time) / step_duration, 0.0), 1.0)
        delta = (self.curr_angle - self.prev_angle + 180.0) % 360.0 - 180.0
        return self.prev_angle + t * delta


@dataclass
class RingEffect:
    junction_id: str
    position: Point
    start_time: float
    duration: float = 1.0


class MAESTRORenderer:
    CANVAS_W = 1400
    CANVAS_H = 900
    PANEL_W = 320
    FPS = 30

    BG_COLOR = (82, 139, 74)
    GRASS_DARK = (67, 122, 63)
    GRASS_LIGHT = (96, 157, 85)
    ROAD_COLOR = (78, 78, 78)
    ROAD_HIGHLIGHT = (94, 94, 94)
    ROAD_EDGE = (54, 54, 54)
    LANE_MARKING = (238, 238, 224)
    CENTER_MARKING = (236, 198, 70)
    ROUTE_COLOR = (64, 158, 160)
    VEHICLE_COLOR = (32, 106, 207)
    AMBULANCE_COLOR = (255, 60, 60)
    TL_GREEN = (50, 220, 100)
    TL_RED = (220, 60, 60)
    TL_YELLOW = (220, 180, 60)
    PRIORITY_GLOW = (255, 100, 50)
    QUEUE_COLOR = (220, 80, 80)
    TEXT_COLOR = (200, 205, 215)
    MUTED_TEXT = (120, 128, 140)
    PANEL_BG = (22, 25, 32)
    ACCENT = (255, 140, 0)
    SIGNAL_BOX = (16, 18, 18)
    SIGNAL_INACTIVE = (50, 50, 50)

    def __init__(
        self,
        net_file: str,
        route_edges: Optional[Iterable[str]] = None,
        command_callback: Optional[Callable[[str], None]] = None,
        canvas_width: int = CANVAS_W,
        canvas_height: int = CANVAS_H,
        start_thread: bool = True,
    ) -> None:
        if pygame is None:
            raise RuntimeError("pygame is not installed")
        if sumolib is None:
            raise RuntimeError("sumolib is not available; check SUMO_HOME/tools on PYTHONPATH")

        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.command_callback = command_callback
        self._lock = threading.RLock()
        self._state: Dict = {}
        self._running = True
        self._paused = False
        self._manual_camera = False
        self._dragging = False
        self._last_mouse = (0, 0)
        self._last_step_wall_time = time.time()

        self.vehicle_states: Dict[str, VehicleState] = {}
        self.ring_effects: List[RingEffect] = []
        self.tls_positions: Dict[str, Point] = {}
        self.route_edges = set(route_edges or [])
        self.latest_results: Dict[str, Dict] = {}
        self._scaled_surface_cache = {}
        self._vehicle_sprite_cache = {}
        self.asset_dir = Path(__file__).resolve().parent.parent / "images"
        self.background_image = None
        self.vehicle_image_assets = {"car": [], "truck": [], "ambulance": []}
        self.signal_image_assets = {}

        self.rendered_edges, self.junction_positions, self.lane_shapes, self.to_canvas = (
            self._build_render_geometry(net_file)
        )
        self.camera_center = np.array([self.canvas_width / 2, self.canvas_height / 2], dtype=float)
        self.camera_pan = np.array([0.0, 0.0], dtype=float)
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.target_center = self.camera_center.copy()

        self.font = None
        self.font_small = None
        self.font_large = None
        self.font_metric = None
        self.static_surface = None

        self._thread: Optional[threading.Thread] = None
        if start_thread:
            self._thread = threading.Thread(target=self.run, daemon=True)
            self._thread.start()

    @staticmethod
    def available() -> bool:
        return pygame is not None and sumolib is not None

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.5)
        try:
            pygame.quit()
        except Exception:
            pass

    def update(self, state_dict: Dict) -> None:
        now = time.time()
        with self._lock:
            if state_dict.get("event") == "ambulance_detected":
                self.route_edges = set(state_dict.get("route_edges") or [])
                if self.static_surface is not None:
                    self.static_surface = self._build_static_surface()
                self._scaled_surface_cache = {}
                return

            if state_dict.get("event") == "priority_sync":
                junction_id = state_dict.get("junction_id")
                pos = self.tls_positions.get(junction_id) or self.junction_positions.get(junction_id)
                if not pos and state_dict.get("pos"):
                    raw_pos = state_dict.get("pos")
                    pos = self.to_canvas(raw_pos[0], raw_pos[1])
                if pos:
                    self.ring_effects.append(RingEffect(junction_id=junction_id, position=pos, start_time=now))
                return

            self._state = state_dict
            self._last_step_wall_time = now
            for tls in state_dict.get("render", {}).get("traffic_lights", []):
                if tls.get("pos"):
                    raw_pos = tls["pos"]
                    self.tls_positions[tls.get("id")] = self.to_canvas(raw_pos[0], raw_pos[1])
            for vehicle in state_dict.get("render", {}).get("vehicles", []):
                pos = vehicle.get("pos")
                if not pos:
                    continue
                veh_id = vehicle.get("id")
                state = self.vehicle_states.setdefault(veh_id, VehicleState())
                canvas_pos = self.to_canvas(pos[0], pos[1])
                state.update(canvas_pos[0], canvas_pos[1], float(vehicle.get("angle", 0.0)), now)

            active_ids = {v.get("id") for v in state_dict.get("render", {}).get("vehicles", [])}
            for veh_id in list(self.vehicle_states):
                if veh_id not in active_ids:
                    self.vehicle_states.pop(veh_id, None)

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("MAESTRO-FL Visual Renderer")
        self.screen = pygame.display.set_mode((self.canvas_width, self.canvas_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)
        self.font_small = pygame.font.SysFont("arial", 14)
        self.font_large = pygame.font.SysFont("arial", 30, bold=True)
        self.font_metric = pygame.font.SysFont("arial", 38, bold=True)
        self._load_image_assets()
        self.static_surface = self._build_static_surface()

        while self._running:
            self._handle_events()
            with self._lock:
                state = dict(self._state)
                self._draw_frame(state)
            pygame.display.flip()
            self.clock.tick(self.FPS)

    def show_comparison_screen(self, fixed_results: Dict, maestro_results: Dict) -> None:
        """Blocking final screen for scripted demos that run two conditions."""
        if not pygame.get_init():
            return

        def metric(result: Dict, key: str, default: float = 0.0) -> float:
            return float(result.get(key, default) or default)

        fixed_time = metric(fixed_results, "ambulance_travel_time")
        maestro_time = metric(maestro_results, "ambulance_travel_time")
        faster = 0.0 if fixed_time <= 0 else max(0.0, (fixed_time - maestro_time) / fixed_time * 100.0)
        wait_zero = metric(maestro_results, "ambulance_waiting_time") <= 0.01

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False

            self.screen.fill(self.BG_COLOR)
            title = f"{faster:.0f}% faster. {'Zero waiting.' if wait_zero else 'Less waiting.'}"
            self._draw_text(title, (self.canvas_width // 2, 80), self.font_metric, self.ACCENT, center=True)
            self._draw_results_panel((110, 170, 500, 520), "Fixed Timer", fixed_results, good=False)
            self._draw_results_panel((790, 170, 500, 520), "MAESTRO-FL", maestro_results, good=True)
            pygame.display.flip()
            self.clock.tick(self.FPS)

    def _build_render_geometry(self, net_file: str, padding: int = 60):
        net = sumolib.net.readNet(net_file)
        edges = []
        all_points = []
        lane_shapes = {}

        for edge in net.getEdges():
            shape = edge.getShape()
            if shape:
                speed = edge.getSpeed()
                edges.append((edge.getID(), shape, speed))
                all_points.extend(shape)
            for lane in edge.getLanes():
                lane_shape = lane.getShape() or shape
                if lane_shape:
                    lane_shapes[lane.getID()] = lane_shape

        if not all_points:
            raise RuntimeError(f"No drawable edge geometry found in {net_file}")

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        scale_x = (self.canvas_width - 2 * padding - self.PANEL_W) / max(max_x - min_x, 1.0)
        scale_y = (self.canvas_height - 2 * padding) / max(max_y - min_y, 1.0)
        scale = min(scale_x, scale_y)

        def to_canvas(x: float, y: float) -> Tuple[int, int]:
            px = int((x - min_x) * scale + padding)
            py = int(self.canvas_height - ((y - min_y) * scale + padding))
            return px, py

        rendered_edges = []
        for edge_id, shape, speed in edges:
            canvas_shape = [to_canvas(x, y) for x, y in shape]
            rendered_edges.append((edge_id, canvas_shape, speed))

        rendered_lanes = {}
        for lane_id, shape in lane_shapes.items():
            rendered_lanes[lane_id] = [to_canvas(x, y) for x, y in shape]

        junction_positions = {}
        for junction in net.getNodes():
            jx, jy = junction.getCoord()
            junction_positions[junction.getID()] = to_canvas(jx, jy)

        return rendered_edges, junction_positions, rendered_lanes, to_canvas

    def _load_image_assets(self):
        if not self.asset_dir.exists():
            return

        image_paths = sorted(
            p for p in self.asset_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        if not image_paths:
            return

        loaded = []
        for path in image_paths:
            try:
                image = pygame.image.load(str(path))
                if image.get_alpha() is not None:
                    image = image.convert_alpha()
                else:
                    image = image.convert()
                loaded.append((path, image))
            except Exception as e:
                print(f"[Renderer] Could not load image asset {path}: {e}")

        if not loaded:
            return

        background_path, background = max(
            loaded,
            key=lambda item: item[1].get_width() * item[1].get_height(),
        )
        if background.get_width() >= 500 and background.get_height() >= 400:
            self.background_image = background

        for path, image in loaded:
            if path == background_path:
                continue

            width, height = image.get_size()
            name = path.name.lower()
            if height >= 75 and width <= 42:
                if "green" in name or not self.signal_image_assets.get("green") and self._dominant_signal_color(image) == "green":
                    self.signal_image_assets["green"] = image
                elif "yellow" in name or not self.signal_image_assets.get("yellow") and self._dominant_signal_color(image) == "yellow":
                    self.signal_image_assets["yellow"] = image
                elif "red" in name or not self.signal_image_assets.get("red"):
                    self.signal_image_assets["red"] = image
                continue

            vehicle = self._normalize_vehicle_asset(image)
            if "ambulance" in name or "emergency" in name or "police" in name:
                self.vehicle_image_assets["ambulance"].append(vehicle)
            elif width >= 60 or "truck" in name or "bus" in name:
                self.vehicle_image_assets["truck"].append(vehicle)
            else:
                self.vehicle_image_assets["car"].append(vehicle)

        if not self.vehicle_image_assets["ambulance"]:
            emergency_candidates = self.vehicle_image_assets["car"] + self.vehicle_image_assets["truck"]
            if emergency_candidates:
                self.vehicle_image_assets["ambulance"].append(emergency_candidates[0])

    def _dominant_signal_color(self, image):
        red = yellow = green = 0
        width, height = image.get_size()
        for x in range(0, width, max(1, width // 8)):
            for y in range(0, height, max(1, height // 12)):
                r, g, b, *rest = image.get_at((x, y))
                alpha = rest[0] if rest else 255
                if alpha < 80:
                    continue
                if r > 160 and g < 120:
                    red += 1
                elif r > 160 and g > 140:
                    yellow += 1
                elif g > 140 and r < 120:
                    green += 1
        if green >= red and green >= yellow:
            return "green"
        if yellow >= red and yellow >= green:
            return "yellow"
        return "red"

    def _normalize_vehicle_asset(self, image):
        if image.get_height() > image.get_width():
            image = pygame.transform.rotate(image, -90)
        return image

    def _build_static_surface(self):
        surface = pygame.Surface((self.canvas_width, self.canvas_height)).convert_alpha()
        if self.background_image is not None:
            bg = self._scale_to_cover(self.background_image, self.canvas_width, self.canvas_height)
            surface.blit(bg, (0, 0))
        else:
            surface.fill((*self.BG_COLOR, 255))

            for x in range(0, self.canvas_width, 36):
                for y in range(0, self.canvas_height, 36):
                    if ((x * 17 + y * 31) % 7) < 3:
                        color = self.GRASS_LIGHT
                    else:
                        color = self.GRASS_DARK
                    pygame.draw.circle(surface, (*color, 32), (x + 11, y + 9), 9)

        if self.background_image is not None:
            return surface

        lane_counts = {}
        for lane_id in self.lane_shapes:
            edge_id = lane_id.rsplit("_", 1)[0]
            lane_counts[edge_id] = lane_counts.get(edge_id, 0) + 1

        for edge_id, points, speed in self.rendered_edges:
            if len(points) < 2:
                continue
            lane_count = max(1, lane_counts.get(edge_id, 1))
            road_width = max(14, lane_count * 9 + 8)
            pygame.draw.lines(surface, self.ROAD_EDGE, False, points, road_width + 4)
            road_color = self.ROAD_HIGHLIGHT if speed * 3.6 > 50 else self.ROAD_COLOR
            pygame.draw.lines(surface, road_color, False, points, road_width)

        for lane_id, points in self.lane_shapes.items():
            if len(points) < 2:
                continue
            pygame.draw.lines(surface, self.ROAD_HIGHLIGHT, False, points, 5)
            self._draw_dashed_polyline(surface, points, self.LANE_MARKING, width=2, dash=12, gap=12)

        for edge_id, points, speed in self.rendered_edges:
            if len(points) < 2:
                continue
            lane_count = max(1, lane_counts.get(edge_id, 1))
            if lane_count > 1 or speed * 3.6 > 50:
                self._draw_dashed_polyline(surface, points, self.CENTER_MARKING, width=2, dash=18, gap=14)

        for edge_id, points, _speed in self.rendered_edges:
            if edge_id in self.route_edges and len(points) >= 2:
                route_overlay = pygame.Surface((self.canvas_width, self.canvas_height), pygame.SRCALPHA)
                pygame.draw.lines(route_overlay, (*self.ROUTE_COLOR, 125), False, points, 7)
                surface.blit(route_overlay, (0, 0))
        return surface

    def _scale_to_cover(self, image, width, height):
        scale = max(width / image.get_width(), height / image.get_height())
        scaled_size = (int(image.get_width() * scale), int(image.get_height() * scale))
        scaled = pygame.transform.smoothscale(image, scaled_size)
        crop = pygame.Surface((width, height)).convert()
        crop.blit(scaled, ((width - scaled_size[0]) // 2, (height - scaled_size[1]) // 2))
        return crop

    def _draw_dashed_polyline(self, surface, points, color, width=2, dash=14, gap=12):
        for start, end in zip(points, points[1:]):
            x1, y1 = start
            x2, y2 = end
            dx = x2 - x1
            dy = y2 - y1
            segment_len = math.hypot(dx, dy)
            if segment_len <= 0:
                continue
            ux = dx / segment_len
            uy = dy / segment_len
            distance = 0.0
            while distance < segment_len:
                dash_end = min(distance + dash, segment_len)
                p1 = (int(x1 + ux * distance), int(y1 + uy * distance))
                p2 = (int(x1 + ux * dash_end), int(y1 + uy * dash_end))
                pygame.draw.line(surface, color, p1, p2, width)
                distance += dash + gap

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._emit_command("quit")
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._emit_command("quit")
                    self._running = False
                elif event.key == pygame.K_SPACE:
                    self._paused = not self._paused
                    self._emit_command("toggle_pause")
                elif event.key == pygame.K_a:
                    self._emit_command("inject_ambulance")
                elif event.key == pygame.K_r:
                    self._reset_camera()
                elif event.key == pygame.K_1:
                    self._emit_command("condition:fixed_timer")
                elif event.key == pygame.K_2:
                    self._emit_command("condition:ppo_only")
                elif event.key == pygame.K_3:
                    self._emit_command("condition:maestro_fl")
            elif event.type == pygame.MOUSEWHEEL:
                mouse = np.array(pygame.mouse.get_pos(), dtype=float)
                world_before = self._screen_to_world(mouse)
                self.zoom = float(np.clip(self.zoom * (1.12 ** event.y), 0.6, 4.0))
                self.target_zoom = self.zoom
                world_after = self._screen_to_world(mouse)
                self.camera_pan += (world_after - world_before) * self.zoom
                self._manual_camera = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._dragging = True
                self._last_mouse = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._dragging = False
            elif event.type == pygame.MOUSEMOTION and self._dragging:
                dx = event.pos[0] - self._last_mouse[0]
                dy = event.pos[1] - self._last_mouse[1]
                self.camera_pan += np.array([dx, dy], dtype=float)
                self._last_mouse = event.pos
                self._manual_camera = True

    def _emit_command(self, command: str) -> None:
        if self.command_callback:
            self.command_callback(command)

    def _reset_camera(self) -> None:
        self._manual_camera = False
        self.camera_pan[:] = 0
        self.target_zoom = 1.0
        self.target_center = np.array([self.canvas_width / 2, self.canvas_height / 2], dtype=float)

    def _draw_frame(self, state: Dict) -> None:
        self._update_camera(state)
        self.screen.fill(self.BG_COLOR)
        self._blit_static_network()
        self._draw_queues(state)
        self._draw_vehicles(state)
        self._draw_traffic_lights(state)
        self._draw_ring_effects()
        self._draw_hud(state)

    def _update_camera(self, state: Dict) -> None:
        if not self._manual_camera and state.get("ambulance", {}).get("active"):
            amb = self.vehicle_states.get("ambulance_1")
            if amb and amb.curr_pos:
                self.target_center = np.array(amb.curr_pos, dtype=float)
                self.target_zoom = 2.5
        elif not self._manual_camera:
            self.target_center = np.array([self.canvas_width / 2, self.canvas_height / 2], dtype=float)
            self.target_zoom = 1.0

        self.camera_center += (self.target_center - self.camera_center) * 0.06
        self.zoom += (self.target_zoom - self.zoom) * 0.06

    def _blit_static_network(self) -> None:
        if abs(self.zoom - 1.0) < 0.01 and np.linalg.norm(self.camera_pan) < 0.5:
            self.screen.blit(self.static_surface, (0, 0))
            return
        scaled_size = (int(self.canvas_width * self.zoom), int(self.canvas_height * self.zoom))
        cache_key = (round(self.zoom, 2), scaled_size)
        scaled = self._scaled_surface_cache.get(cache_key)
        if scaled is None:
            scaled = pygame.transform.smoothscale(self.static_surface, scaled_size)
            self._scaled_surface_cache = {cache_key: scaled}
        top_left = self._world_to_screen(np.array([0.0, 0.0], dtype=float))
        self.screen.blit(scaled, top_left)

    def _draw_queues(self, state: Dict) -> None:
        overlay = pygame.Surface((self.canvas_width, self.canvas_height), pygame.SRCALPHA)
        for lane in state.get("render", {}).get("lanes", []):
            lane_id = lane.get("id")
            queue = int(lane.get("queue", 0) or 0)
            points = self.lane_shapes.get(lane_id)
            if not points or len(points) < 2 or queue <= 0:
                continue
            intensity = min(queue / 5.0, 1.0)
            color = (
                int(self.ROAD_COLOR[0] * (1 - intensity) + self.QUEUE_COLOR[0] * intensity),
                int(self.ROAD_COLOR[1] * (1 - intensity) + self.QUEUE_COLOR[1] * intensity),
                int(self.ROAD_COLOR[2] * (1 - intensity) + self.QUEUE_COLOR[2] * intensity),
                int(65 + 120 * intensity),
            )
            screen_points = [self._world_to_screen_tuple(p) for p in self._last_segment(points, fraction=0.35)]
            if len(screen_points) >= 2:
                pygame.draw.lines(overlay, color, False, screen_points, max(4, int(8 * self.zoom)))
        self.screen.blit(overlay, (0, 0))

    def _draw_vehicles(self, state: Dict) -> None:
        now = time.time()
        priority_active = bool(state.get("priority_active"))
        for vehicle in state.get("render", {}).get("vehicles", []):
            veh_id = vehicle.get("id")
            vstate = self.vehicle_states.get(veh_id)
            if not vstate:
                continue
            pos = vstate.interpolated_pos(now)
            if pos is None:
                continue
            angle = vstate.interpolated_angle(now)
            is_ambulance = veh_id == "ambulance_1"
            screen_pos = self._world_to_screen(np.array(pos, dtype=float))

            if is_ambulance and priority_active:
                pulse = (math.sin(now * 5.0) + 1.0) / 2.0
                radius = int((20 + pulse * 15) * self.zoom)
                halo = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(halo, (255, 255, 255, int(70 + 70 * pulse)), (radius + 2, radius + 2), radius, 2)
                self.screen.blit(halo, (screen_pos[0] - radius - 2, screen_pos[1] - radius - 2))

            sprite_kind = "ambulance" if is_ambulance else self._vehicle_kind_for_id(veh_id)
            sprite = self._vehicle_sprite(sprite_kind, veh_id)
            scale = max(0.55, min(3.0, self.zoom))
            rotated = pygame.transform.rotozoom(sprite, 90.0 - angle, scale)
            rect = rotated.get_rect(center=screen_pos)
            self.screen.blit(rotated, rect)

    def _vehicle_kind_for_id(self, veh_id: str) -> str:
        key = str(veh_id).lower()
        if "truck" in key or "bus" in key or "heavy" in key:
            return "truck"
        return "car"

    def _vehicle_sprite(self, kind: str, veh_id: str = ""):
        loaded_assets = self.vehicle_image_assets.get(kind) or self.vehicle_image_assets.get("car") or []
        if loaded_assets:
            index = abs(hash(str(veh_id))) % len(loaded_assets)
            return loaded_assets[index]

        cached = self._vehicle_sprite_cache.get(kind)
        if cached:
            return cached

        if kind == "ambulance":
            size = (34, 18)
            body = self.AMBULANCE_COLOR
            roof = (245, 245, 245)
        elif kind == "truck":
            size = (32, 16)
            body = (225, 166, 50)
            roof = (178, 118, 34)
        else:
            size = (24, 14)
            body = self.VEHICLE_COLOR
            roof = (22, 60, 120)

        width, height = size
        sprite = pygame.Surface((width + 8, height + 8), pygame.SRCALPHA)
        body_rect = pygame.Rect(4, 4, width, height)
        pygame.draw.rect(sprite, (18, 18, 18, 145), body_rect.move(2, 2), border_radius=4)
        pygame.draw.rect(sprite, body, body_rect, border_radius=4)

        roof_w = int(width * (0.42 if kind != "truck" else 0.55))
        roof_h = max(6, int(height * 0.55))
        roof_rect = pygame.Rect(0, 0, roof_w, roof_h)
        roof_rect.center = body_rect.center
        if kind == "truck":
            roof_rect.centerx -= int(width * 0.12)
        pygame.draw.rect(sprite, roof, roof_rect, border_radius=3)

        wheel_w = max(4, int(width * 0.16))
        wheel_h = max(3, int(height * 0.22))
        wheel_positions = [
            (body_rect.left + 3, body_rect.top - 1),
            (body_rect.right - wheel_w - 3, body_rect.top - 1),
            (body_rect.left + 3, body_rect.bottom - wheel_h + 1),
            (body_rect.right - wheel_w - 3, body_rect.bottom - wheel_h + 1),
        ]
        for wx, wy in wheel_positions:
            pygame.draw.rect(sprite, (12, 12, 12), (wx, wy, wheel_w, wheel_h), border_radius=2)

        if kind == "ambulance":
            cross_x = body_rect.centerx + int(width * 0.18)
            cross_y = body_rect.centery
            pygame.draw.line(sprite, (215, 0, 0), (cross_x - 4, cross_y), (cross_x + 4, cross_y), 2)
            pygame.draw.line(sprite, (215, 0, 0), (cross_x, cross_y - 4), (cross_x, cross_y + 4), 2)
            pygame.draw.circle(sprite, (80, 165, 255), (body_rect.left + 5, body_rect.centery - 3), 2)
            pygame.draw.circle(sprite, (255, 255, 255), (body_rect.left + 5, body_rect.centery + 3), 2)
        else:
            pygame.draw.circle(sprite, (240, 240, 190), (body_rect.right - 3, body_rect.top + 4), 2)
            pygame.draw.circle(sprite, (240, 240, 190), (body_rect.right - 3, body_rect.bottom - 4), 2)

        self._vehicle_sprite_cache[kind] = sprite
        return sprite

    def _draw_traffic_lights(self, state: Dict) -> None:
        now = time.time()
        active_priority = set(state.get("render", {}).get("active_priority_junctions", []))
        for tls in state.get("render", {}).get("traffic_lights", []):
            tls_id = tls.get("id")
            pos = self.tls_positions.get(tls_id) or self.junction_positions.get(tls_id)
            if not pos:
                continue
            state_text = tls.get("state", "")
            if "y" in state_text:
                active_light = "yellow"
            elif "G" in state_text or "g" in state_text:
                active_light = "green"
            else:
                active_light = "red"
            screen_pos = self._world_to_screen(np.array(pos, dtype=float))

            signal_asset = self.signal_image_assets.get(active_light)
            if signal_asset is not None:
                signal_scale = max(0.45, min(1.45, self.zoom * 0.72))
                signal_surface = pygame.transform.rotozoom(signal_asset, 0, signal_scale)
                box_rect = signal_surface.get_rect(center=(screen_pos[0], screen_pos[1] - int(34 * self.zoom)))
                self.screen.blit(signal_surface, box_rect)
            else:
                box_w = max(14, int(18 * self.zoom))
                box_h = max(34, int(46 * self.zoom))
                radius = max(3, int(4 * self.zoom))
                box_rect = pygame.Rect(0, 0, box_w, box_h)
                box_rect.center = (screen_pos[0], screen_pos[1] - int(22 * self.zoom))
                pygame.draw.rect(self.screen, self.SIGNAL_BOX, box_rect, border_radius=max(3, int(4 * self.zoom)))
                pygame.draw.rect(self.screen, (5, 5, 5), box_rect, max(1, int(2 * self.zoom)), border_radius=max(3, int(4 * self.zoom)))

                light_specs = [
                    ("red", self.TL_RED),
                    ("yellow", self.TL_YELLOW),
                    ("green", self.TL_GREEN),
                ]
                spacing = box_h / 4.0
                for index, (name, color) in enumerate(light_specs, start=1):
                    light_center = (box_rect.centerx, int(box_rect.top + spacing * index))
                    draw_color = color if name == active_light else self.SIGNAL_INACTIVE
                    pygame.draw.circle(self.screen, draw_color, light_center, radius)
                    if name == active_light:
                        pygame.draw.circle(self.screen, (245, 245, 230), light_center, radius + 1, 1)

            phase_remaining = tls.get("phase_remaining")
            if phase_remaining is not None:
                label = f"{int(math.ceil(float(phase_remaining)))}s"
            elif tls.get("phase") is not None:
                label = f"P{tls.get('phase')}"
            else:
                label = active_light.upper()[0]
            timer_surface = self.font_small.render(label, True, (245, 245, 230))
            timer_bg = timer_surface.get_rect()
            timer_bg.midleft = (box_rect.right + 5, box_rect.centery)
            padded_bg = timer_bg.inflate(8, 4)
            pygame.draw.rect(self.screen, (20, 20, 20), padded_bg, border_radius=4)
            self.screen.blit(timer_surface, timer_bg)

            if tls.get("priority_override") or tls_id in active_priority:
                pulse = (math.sin(now * 6.0) + 1.0) / 2.0
                ring_radius = max(box_rect.width, int((20 + pulse * 7) * self.zoom))
                pygame.draw.circle(self.screen, self.PRIORITY_GLOW, screen_pos, ring_radius, max(2, int(3 * self.zoom)))

    def _draw_ring_effects(self) -> None:
        now = time.time()
        remaining = []
        for ring in self.ring_effects:
            age = now - ring.start_time
            t = age / ring.duration
            if t >= 1.0:
                continue
            remaining.append(ring)
            radius = int((10 + 50 * t) * self.zoom)
            alpha = int(210 * (1.0 - t))
            screen_pos = self._world_to_screen(np.array(ring.position, dtype=float))
            surface = pygame.Surface((radius * 2 + 8, radius * 2 + 8), pygame.SRCALPHA)
            pygame.draw.circle(surface, (*self.PRIORITY_GLOW, alpha), (radius + 4, radius + 4), radius, max(2, int(4 * self.zoom)))
            self.screen.blit(surface, (screen_pos[0] - radius - 4, screen_pos[1] - radius - 4))
        self.ring_effects = remaining

    def _draw_hud(self, state: Dict) -> None:
        x = self.canvas_width - self.PANEL_W
        panel = pygame.Surface((self.PANEL_W, self.canvas_height), pygame.SRCALPHA)
        panel.fill((*self.PANEL_BG, 215))
        self.screen.blit(panel, (x, 0))

        condition = str(state.get("condition", "fixed_timer")).replace("_", " ").upper()
        metrics = state.get("metrics", {})
        ambulance = state.get("ambulance", {})
        fl = state.get("fl", {})
        y = 30

        self._draw_text("MAESTRO-FL", (x + 24, y), self.font_large, self.ACCENT)
        y += 48
        cond_color = self.ACCENT if condition == "MAESTRO FL" else self.TEXT_COLOR
        self._draw_text(condition, (x + 24, y), self.font_large, cond_color)
        y += 54

        active = bool(ambulance.get("active"))
        self._draw_status_row(x + 24, y, "Ambulance", "ACTIVE" if active else "INACTIVE", self.TL_GREEN if active else self.MUTED_TEXT)
        y += 48

        travel = float(ambulance.get("travel_time", 0) or 0)
        wait = float(ambulance.get("waiting_time", 0) or 0)
        avg_queue = float(metrics.get("avg_queue_length", 0) or 0)
        vehicle_count = int(state.get("render", {}).get("vehicle_count", 0) or 0)

        y = self._draw_metric(x + 24, y, "Travel time", f"{travel:.0f}s", self.TEXT_COLOR)
        y = self._draw_metric(x + 24, y, "Waiting time", f"{wait:.1f}s", self.TL_RED if wait > 0 else self.TL_GREEN)
        y = self._draw_metric(x + 24, y, "Vehicles", str(vehicle_count), self.TEXT_COLOR)

        self._draw_text("Avg queue", (x + 24, y), self.font_small, self.MUTED_TEXT)
        self._draw_text(f"{avg_queue:.2f}", (x + 210, y - 6), self.font, self.TEXT_COLOR)
        y += 24
        pygame.draw.rect(self.screen, (38, 42, 52), (x + 24, y, 250, 10), border_radius=4)
        pygame.draw.rect(self.screen, self.QUEUE_COLOR, (x + 24, y, min(250, int(avg_queue * 35)), 10), border_radius=4)
        y += 48

        if state.get("condition") == "maestro_fl":
            self._draw_text("FL status", (x + 24, y), self.font, self.TEXT_COLOR)
            y += 30
            self._draw_text(f"Round {fl.get('round', 0)}", (x + 24, y), self.font, self.TEXT_COLOR)
            y += 30
            if state.get("priority_active") and int(time.time() * 3) % 2 == 0:
                self._draw_text("PRIORITY ACTIVE", (x + 24, y), self.font_large, self.TL_RED)
                y += 42

        self._draw_text(f"Step {state.get('step', 0)}", (x + 24, self.canvas_height - 48), self.font, self.MUTED_TEXT)

    def _draw_metric(self, x: int, y: int, label: str, value: str, color: Color) -> int:
        self._draw_text(label, (x, y), self.font_small, self.MUTED_TEXT)
        self._draw_text(value, (x, y + 18), self.font_metric, color)
        return y + 78

    def _draw_status_row(self, x: int, y: int, label: str, value: str, color: Color) -> None:
        self._draw_text(label, (x, y), self.font, self.TEXT_COLOR)
        self._draw_text(value, (x + 160, y), self.font, color)

    def _draw_results_panel(self, rect: Tuple[int, int, int, int], title: str, result: Dict, good: bool) -> None:
        x, y, w, h = rect
        pygame.draw.rect(self.screen, self.PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(self.screen, self.TL_GREEN if good else self.TL_RED, rect, 2, border_radius=8)
        self._draw_text(title, (x + 28, y + 30), self.font_large, self.TL_GREEN if good else self.TL_RED)
        rows = [
            ("Travel time", f"{float(result.get('ambulance_travel_time', 0) or 0):.0f}s"),
            ("Wait time", f"{float(result.get('ambulance_waiting_time', 0) or 0):.1f}s"),
            ("Avg queue", f"{float(result.get('avg_queue_length', 0) or 0):.2f}"),
        ]
        mark = "OK" if good else "X"
        for idx, (label, value) in enumerate(rows):
            yy = y + 130 + idx * 95
            self._draw_text(mark, (x + 32, yy), self.font_large, self.TL_GREEN if good else self.TL_RED)
            self._draw_text(label, (x + 88, yy), self.font, self.MUTED_TEXT)
            self._draw_text(value, (x + 88, yy + 26), self.font_metric, self.TEXT_COLOR)

    def _draw_text(self, text: str, pos: Tuple[int, int], font, color: Color, center: bool = False) -> None:
        surface = font.render(str(text), True, color)
        rect = surface.get_rect()
        rect.center = pos if center else rect.center
        if not center:
            rect.topleft = pos
        self.screen.blit(surface, rect)

    def _draw_rotated_rect(self, center: Tuple[int, int], width: float, height: float, angle: float, color: Color) -> None:
        # SUMO angle is clockwise from north; Pygame rotation here uses screen coordinates.
        theta = math.radians(angle - 90.0)
        c, s = math.cos(theta), math.sin(theta)
        hw, hh = width / 2.0, height / 2.0
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        pts = []
        for x, y in corners:
            pts.append((int(center[0] + x * c - y * s), int(center[1] + x * s + y * c)))
        pygame.gfxdraw.filled_polygon(self.screen, pts, color)
        pygame.gfxdraw.aapolygon(self.screen, pts, color)

    def _world_to_screen(self, world: np.ndarray) -> Tuple[int, int]:
        screen_center = np.array([self.canvas_width / 2, self.canvas_height / 2], dtype=float)
        screen = (world - self.camera_center) * self.zoom + screen_center + self.camera_pan
        return int(screen[0]), int(screen[1])

    def _screen_to_world(self, screen: np.ndarray) -> np.ndarray:
        screen_center = np.array([self.canvas_width / 2, self.canvas_height / 2], dtype=float)
        return (screen - screen_center - self.camera_pan) / max(self.zoom, 0.001) + self.camera_center

    def _world_to_screen_tuple(self, point: Tuple[int, int]) -> Tuple[int, int]:
        return self._world_to_screen(np.array(point, dtype=float))

    @staticmethod
    def _last_segment(points: List[Tuple[int, int]], fraction: float = 0.35) -> List[Tuple[int, int]]:
        if len(points) <= 2:
            return points
        start = max(0, int(len(points) * (1.0 - fraction)) - 1)
        return points[start:]
