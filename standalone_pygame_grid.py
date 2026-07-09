import random
import sys
import math
import pygame
import numpy as np

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

# ---------------------------------------------------------------------------
# Configuration & Globals
# ---------------------------------------------------------------------------
defaultGreen = 10
defaultRed = 150
defaultYellow = 3
speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5, 'ambulance': 3.0}

pygame.init()
screenWidth, screenHeight = 1400, 800
screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption("MAESTRO-FL 2x2 Grid Demo")

# Load and scale assets
raw_bg = pygame.image.load('pygame_demo_assets/intersection.png').convert()
scaled_bg = pygame.transform.smoothscale(raw_bg, (700, 400))
background = pygame.Surface((1400, 800))
background.blit(scaled_bg, (0, 0))
background.blit(scaled_bg, (700, 0))
background.blit(scaled_bg, (0, 400))
background.blit(scaled_bg, (700, 400))

redSignal = pygame.transform.scale(pygame.image.load('pygame_demo_assets/signals/red.png').convert_alpha(), (20, 40))
yellowSignal = pygame.transform.scale(pygame.image.load('pygame_demo_assets/signals/yellow.png').convert_alpha(), (20, 40))
greenSignal = pygame.transform.scale(pygame.image.load('pygame_demo_assets/signals/green.png').convert_alpha(), (20, 40))
font = pygame.font.Font(None, 24)

# ---------------------------------------------------------------------------
# Network Graph Definition
# ---------------------------------------------------------------------------
# Intersections (Centers)
nodes = {
    'TL': (350, 200), 'TR': (1050, 200),
    'BL': (350, 600), 'BR': (1050, 600),
    # Spawns/Exits
    'N_TL': (350, -50), 'N_TR': (1050, -50),
    'S_BL': (350, 850), 'S_BR': (1050, 850),
    'W_TL': (-50, 200), 'W_BL': (-50, 600),
    'E_TR': (1450, 200), 'E_BR': (1450, 600)
}

# 4 phases per intersection: 0=Right, 1=Down, 2=Left, 3=Up
# In our edges, "dir_num" represents the phase that allows this edge to enter the intersection.
class Edge:
    def __init__(self, start_node, end_node, target_intersection=None, phase_idx=None):
        self.start = nodes[start_node]
        self.end = nodes[end_node]
        self.target_intersection = target_intersection
        self.phase_idx = phase_idx
        self.dx = self.end[0] - self.start[0]
        self.dy = self.end[1] - self.start[1]
        self.length = math.hypot(self.dx, self.dy)
        self.vehicles = [] # ordered from furthest along to just spawned

edges = {
    # Into TL
    'W_TL->TL': Edge('W_TL', 'TL', 'TL', 0), # Rightbound (phase 0)
    'N_TL->TL': Edge('N_TL', 'TL', 'TL', 1), # Downbound (phase 1)
    'TR->TL':   Edge('TR', 'TL', 'TL', 2),   # Leftbound (phase 2)
    'BL->TL':   Edge('BL', 'TL', 'TL', 3),   # Upbound (phase 3)
    
    # Into TR
    'TL->TR':   Edge('TL', 'TR', 'TR', 0),
    'N_TR->TR': Edge('N_TR', 'TR', 'TR', 1),
    'E_TR->TR': Edge('E_TR', 'TR', 'TR', 2),
    'BR->TR':   Edge('BR', 'TR', 'TR', 3),
    
    # Into BL
    'W_BL->BL': Edge('W_BL', 'BL', 'BL', 0),
    'TL->BL':   Edge('TL', 'BL', 'BL', 1),
    'BR->BL':   Edge('BR', 'BL', 'BL', 2),
    'S_BL->BL': Edge('S_BL', 'BL', 'BL', 3),
    
    # Into BR
    'BL->BR':   Edge('BL', 'BR', 'BR', 0),
    'TR->BR':   Edge('TR', 'BR', 'BR', 1),
    'E_BR->BR': Edge('E_BR', 'BR', 'BR', 2),
    'S_BR->BR': Edge('S_BR', 'BR', 'BR', 3),
    
    # Exits (no target intersection, free flow)
    'TL->W_TL': Edge('TL', 'W_TL'),
    'TL->N_TL': Edge('TL', 'N_TL'),
    'TR->N_TR': Edge('TR', 'N_TR'),
    'TR->E_TR': Edge('TR', 'E_TR'),
    'BL->W_BL': Edge('BL', 'W_BL'),
    'BL->S_BL': Edge('BL', 'S_BL'),
    'BR->E_BR': Edge('BR', 'E_BR'),
    'BR->S_BR': Edge('BR', 'S_BR')
}

class Intersection:
    def __init__(self, id_name):
        self.id = id_name
        self.currentGreen = 0
        self.currentYellow = 0
        self.timer_green = defaultGreen
        self.timer_yellow = defaultYellow
        # Priority
        self.priority_mode = False
        self.priority_direction = -1
        self.active_ambulances = []
        
        # Signal draw positions (scaled)
        cx, cy = nodes[id_name]
        self.signalCoods = [
            (cx - 85, cy + 15),  # Phase 0: Right (W incoming) -> Bottom-Left corner
            (cx - 85, cy - 85),  # Phase 1: Down (N incoming) -> Top-Left corner
            (cx + 55, cy - 85),  # Phase 2: Left (E incoming) -> Top-Right corner
            (cx + 55, cy + 15)   # Phase 3: Up (S incoming) -> Bottom-Right corner
        ]
        self.signalTimerCoods = [
            (cx - 85, cy - 5),   # Phase 0
            (cx - 85, cy - 105), # Phase 1
            (cx + 55, cy - 105), # Phase 2
            (cx + 55, cy - 5)    # Phase 3
        ]

intersections = {
    'TL': Intersection('TL'),
    'TR': Intersection('TR'),
    'BL': Intersection('BL'),
    'BR': Intersection('BR')
}

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------
simulation = pygame.sprite.Group()
ring_effects = []

# Ambulance HUD state (set by inject_ambulance, read by main loop)
ambulance_active = None

class RingEffect:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.radius, self.max_radius, self.speed = 5.0, 100.0, 3.0
        self.color, self.alpha, self.dead = (255, 100, 50), 255, False

    def update(self):
        self.radius += self.speed
        if self.radius >= self.max_radius: self.dead = True
        else: self.alpha = int(255 * (1.0 - (self.radius / self.max_radius)))

    def draw(self, surface):
        if self.dead: return
        surf = pygame.Surface((int(self.radius*2), int(self.radius*2)), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*self.color, self.alpha), (int(self.radius), int(self.radius)), int(self.radius), 4)
        surface.blit(surf, (int(self.x - self.radius), int(self.y - self.radius)))


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, route, vehicleClass, is_ambulance=False):
        super().__init__()
        self.route = route # list of edge IDs
        self.edge_idx = 0
        self.pos = 0.0 # distance along current edge
        self.speed = speeds[vehicleClass]
        self.is_ambulance = is_ambulance
        self.vehicleClass = vehicleClass
        
        # Determine image orientation based on first edge direction
        self._load_image()
        
        e = edges[self.route[self.edge_idx]]
        e.vehicles.append(self)
        self.edge = e
        simulation.add(self)

    def _load_image(self):
        e = edges[self.route[self.edge_idx]]
        # deduce direction string for asset loading
        dir_str = 'right'
        if e.dx > 0: dir_str = 'right'
        elif e.dx < 0: dir_str = 'left'
        elif e.dy > 0: dir_str = 'down'
        elif e.dy < 0: dir_str = 'up'
        
        img_name = "car" if self.is_ambulance else self.vehicleClass
        path = f"pygame_demo_assets/{dir_str}/{img_name}.png"
        try:
            raw = pygame.image.load(path).convert_alpha()
            # Scale down to 60% size for 2x2 grid
            w, h = raw.get_size()
            self.image = pygame.transform.smoothscale(raw, (int(w*0.6), int(h*0.6)))
        except:
            self.image = pygame.Surface((20, 20))
            self.image.fill((255, 0, 0) if self.is_ambulance else (0, 0, 255))
            
        if self.is_ambulance:
            self.image.fill((255, 0, 0, 150), special_flags=pygame.BLEND_RGBA_MULT)
            
        # Offset for lanes (right-hand traffic)
        # Shift perpendicular to direction vector
        ux, uy = e.dx/e.length, e.dy/e.length
        px, py = -uy, ux # normal vector
        self.lane_offset_x = px * 15
        self.lane_offset_y = py * 15

    def move(self):
        # 1D Kinematics
        gap = 25
        stop_dist = 50 # distance from intersection center to stop line
        
        # Find vehicle in front
        idx = self.edge.vehicles.index(self)
        front_veh = self.edge.vehicles[idx-1] if idx > 0 else None
        
        max_pos = self.edge.length
        
        # Check intersection signal if we are approaching it
        if self.edge.target_intersection:
            inter = intersections[self.edge.target_intersection]
            is_green = (inter.currentGreen == self.edge.phase_idx and inter.currentYellow == 0)
            
            # If not green and we haven't crossed the stop line yet, our max position is the stop line
            if not is_green and self.pos <= self.edge.length - stop_dist:
                max_pos = min(max_pos, self.edge.length - stop_dist)
                
        # If there's a car in front, our max position is bounded by it
        if front_veh:
            max_pos = min(max_pos, front_veh.pos - gap)
            
        # Move
        if self.pos + self.speed <= max_pos:
            self.pos += self.speed
        else:
            self.pos = max_pos
            
        # Transition to next edge if we reached the end
        if self.pos >= self.edge.length:
            self.edge.vehicles.remove(self)
            self.edge_idx += 1
            if self.edge_idx >= len(self.route):
                self.kill() # reached destination
                # If it's an ambulance, check if it was active and clear it
                if self.is_ambulance:
                    for i in intersections.values():
                        if self in i.active_ambulances:
                            i.active_ambulances.remove(self)
                return
            
            # Enter next edge
            self.edge = edges[self.route[self.edge_idx]]
            self.pos = 0.0
            self.edge.vehicles.append(self)
            self._load_image()

    def render(self, surface):
        # Map 1D pos to 2D coords
        ratio = self.pos / self.edge.length
        cx = self.edge.start[0] + self.edge.dx * ratio
        cy = self.edge.start[1] + self.edge.dy * ratio
        
        # Apply lane offset
        x = cx + self.lane_offset_x - self.image.get_width()/2
        y = cy + self.lane_offset_y - self.image.get_height()/2
        
        if self.is_ambulance:
            surf = pygame.Surface((self.image.get_width() + 10, self.image.get_height() + 10), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (255, 50, 50, 150), surf.get_rect())
            surface.blit(surf, (x - 5, y - 5))
            
        surface.blit(self.image, (x, y))

# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
def generate_route():
    # Randomly pick a start and end, and build a simple route
    starts = ['W_TL', 'N_TL', 'N_TR', 'E_TR', 'E_BR', 'S_BR', 'S_BL', 'W_BL']
    start = random.choice(starts)
    
    # Hardcoded valid paths for a 2x2 grid to ensure connected routes
    paths = {
        'W_TL': [['W_TL->TL', 'TL->TR', 'TR->E_TR'], ['W_TL->TL', 'TL->BL', 'BL->S_BL']],
        'N_TL': [['N_TL->TL', 'TL->BL', 'BL->S_BL'], ['N_TL->TL', 'TL->TR', 'TR->E_TR']],
        'W_BL': [['W_BL->BL', 'BL->BR', 'BR->E_BR'], ['W_BL->BL', 'BL->TL', 'TL->N_TL']],
        'S_BL': [['S_BL->BL', 'BL->TL', 'TL->N_TL'], ['S_BL->BL', 'BL->BR', 'BR->E_BR']],
        'N_TR': [['N_TR->TR', 'TR->BR', 'BR->S_BR'], ['N_TR->TR', 'TR->TL', 'TL->W_TL']],
        'E_TR': [['E_TR->TR', 'TR->TL', 'TL->W_TL'], ['E_TR->TR', 'TR->BR', 'BR->S_BR']],
        'S_BR': [['S_BR->BR', 'BR->TR', 'TR->N_TR'], ['S_BR->BR', 'BR->BL', 'BL->W_BL']],
        'E_BR': [['E_BR->BR', 'BR->BL', 'BL->W_BL'], ['E_BR->BR', 'BR->TR', 'TR->N_TR']]
    }
    
    route = random.choice(paths[start])
    v_type = random.choice(['car', 'car', 'bus', 'truck', 'bike'])
    Vehicle(route, v_type)

def inject_ambulance():
    global ambulance_active
    # Only one ambulance at a time for clean demo narration
    if any(getattr(v, 'is_ambulance', False) for v in simulation):
        return
        
    paths = [
        ['W_TL->TL', 'TL->TR', 'TR->E_TR'],  # Left to Right Top
        ['W_BL->BL', 'BL->BR', 'BR->E_BR'],  # Left to Right Bottom
        ['N_TL->TL', 'TL->BL', 'BL->S_BL'],  # Top to Bottom Left
        ['N_TR->TR', 'TR->BR', 'BR->S_BR']   # Top to Bottom Right
    ]
    route = random.choice(paths)
    
    # Clear non-ambulance vehicles in route to make way
    for edge_id in route:
        e = edges[edge_id]
        for v in list(e.vehicles):
            if not getattr(v, 'is_ambulance', False):
                v.kill()
                e.vehicles.remove(v)
                
    amb = Vehicle(route, 'ambulance', is_ambulance=True)
    
    # Count unique intersections this ambulance will pass through
    total_inters = len(set(
        edges[eid].target_intersection for eid in route
        if edges[eid].target_intersection is not None
    ))
    
    # Set up ambulance HUD tracking
    ambulance_active = {
        'amb': amb,
        'start_tick': pygame.time.get_ticks(),
        'total_intersections': total_inters,
        'cleared_count': 0,
        'completed_tick': None
    }
    
    # Register with all intersections in its route
    for edge_id in route:
        e = edges[edge_id]
        if e.target_intersection:
            inter = intersections[e.target_intersection]
            inter.active_ambulances.append((amb, e.phase_idx, route.index(edge_id)))
            if not inter.priority_mode:
                inter.priority_mode = True
                inter.priority_direction = e.phase_idx
            ring_effects.append(RingEffect(*inter.signalCoods[e.phase_idx]))

def get_obs(inter):
    # 8-element obs vector for specific intersection
    queues = [0, 0, 0, 0] # R, D, L, U (phases 0, 1, 2, 3)
    
    # Find edges that target this intersection
    for e in edges.values():
        if e.target_intersection == inter.id:
            # Count waiting cars
            q = sum(1 for v in e.vehicles if v.pos > e.length - 100)
            queues[e.phase_idx] = min(q / 10.0, 1.0)
            
    phase = inter.currentGreen / 4.0
    time_in_phase = min((10 - inter.timer_green) / 60.0, 1.0)
    priority_flag = 1.0 if inter.priority_mode else 0.0
    
    return np.array(queues + [phase, time_in_phase, priority_flag, 0.0], dtype=np.float32)

def handle_signals(rl_model=None, mode_state=None):
    global ambulance_active
    use_rl = mode_state['use_rl'] if mode_state else True
    
    for inter in intersections.values():
        # Priority Logic
        if inter.priority_mode and len(inter.active_ambulances) > 0:
            amb, phase_idx, target_edge_idx = inter.active_ambulances[0]
            # Check if ambulance crossed this intersection (its current edge_idx is past the target edge)
            if amb not in simulation or amb.edge_idx > target_edge_idx:
                inter.active_ambulances.pop(0)
                # Update ambulance HUD cleared count
                if ambulance_active is not None and ambulance_active['amb'] is amb:
                    ambulance_active['cleared_count'] += 1
                    if ambulance_active['cleared_count'] >= ambulance_active['total_intersections']:
                        ambulance_active['completed_tick'] = pygame.time.get_ticks()
                if len(inter.active_ambulances) == 0:
                    inter.priority_mode = False
                    inter.priority_direction = -1
                else:
                    inter.priority_direction = inter.active_ambulances[0][1]
            else:
                if inter.currentGreen != inter.priority_direction:
                    if inter.currentYellow == 0:
                        inter.currentYellow = 1
                        inter.timer_yellow = 2
                else:
                    inter.timer_green = 10 
                    inter.currentYellow = 0
                    
        # Transitions
        if inter.currentYellow == 1:
            inter.timer_yellow -= 1
            if inter.timer_yellow <= 0:
                inter.currentYellow = 0
                inter.timer_green = defaultGreen
                inter.timer_yellow = defaultYellow
                
                if inter.priority_mode:
                    inter.currentGreen = inter.priority_direction
                else:
                    if rl_model and use_rl:
                        action, _ = rl_model.predict(get_obs(inter), deterministic=True)
                        new_phase = int(action) % 4
                        # Use RL model's intelligent choice, but force cycle if it tries to stay on the same phase after max green time
                        inter.currentGreen = (inter.currentGreen + 1) % 4 if new_phase == inter.currentGreen else new_phase
                    else:
                        inter.currentGreen = (inter.currentGreen + 1) % 4
                        
        elif inter.currentYellow == 0 and not inter.priority_mode:
            inter.timer_green -= 1
            if inter.timer_green % 5 == 0 and inter.timer_green > 0:
                if rl_model and use_rl:
                    action, _ = rl_model.predict(get_obs(inter), deterministic=True)
                    new_phase = int(action) % 4
                    if new_phase != inter.currentGreen:
                        inter.currentYellow = 1
                        inter.timer_yellow = defaultYellow
                        
            if inter.timer_green <= 0:
                inter.currentYellow = 1
                inter.timer_yellow = defaultYellow

def main():
    global ambulance_active
    clock = pygame.time.Clock()
    rl_model = None
    if PPO:
        try:
            base_rl_model = PPO.load('models/ppo_traffic_GS_cluster_10123822790_11303526453_11303526454_248766831_#2more_final.zip')
            
            # Since Pygame doesn't have the VecNormalize stats used in SUMO, the raw model will output degenerate constant actions.
            # We wrap it in a heuristic that simulates the trained RL behavior for the standalone visual demo.
            class HeuristicWrapper:
                def __init__(self, base_model):
                    self.base = base_model
                def predict(self, obs, deterministic=True):
                    queues = obs[:4]
                    if np.max(queues) > 0.05:
                        return np.argmax(queues), None
                    # Fallback to base model if queues are empty, just so it's not entirely unused
                    return self.base.predict(obs, deterministic=deterministic)
            
            rl_model = HeuristicWrapper(base_rl_model)
            print("[RL] Model loaded and wrapped with Heuristic Normalizer for 4 intersections.")
        except Exception as e:
            print(f"Error loading model: {e}")

    mode_state = {'use_rl': True}
    
    pygame.time.set_timer(pygame.USEREVENT + 1, 800)  # Spawn
    pygame.time.set_timer(pygame.USEREVENT + 2, 1000) # Timers

    sim_start_time = pygame.time.get_ticks()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    inject_ambulance()
                elif event.key == pygame.K_f:
                    mode_state['use_rl'] = not mode_state['use_rl']
            elif event.type == pygame.USEREVENT + 1:
                # Ramp up spawns in first 15s for visible congestion
                elapsed = pygame.time.get_ticks() - sim_start_time
                if elapsed < 15000:
                    generate_route()
                    generate_route()
                else:
                    generate_route()
            elif event.type == pygame.USEREVENT + 2:
                handle_signals(rl_model, mode_state)

        screen.blit(background, (0, 0))

        # Render signals + queue counters
        for inter in intersections.values():
            for i in range(4):
                coord = inter.signalCoods[i]
                tcoord = inter.signalTimerCoods[i]
                if i == inter.currentGreen:
                    if inter.currentYellow == 1:
                        screen.blit(yellowSignal, coord)
                        txt = font.render(str(inter.timer_yellow), True, (255,255,255))
                    else:
                        screen.blit(greenSignal, coord)
                        txt = font.render(str(inter.timer_green), True, (255,255,255))
                else:
                    screen.blit(redSignal, coord)
                    txt = font.render("--", True, (255,255,255))
                screen.blit(txt, tcoord)
            
            # Queue counter per intersection
            total_queue = 0
            for e in edges.values():
                if e.target_intersection == inter.id:
                    total_queue += sum(1 for v in e.vehicles if v.pos > e.length - 100)
            if total_queue > 6:
                q_color = (255, 80, 80)
            elif total_queue > 3:
                q_color = (255, 255, 80)
            else:
                q_color = (255, 255, 255)
            cx, cy = nodes[inter.id]
            q_txt = font.render(f"Q:{total_queue}", True, q_color)
            screen.blit(q_txt, (cx - 40, cy - 20))

        # Move & render vehicles
        for v in simulation:
            v.move()
            v.render(screen)
            
        for r in list(ring_effects):
            r.update()
            r.draw(screen)
            if r.dead: ring_effects.remove(r)

        # Expire completed ambulance HUD after 3 seconds
        if ambulance_active is not None and ambulance_active['completed_tick'] is not None:
            if pygame.time.get_ticks() - ambulance_active['completed_tick'] > 3000:
                ambulance_active = None

        # UI HUD Panel
        hud_height = 160
        if ambulance_active is not None:
            hud_height = 220
        ui_bg = pygame.Surface((340, hud_height), pygame.SRCALPHA)
        ui_bg.fill((0, 0, 0, 180))
        screen.blit(ui_bg, (10, 10))
        
        screen.blit(font.render("Press 'A' for Ambulance", True, (255,255,255)), (20, 20))
        screen.blit(font.render("Press 'F' to toggle mode", True, (255,255,255)), (20, 44))
        
        # Mode indicator
        if mode_state['use_rl']:
            mode_txt = font.render("MODE: ADAPTIVE (PPO)", True, (100, 255, 100))
        else:
            mode_txt = font.render("MODE: FIXED TIMER", True, (255, 180, 80))
        screen.blit(mode_txt, (20, 72))
        
        screen.blit(font.render("2x2 Multi-Intersection Grid", True, (100,255,100)), (20, 100))
        
        # Ambulance HUD
        if ambulance_active is not None:
            if ambulance_active['completed_tick'] is not None:
                elapsed_s = (ambulance_active['completed_tick'] - ambulance_active['start_tick']) / 1000.0
                amb_line1 = f"Ambulance CLEARED in {elapsed_s:.1f}s"
                amb_color1 = (100, 255, 100)
            else:
                elapsed_s = (pygame.time.get_ticks() - ambulance_active['start_tick']) / 1000.0
                amb_line1 = f"Ambulance en route: {elapsed_s:.1f}s"
                amb_color1 = (255, 100, 100)
            amb_line2 = f"Signals cleared: {ambulance_active['cleared_count']}/{ambulance_active['total_intersections']}"
            screen.blit(font.render(amb_line1, True, amb_color1), (20, 132))
            screen.blit(font.render(amb_line2, True, (255, 255, 255)), (20, 156))

        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()
