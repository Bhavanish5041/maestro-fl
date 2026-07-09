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
defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
defaultRed = 150
defaultYellow = 3

signals = []
noOfSignals = 4
currentGreen = 0
currentYellow = 0

# Average speeds of vehicles
speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5, 'ambulance': 3.0}

# Coordinates of vehicles' start
default_x = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
default_y = {'right': [348, 370, 398], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

vehicles = {'right': {0: [], 1: [], 2: [], 'crossed': 0},
            'down': {0: [], 1: [], 2: [], 'crossed': 0},
            'left': {0: [], 1: [], 2: [], 'crossed': 0},
            'up': {0: [], 1: [], 2: [], 'crossed': 0}}
vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Coordinates of signal image and timer text
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

stoppingGap = 20
movingGap = 20

# ---------------------------------------------------------------------------
# Priority Logic Globals
# ---------------------------------------------------------------------------
active_ambulances = []
priority_mode = False
priority_direction = -1
ring_effects = []

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""

class RingEffect:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 10.0
        self.max_radius = 150.0
        self.speed = 3.0
        self.color = (255, 100, 50)
        self.alpha = 255
        self.dead = False

    def update(self):
        self.radius += self.speed
        if self.radius >= self.max_radius:
            self.dead = True
        else:
            self.alpha = int(255 * (1.0 - (self.radius / self.max_radius)))

    def draw(self, surface):
        if self.dead: return
        surf = pygame.Surface((int(self.radius * 2), int(self.radius * 2)), pygame.SRCALPHA)
        c = (*self.color, self.alpha)
        pygame.draw.circle(surf, c, (int(self.radius), int(self.radius)), int(self.radius), width=4)
        surface.blit(surf, (int(self.x - self.radius), int(self.y - self.radius)))


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, is_ambulance=False):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.crossed = 0
        self.is_ambulance = is_ambulance
        
        # Load image
        img_name = "car" if is_ambulance else vehicleClass
        path = f"pygame_demo_assets/{direction}/{img_name}.png"
        
        try:
            self.image = pygame.image.load(path).convert_alpha()
        except:
            # fallback if image missing
            self.image = pygame.Surface((30, 30))
            self.image.fill((255, 0, 0) if is_ambulance else (0, 0, 255))
            
        if self.is_ambulance:
            # Color overlay to make ambulance red
            self.image.fill((255, 0, 0, 100), special_flags=pygame.BLEND_RGBA_MULT)

        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1

        # Calculate dynamic spawn position
        last_veh = vehicles[direction][lane][-2] if self.index > 0 else None
        
        if direction == 'right':
            base = default_x['right'][lane]
            self.y = default_y['right'][lane]
            if last_veh and last_veh.x < base + self.image.get_width() + stoppingGap:
                self.x = last_veh.x - self.image.get_width() - stoppingGap
            else:
                self.x = base
                
        elif direction == 'left':
            base = default_x['left'][lane]
            self.y = default_y['left'][lane]
            if last_veh and last_veh.x > base - self.image.get_width() - stoppingGap:
                self.x = last_veh.x + self.image.get_width() + stoppingGap
            else:
                self.x = base
                
        elif direction == 'down':
            base = default_y['down'][lane]
            self.x = default_x['down'][lane]
            if last_veh and last_veh.y < base + self.image.get_height() + stoppingGap:
                self.y = last_veh.y - self.image.get_height() - stoppingGap
            else:
                self.y = base
                
        elif direction == 'up':
            base = default_y['up'][lane]
            self.x = default_x['up'][lane]
            if last_veh and last_veh.y > base - self.image.get_height() - stoppingGap:
                self.y = last_veh.y + self.image.get_height() + stoppingGap
            else:
                self.y = base

        self._calculate_stop()

        simulation.add(self)

    def _calculate_stop(self):
        if len(vehicles[self.direction][self.lane]) > 1 and vehicles[self.direction][self.lane][self.index - 1].crossed == 0:
            prev_veh = vehicles[self.direction][self.lane][self.index - 1]
            if self.direction == 'right':
                self.stop = prev_veh.stop - prev_veh.image.get_width() - stoppingGap
            elif self.direction == 'left':
                self.stop = prev_veh.stop + prev_veh.image.get_width() + stoppingGap
            elif self.direction == 'down':
                self.stop = prev_veh.stop - prev_veh.image.get_height() - stoppingGap
            elif self.direction == 'up':
                self.stop = prev_veh.stop + prev_veh.image.get_height() + stoppingGap
        else:
            self.stop = defaultStop[self.direction]

    def render(self, screen):
        if self.is_ambulance:
            # draw a glowing halo behind ambulance
            surf = pygame.Surface((self.image.get_width() + 20, self.image.get_height() + 20), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (255, 50, 50, 150), surf.get_rect())
            screen.blit(surf, (self.x - 10, self.y - 10))
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        # dynamic stop updates
        self._calculate_stop()
        
        green_active = (currentGreen == self.direction_number and currentYellow == 0)

        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.image.get_width() > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if (self.x + self.image.get_width() <= self.stop or self.crossed == 1 or green_active) and \
               (self.index == 0 or self.x + self.image.get_width() < (vehicles[self.direction][self.lane][self.index - 1].x - movingGap)):
                self.x += self.speed

        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.image.get_height() > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if (self.y + self.image.get_height() <= self.stop or self.crossed == 1 or green_active) and \
               (self.index == 0 or self.y + self.image.get_height() < (vehicles[self.direction][self.lane][self.index - 1].y - movingGap)):
                self.y += self.speed

        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if (self.x >= self.stop or self.crossed == 1 or green_active) and \
               (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].image.get_width() + movingGap)):
                self.x -= self.speed

        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if (self.y >= self.stop or self.crossed == 1 or green_active) and \
               (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].image.get_height() + movingGap)):
                self.y -= self.speed


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def initialize_signals():
    signals.clear()
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen[1])
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2])
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
    signals.append(ts4)

def generate_vehicle():
    vehicle_type = random.randint(0, 3)
    lane_number = random.randint(1, 2)
    temp = random.randint(0, 99)
    if temp < 25:
        dir_num = 0
    elif temp < 50:
        dir_num = 1
    elif temp < 75:
        dir_num = 2
    else:
        dir_num = 3
    Vehicle(lane_number, vehicleTypes[vehicle_type], dir_num, directionNumbers[dir_num])

def inject_ambulance():
    global priority_mode, priority_direction
    dir_num = random.randint(0, 3)
    lane_number = 1  # middle lane
    
    direction = directionNumbers[dir_num]
    
    # Emergency! Clear the lane so the ambulance spawns immediately on screen
    for v in list(vehicles[direction][lane_number]):
        if v in simulation:
            simulation.remove(v)
    vehicles[direction][lane_number].clear()
    
    new_ambulance = Vehicle(lane_number, 'ambulance', dir_num, direction, is_ambulance=True)
    active_ambulances.append(new_ambulance)
    
    if not priority_mode:
        priority_mode = True
        priority_direction = dir_num
    
    # Spawn a ring effect at the center
    ring_effects.append(RingEffect(700, 400))
    print(f"[Priority] Ambulance approaching from {direction.upper()}")

def get_obs():
    # Construct 8-element observation vector expected by MAESTRO-FL PPO model
    # obs: 4 lane queues + current_phase + time_in_phase + priority_flag + predicted_congestion
    
    queues = []
    # Order: right(0), down(1), left(2), up(3)
    for direction in ['right', 'down', 'left', 'up']:
        q_count = 0
        for lane in [0, 1, 2]:
            for v in vehicles[direction][lane]:
                if v.crossed == 0:  # waiting or approaching
                    q_count += 1
        queues.append(min(q_count / 20.0, 1.0))
        
    phase = currentGreen / 4.0
    time_in_phase = min((10 - signals[currentGreen].green) / 60.0, 1.0)
    priority_flag = 1.0 if priority_mode else 0.0
    predicted_congestion = 0.0 # dummy
    
    return np.array(queues + [phase, time_in_phase, priority_flag, predicted_congestion], dtype=np.float32)

def handle_signal_timers(rl_model=None):
    global currentGreen, currentYellow, priority_mode, priority_direction
    
    # Priority Override Logic (Ambulance forces its way)
    if priority_mode and len(active_ambulances) > 0:
        current_amb = active_ambulances[0]
        if current_amb.crossed == 1 or current_amb not in simulation:
            print(f"[Priority] Ambulance from {directionNumbers[priority_direction].upper()} crossed.")
            active_ambulances.pop(0)
            
            if len(active_ambulances) == 0:
                print("[Priority] All ambulances cleared. Reverting to RL model control.")
                priority_mode = False
                priority_direction = -1
            else:
                priority_direction = active_ambulances[0].direction_number
                print(f"[Priority] Shifting priority to next ambulance from {directionNumbers[priority_direction].upper()}.")
        else:
            if currentGreen != priority_direction:
                if currentYellow == 0:
                    currentYellow = 1
                    signals[currentGreen].yellow = 2
            else:
                signals[currentGreen].green = 10 
                currentYellow = 0

    # Yellow Transition
    if currentYellow == 1:
        signals[currentGreen].yellow -= 1
        if signals[currentGreen].yellow <= 0:
            currentYellow = 0
            signals[currentGreen].green = defaultGreen[currentGreen]
            signals[currentGreen].yellow = defaultYellow
            signals[currentGreen].red = defaultRed
            
            # Switch phase
            if priority_mode:
                currentGreen = priority_direction
            else:
                # RL Action determines the next phase
                if rl_model is not None:
                    action, _ = rl_model.predict(get_obs(), deterministic=True)
                    currentGreen = int(action) % noOfSignals
                else:
                    currentGreen = (currentGreen + 1) % noOfSignals
                    
            nextG = (currentGreen + 1) % noOfSignals
            signals[nextG].red = signals[currentGreen].yellow + signals[currentGreen].green
            
    # Green Phase logic
    elif currentYellow == 0 and not priority_mode:
        signals[currentGreen].green -= 1
        
        # Every 5 seconds, poll the RL model for a decision
        if signals[currentGreen].green % 5 == 0 and signals[currentGreen].green > 0:
            if rl_model is not None:
                action, _ = rl_model.predict(get_obs(), deterministic=True)
                new_phase = int(action) % noOfSignals
                if new_phase != currentGreen:
                    currentYellow = 1
                    signals[currentGreen].yellow = defaultYellow

        if signals[currentGreen].green <= 0:
            currentYellow = 1
            signals[currentGreen].yellow = defaultYellow


def main():
    pygame.init()
    global simulation
    simulation = pygame.sprite.Group()
    
    screenWidth, screenHeight = 1400, 800
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("MAESTRO-FL Standalone Pygame Priority Demo")
    
    try:
        background = pygame.image.load('pygame_demo_assets/intersection.png').convert()
        redSignal = pygame.image.load('pygame_demo_assets/signals/red.png').convert_alpha()
        yellowSignal = pygame.image.load('pygame_demo_assets/signals/yellow.png').convert_alpha()
        greenSignal = pygame.image.load('pygame_demo_assets/signals/green.png').convert_alpha()
    except Exception as e:
        print(f"Error loading assets: {e}")
        return

    font = pygame.font.Font(None, 40)
    clock = pygame.time.Clock()
    
    # Load MAESTRO-FL RL Model
    rl_model = None
    if PPO is not None:
        try:
            # Load one of the trained models from the repo
            model_path = 'models/ppo_traffic_GS_cluster_10123822790_11303526453_11303526454_248766831_#2more_final.zip'
            rl_model = PPO.load(model_path)
            print(f"[RL] Successfully loaded MAESTRO-FL PPO Model from {model_path}")
        except Exception as e:
            print(f"[RL] Could not load model: {e}")
    else:
        print("[RL] stable_baselines3 not installed, running on fixed timers.")

    initialize_signals()

    # Timers using Pygame UserEvents
    SPAWN_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(SPAWN_EVENT, 1200) # spawn vehicle every 1.2s
    
    TIMER_EVENT = pygame.USEREVENT + 2
    pygame.time.set_timer(TIMER_EVENT, 1000) # 1 second clock tick

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    inject_ambulance()
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    sys.exit()
            elif event.type == SPAWN_EVENT:
                generate_vehicle()
            elif event.type == TIMER_EVENT:
                handle_signal_timers(rl_model)

        screen.blit(background, (0, 0))

        # Cleanup off-screen vehicles
        for vehicle in list(simulation):
            if vehicle.x < -100 or vehicle.x > screenWidth + 100 or vehicle.y < -100 or vehicle.y > screenHeight + 100:
                simulation.remove(vehicle)
                vehicles[vehicle.direction][vehicle.lane].remove(vehicle)
                # recalculate indices
                for idx, v in enumerate(vehicles[vehicle.direction][vehicle.lane]):
                    v.index = idx

        # Draw signals and timers
        for i in range(noOfSignals):
            if i == currentGreen:
                if currentYellow == 1:
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if signals[i].red <= 10:
                    signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])

            # Draw timer text
            color = (255, 100, 100) if (priority_mode and i == priority_direction) else (255, 255, 255)
            txt_surface = font.render(str(signals[i].signalText), True, color, (0, 0, 0))
            screen.blit(txt_surface, signalTimerCoods[i])

        # Move and render vehicles
        for vehicle in simulation:
            vehicle.move()
            vehicle.render(screen)

        # Update and render effects
        for effect in list(ring_effects):
            effect.update()
            effect.draw(screen)
            if effect.dead:
                ring_effects.remove(effect)

        # Draw UI overlay
        ui_bg = pygame.Surface((300, 100), pygame.SRCALPHA)
        ui_bg.fill((0, 0, 0, 180))
        screen.blit(ui_bg, (10, 10))
        
        info1 = font.render("Press 'A' to inject Ambulance", True, (255, 255, 255))
        screen.blit(info1, (20, 20))
        
        status_color = (255, 100, 50) if priority_mode else ((100, 255, 100) if rl_model else (200, 200, 200))
        if priority_mode:
            status_text = f"PRIORITY OVERRIDE ({len(active_ambulances)} ACTIVE)"
        else:
            status_text = "RL MODEL CONTROL" if rl_model else "FIXED TIMERS"
        info2 = font.render(f"Status: {status_text}", True, status_color)
        screen.blit(info2, (20, 60))

        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()