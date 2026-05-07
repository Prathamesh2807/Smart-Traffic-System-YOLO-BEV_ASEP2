"""
=============================================================
  AI Smart Traffic Signal Chowk Simulation – Configuration
=============================================================
  Edit this file to change simulation parameters easily.
"""

# ─────────────────────────────────────────────────────────
#  WINDOW / DISPLAY
# ─────────────────────────────────────────────────────────
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FPS           = 60
TITLE         = "AI Smart Traffic Signal – Chowk Simulation"

# ─────────────────────────────────────────────────────────
#  ROAD GEOMETRY  (pixels)
# ─────────────────────────────────────────────────────────
ROAD_WIDTH    = 150
LANE_WIDTH    = ROAD_WIDTH // 2   # incoming + outgoing lane each

CENTER_X = WINDOW_WIDTH  // 2
CENTER_Y = WINDOW_HEIGHT // 2

STOP_LINE_DIST = ROAD_WIDTH // 2 + 12   # px from centre to stop line

# ─────────────────────────────────────────────────────────
#  VEHICLE TYPES
#  (width, height, speed px/sec, signal weight, color RGB)
# ─────────────────────────────────────────────────────────
VEHICLE_TYPES = {
    "bike": {
        "width": 14, "height": 28,
        "speed": 125,
        "weight": 1,
        "color": (140, 190, 255),
        "label": "Bike",
    },
    "car": {
        "width": 22, "height": 40,
        "speed": 95,
        "weight": 2,
        "color": (70, 185, 100),
        "label": "Car",
    },
    "bus": {
        "width": 28, "height": 62,
        "speed": 62,
        "weight": 3,
        "color": (240, 165, 45),
        "label": "Bus",
    },
    "truck": {
        "width": 30, "height": 74,
        "speed": 52,
        "weight": 4,
        "color": (200, 85, 65),
        "label": "Truck",
    },
    "ambulance": {
        "width": 22, "height": 48,
        "speed": 155,
        "weight": 0,           # bypasses normal timing
        "color": (245, 245, 252),
        "label": "Ambulance",
    },
}

# ─────────────────────────────────────────────────────────
#  SPAWNING
# ─────────────────────────────────────────────────────────
SPAWN_INTERVAL   = 2.0        # seconds between spawn attempts per side
SPAWN_CHANCE     = 0.88       # probability a vehicle spawns in each interval
AMBULANCE_CHANCE = 0.05       # probability that a spawned vehicle is ambulance
MIN_SPAWN_GAP    = 32         # minimum px gap between consecutive vehicles at spawn

VEHICLE_WEIGHTS = {           # proportions for non-ambulance spawn
    "bike":  0.30,
    "car":   0.44,
    "bus":   0.15,
    "truck": 0.11,
}

# ─────────────────────────────────────────────────────────
#  SIGNAL TIMING
# ─────────────────────────────────────────────────────────
MIN_GREEN_TIME       = 5
MAX_GREEN_TIME       = 30          # cap (as specified)
YELLOW_TIME          = 3
MIN_RED_WAIT         = 4

AMBULANCE_DETECT_DIST  = 380       # px from stop line at which ambulance is detected
AMBULANCE_GREEN_HOLD   = 13        # seconds green is held per ambulance pass
AMBULANCE_EXTEND_HOLD  = 4         # extra seconds if more ambulances on same side

# ─────────────────────────────────────────────────────────
#  PHYSICS
# ─────────────────────────────────────────────────────────
SAFE_FOLLOW_GAP = 12    # minimum px gap between bumpers (was 6, too small)
DECEL_DISTANCE  = 120   # start braking earlier (was 90)

# ─────────────────────────────────────────────────────────
#  COLOURS
# ─────────────────────────────────────────────────────────
COLOR_BG            = (28,  33,  44)
COLOR_ROAD          = (52,  58,  68)
COLOR_LANE_MARK     = (195, 195, 155)
COLOR_CURB          = (75,  80,  92)
COLOR_INTERSECTION  = (58,  63,  74)
COLOR_BUILDING      = [
    (68,  52,  82),
    (48,  68,  88),
    (84,  58,  52),
    (52,  78,  68),
    (74,  62,  48),
    (58,  74,  84),
    (82,  72,  58),
    (48,  82,  72),
]
COLOR_WINDOW_LIGHT  = (255, 240, 175)
COLOR_SIGNAL_RED    = (225,  48,  48)
COLOR_SIGNAL_YELLOW = (225, 200,  28)
COLOR_SIGNAL_GREEN  = ( 48, 215,  78)
COLOR_SIGNAL_OFF    = ( 38,  38,  40)
COLOR_POLE          = (115, 118, 130)

# Dashboard
COLOR_DASH_BG       = (16,  20,  30)
COLOR_DASH_BORDER   = (55,  75, 115)
COLOR_TEXT_BRIGHT   = (218, 228, 255)
COLOR_TEXT_DIM      = (125, 140, 170)
COLOR_HIGHLIGHT     = ( 75, 155, 255)
COLOR_EMERGENCY     = (255,  65,  65)
COLOR_SUCCESS       = ( 55, 210,  95)
COLOR_AMBU_BLUE     = ( 50, 120, 255)
COLOR_AMBU_RED      = (255,  40,  40)

# ─────────────────────────────────────────────────────────
#  DIRECTION HELPERS
# ─────────────────────────────────────────────────────────
DIRECTIONS = ["north", "south", "east", "west"]
OPPOSITE   = {"north": "south", "south": "north", "east": "west", "west": "east"}

# ─────────────────────────────────────────────────────────
#  CLOUD DASHBOARD  &  AI PREDICTION
# ─────────────────────────────────────────────────────────
DASHBOARD_ENABLED  = True
DASHBOARD_PORT     = 5000
DASHBOARD_HOST     = "0.0.0.0"   # accessible from WiFi

PREDICT_HORIZON    = [30, 60, 90]  # seconds ahead to forecast
HISTORY_WINDOW     = 120           # samples kept (60 s at 0.5 s interval)

# ─────────────────────────────────────────────────────────
#  ACCIDENT SYSTEM
# ─────────────────────────────────────────────────────────
ACCIDENT_STUCK_SPEED     = 2.0     # px/s threshold — below this = "stuck"
ACCIDENT_STUCK_TIME      = 4.0     # seconds a vehicle must be stuck to auto-trigger
ACCIDENT_DISPATCH_MIN    = 5.0     # min delay (seconds) before ambulance auto-dispatched
ACCIDENT_DISPATCH_MAX    = 8.0     # max delay (seconds) before ambulance auto-dispatched
ACCIDENT_SMOKE_PARTICLES = 4       # number of smoke circles drawn per frame
