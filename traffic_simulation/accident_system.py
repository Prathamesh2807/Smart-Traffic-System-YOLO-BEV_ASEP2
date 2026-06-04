"""
=============================================================
  accident_system.py  -  Accident Detection, Simulation
                         & Response System
=============================================================
  Modular add-on - does NOT touch TrafficController, Predictor,
  or core vehicle movement logic.

  Features:
    - Manual trigger (press C) / reset (press R)
    - Flashing crash visuals + smoke (drawn by renderer.py)
    - Emergency ambulance dispatch after random delay
    - Active lane clearing: vehicles drive to intersection
      corner then arc 90deg onto the free diversion lane
    - Dashboard state reporting

  Safety rules:
    - Only one accident active at a time
    - Ambulance dispatched via normal Vehicle() so existing
      priority system handles it automatically
    - Signal overrides are display-only
=============================================================
"""

import math
import time
import random
from config import (
    CENTER_X, CENTER_Y, ROAD_WIDTH, LANE_WIDTH, DIRECTIONS, OPPOSITE,
    WINDOW_WIDTH, WINDOW_HEIGHT,
    ACCIDENT_STUCK_SPEED, ACCIDENT_STUCK_TIME,
    ACCIDENT_DISPATCH_MIN, ACCIDENT_DISPATCH_MAX,
)


class AccidentSystem:
    """Central accident state manager."""

    def __init__(self):
        # -- Accident state ------------------------------------------
        self.accident_active:   bool  = False
        self.accident_vehicle          = None
        self.accident_time:     float = 0.0
        self.accident_duration: float = 0.0

        # -- Auto-detection: per-vehicle stuck timers ----------------
        self._stuck_tracker: dict = {}

        # -- Emergency response --------------------------------------
        self._ambulance_dispatched:     bool  = False
        self._ambulance_dispatch_timer: float = 0.0
        self._dispatch_delay:           float = 0.0
        self._dispatch_direction:       object = None

        # -- Lane clearing (active traffic diversion) ----------------
        self.blocked_direction:   object = None
        self.diversion_direction: object = None
        self._clearing_active:   bool  = False
        self._clearing_timer:    float = 0.0
        self._vehicles_cleared:  int   = 0
        self._clear_interval:    float = 0.8

        # -- Logging callback (set by main.py) -----------------------
        self._log_fn = None

    # -- Public: set log callback ------------------------------------

    def set_logger(self, fn):
        self._log_fn = fn

    def _log(self, msg: str):
        if self._log_fn:
            self._log_fn(msg)
        print(f"[ACCIDENT] {msg}")

    # -- Manual trigger ----------------------------------------------

    def trigger_accident(self, vehicles: list):
        if self.accident_active:
            print("[ACCIDENT] Already active - ignored.")
            return

        candidates = [
            v for v in vehicles
            if (not v.passed
                and not v.is_off_screen()
                and v.vtype != "ambulance"
                and not getattr(v, "is_crashed", False))
        ]
        if not candidates:
            print("[ACCIDENT] No eligible vehicles to crash.")
            return

        candidates.sort(key=lambda v: math.hypot(v.x - CENTER_X, v.y - CENTER_Y))
        top_n = max(1, len(candidates) // 4)
        victim = random.choice(candidates[:top_n])
        self._activate(victim)

    # -- Manual reset ------------------------------------------------

    def reset_accident(self):
        if not self.accident_active:
            print("[ACCIDENT] No active accident to reset.")
            return

        if self.accident_vehicle is not None:
            self.accident_vehicle.is_crashed = False

        self._log("Accident cleared - traffic resuming")
        self.accident_active           = False
        self.accident_vehicle          = None
        self.accident_time             = 0.0
        self.accident_duration         = 0.0
        self._ambulance_dispatched     = False
        self._ambulance_dispatch_timer = 0.0
        self._dispatch_direction       = None
        self.blocked_direction         = None
        self.diversion_direction       = None
        self._clearing_active          = False
        self._clearing_timer           = 0.0
        self._vehicles_cleared         = 0
        self._stuck_tracker.clear()

    # -- Per-frame update --------------------------------------------

    def update(self, dt: float, vehicles: list):
        """
        Called every frame from Simulation._update().
        Returns a newly spawned ambulance Vehicle or None.
        """
        spawned = None

        if self.accident_active:
            self.accident_duration += dt

            if not self._ambulance_dispatched:
                self._ambulance_dispatch_timer += dt
                if self._ambulance_dispatch_timer >= self._dispatch_delay:
                    spawned = self._dispatch_ambulance()

            if self._clearing_active:
                self._clear_lane_traffic(dt, vehicles)

        # Auto-detection DISABLED - accidents are user-controlled (press C)

        active_ids = {v.id for v in vehicles}
        self._stuck_tracker = {
            vid: t for vid, t in self._stuck_tracker.items()
            if vid in active_ids
        }

        return spawned

    # -- State for dashboard -----------------------------------------

    def get_state(self) -> dict:
        return {
            "active":               self.accident_active,
            "vehicle_id":           self.accident_vehicle.id if self.accident_vehicle else None,
            "direction":            self.accident_vehicle.direction if self.accident_vehicle else None,
            "duration":             round(self.accident_duration, 1),
            "ambulance_dispatched": self._ambulance_dispatched,
            "congestion_warning":   self.accident_active,
            "blocked_direction":    self.blocked_direction,
            "diversion_direction":  self.diversion_direction,
            "clearing_active":      self._clearing_active,
            "vehicles_cleared":     self._vehicles_cleared,
        }

    # -- Internal helpers --------------------------------------------

    def _activate(self, vehicle):
        vehicle.is_crashed     = True
        self.accident_active   = True
        self.accident_vehicle  = vehicle
        self.accident_time     = time.monotonic()
        self.accident_duration = 0.0

        self._ambulance_dispatched     = False
        self._ambulance_dispatch_timer = 0.0
        self._dispatch_delay = random.uniform(ACCIDENT_DISPATCH_MIN, ACCIDENT_DISPATCH_MAX)
        self._dispatch_direction = self._nearest_direction(vehicle)

        self._log(f"ACCIDENT at intersection - {vehicle.direction.upper()} lane, {vehicle.vtype}")

        self.blocked_direction   = vehicle.direction
        self.diversion_direction = self._pick_diversion_direction(vehicle.direction)
        self._log(f"{vehicle.direction.upper()} lane CLOSED - diverting traffic to {self.diversion_direction.upper()}")

        self._clearing_active  = True
        self._clearing_timer   = 0.0
        self._vehicles_cleared = 0

    def _dispatch_ambulance(self):
        from vehicles import Vehicle
        direction = self._dispatch_direction or random.choice(DIRECTIONS)
        ambu = Vehicle("ambulance", direction)
        self._ambulance_dispatched = True
        self._log(f"Emergency services dispatched - {direction.upper()} side")
        return ambu

    def _nearest_direction(self, vehicle) -> str:
        vx, vy = vehicle.x, vehicle.y
        dists = {
            "north": vy,
            "south": WINDOW_HEIGHT - vy,
            "east":  WINDOW_WIDTH - vx,
            "west":  vx,
        }
        blocked = vehicle.direction
        options = {d: dist for d, dist in dists.items() if d != blocked}
        return min(options, key=options.get)

    def _pick_diversion_direction(self, blocked: str) -> str:
        opposite = OPPOSITE[blocked]
        adjacent = [d for d in DIRECTIONS if d != blocked and d != opposite]
        return random.choice(adjacent) if adjacent else opposite

    def _clear_lane_traffic(self, dt: float, vehicles: list):
        """
        Gradually divert vehicles out of the blocked lane.
        Each vehicle is given a 3-phase waypoint path:
          Phase 0 - drive forward to the road corner
          Phase 1 - arc 90deg through intersection
          Phase 2 - exit straight on diversion lane
        """
        if not self.blocked_direction:
            return

        self._clearing_timer += dt
        if self._clearing_timer < self._clear_interval:
            return
        self._clearing_timer = 0.0

        lane_vehicles = [
            v for v in vehicles
            if (v.direction == self.blocked_direction
                and not v.passed
                and not v.is_off_screen()
                and not getattr(v, "is_crashed", False)
                and not getattr(v, "is_diverted", False)
                and v.vtype != "ambulance")
        ]

        if not lane_vehicles:
            if self._clearing_active:
                self._clearing_active = False
                div_dir = (self.diversion_direction or "?").upper()
                self._log(f"{self.blocked_direction.upper()} lane cleared - {self._vehicles_cleared} vehicles diverted to {div_dir}")
            return

        # Pick from back of queue (farthest from intersection)
        lane_vehicles.sort(key=lambda v: v.dist_to_stop(), reverse=True)
        victim = lane_vehicles[0]

        # Setup 3-phase waypoint path
        _setup_diversion_path(victim, self.blocked_direction, self.diversion_direction)

        victim.is_diverted   = True
        victim._divert_phase = 0
        victim._divert_arc_t = 0.0
        victim.passed        = True
        victim._counted      = True

        self._vehicles_cleared += 1


# ================================================================
#  Diversion path geometry (module-level, no circular imports)
# ================================================================

def _setup_diversion_path(v, blocked: str, divert_to: str):
    """
    Compute the 3-phase turn path for a diverted vehicle.

    Phase 0 - APPROACH : drive straight to the road-edge corner
              where the turn starts (inside the intersection area).
    Phase 1 - ARC      : follow a 90-degree circular arc through
              the intersection corner (radius = LANE_WIDTH//2).
    Phase 2 - EXIT     : drive straight out on the diversion lane.

    Coordinate system: screen coords (Y increases downward).
    Angles stored in math convention (0=right, pi/2=UP in maths,
    but since screen Y is inverted, pi/2 visually points UP too
    when we use  x += cos(a)*spd,  y -= sin(a)*spd  in update()).
    """
    R   = ROAD_WIDTH // 2    # 75 px  - half road width
    LW  = LANE_WIDTH         # 75 px
    CX, CY = CENTER_X, CENTER_Y
    pi  = math.pi
    r   = LW // 2            # 37 px arc radius

    # Incoming lane centre (the lane the blocked vehicle is in)
    #   north/south: x-coordinate of that lane
    #   east/west  : y-coordinate of that lane
    inc_x = {
        "north": CX - LW // 2,   # north incoming = left half
        "south": CX + LW // 2,   # south incoming = right half
    }
    inc_y = {
        "west": CY - LW // 2,    # west incoming = top half
        "east": CY + LW // 2,    # east incoming = bottom half
    }

    # ----------------------------------------------------------------
    # Geometry table: keyed by (blocked, divert_to)
    #
    # wp    = (x, y)  approach waypoint  -  the exact road-edge corner
    #                 where the vehicle should be when it starts arcing.
    # arc_c = (x, y)  arc circle centre
    # a_s   = arc start angle  (radians, screen-Y-inverted math convention)
    # a_e   = arc end angle
    # exit  = (dx, dy)  unit direction for the straight exit phase
    # ----------------------------------------------------------------

    lxN = inc_x["north"]   # 602
    lxS = inc_x["south"]   # 677
    lyW = inc_y["west"]    # 322
    lyE = inc_y["east"]    # 397

    table = {
        # ---- North blocked (vehicle travelling DOWNWARD into junction) ----
        # To EAST: vehicle in left lane, turns right -> exits eastward
        ("north", "east"): dict(
            wp=(lxN, CY - R + r),
            arc_c=(lxN + r, CY - R + r),
            a_s=pi,   a_e=pi / 2,
            exit=(1, 0)),
        # To WEST: vehicle in left lane, turns left -> exits westward
        ("north", "west"): dict(
            wp=(lxN, CY - R + r),
            arc_c=(lxN - r, CY - R + r),
            a_s=0,    a_e=pi / 2,
            exit=(-1, 0)),

        # ---- South blocked (vehicle travelling UPWARD into junction) ----
        # To EAST: turns left -> exits eastward
        ("south", "east"): dict(
            wp=(lxS, CY + R - r),
            arc_c=(lxS + r, CY + R - r),
            a_s=pi,   a_e=-pi / 2,
            exit=(1, 0)),
        # To WEST: turns right -> exits westward
        ("south", "west"): dict(
            wp=(lxS, CY + R - r),
            arc_c=(lxS - r, CY + R - r),
            a_s=0,    a_e=-pi / 2,
            exit=(-1, 0)),

        # ---- West blocked (vehicle travelling RIGHTWARD into junction) ----
        # To NORTH: turns right -> exits northward (upward on screen)
        ("west", "north"): dict(
            wp=(CX - R + r, lyW),
            arc_c=(CX - R + r, lyW - r),
            a_s=-pi / 2, a_e=0,
            exit=(0, -1)),
        # To SOUTH: turns left -> exits southward (downward on screen)
        ("west", "south"): dict(
            wp=(CX - R + r, lyW),
            arc_c=(CX - R + r, lyW + r),
            a_s=pi / 2,  a_e=0,
            exit=(0, 1)),

        # ---- East blocked (vehicle travelling LEFTWARD into junction) ----
        # To NORTH: turns left -> exits northward
        ("east", "north"): dict(
            wp=(CX + R - r, lyE),
            arc_c=(CX + R - r, lyE - r),
            a_s=-pi / 2, a_e=pi,
            exit=(0, -1)),
        # To SOUTH: turns right -> exits southward
        ("east", "south"): dict(
            wp=(CX + R - r, lyE),
            arc_c=(CX + R - r, lyE + r),
            a_s=pi / 2,  a_e=pi,
            exit=(0, 1)),
    }

    key = (blocked, divert_to)
    if key not in table:
        # Fallback: drive straight toward divert direction
        fd = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
        edx, edy = fd.get(divert_to, (1, 0))
        v._divert_wp          = [(v.x, v.y)]
        v._divert_arc_center  = (v.x, v.y)
        v._divert_arc_r       = 1.0
        v._divert_arc_a_start = 0.0
        v._divert_arc_a_end   = 0.0
        v._divert_arc_span    = 0.0
        v._divert_exit_dx     = float(edx)
        v._divert_exit_dy     = float(edy)
        v._divert_angle       = math.atan2(-edy, edx)
        return

    g = table[key]

    # Approach waypoint
    v._divert_wp = [g["wp"]]

    # Arc geometry
    cx_a, cy_a = g["arc_c"]
    a_start = g["a_s"]
    a_end   = g["a_e"]
    # Compute signed span (shortest arc)
    span = a_end - a_start
    while span >  pi: span -= 2 * pi
    while span < -pi: span += 2 * pi

    v._divert_arc_center  = (cx_a, cy_a)
    v._divert_arc_r       = float(r)
    v._divert_arc_a_start = a_start
    v._divert_arc_a_end   = a_end
    v._divert_arc_span    = span

    # Exit straight direction
    edx, edy = g["exit"]
    v._divert_exit_dx = float(edx)
    v._divert_exit_dy = float(edy)

    # Initial heading angle matches vehicle's original travel direction
    approach_angle = {
        "north": -pi / 2,   # heading downward on screen
        "south":  pi / 2,   # heading upward on screen
        "east":   pi,       # heading leftward
        "west":   0.0,      # heading rightward
    }[blocked]
    v._divert_angle = approach_angle
