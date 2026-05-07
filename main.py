"""
=============================================================
  main.py  –  Entry point  (run this file to start)
=============================================================

  Required packages:
      pip install pygame numpy

  How to run:
      cd "C:\\Users\\nachi\\OneDrive\\Desktop\\ASEP 2\\traffic_simulation"
      python main.py

  Settings to change:
      config.py  →  MAX_GREEN_TIME, SPAWN_INTERVAL, SPAWN_CHANCE,
                    AMBULANCE_CHANCE, AMBULANCE_DETECT_DIST, VEHICLE_TYPES
=============================================================
"""

import sys
import time
import random
import math
import pygame
import numpy as np

from config   import *
from vehicles import Vehicle, try_spawn
from signals  import TrafficController, SignalState
from renderer import (
    draw_scene, draw_signals, draw_dashboard,
    draw_fps, draw_compass, draw_emergency_banner,
    draw_accident_effects,
)
from predictor        import TrafficPredictor
from dashboard_server import start_dashboard, update_stats, log_ambulance_event
from accident_system  import AccidentSystem


# ─────────────────────────────────────────────────────────
#  Font initialisation
# ─────────────────────────────────────────────────────────

def init_fonts() -> dict:
    pygame.font.init()
    def sf(size, bold=False):
        for name in ("Segoe UI", "Arial", "DejaVu Sans", "Sans"):
            try:
                return pygame.font.SysFont(name, size, bold=bold)
            except Exception:
                pass
        return pygame.font.Font(None, size)
    return {
        "banner": sf(22, bold=True),
        "big":    sf(26),
        "medium": sf(19),
        "small":  sf(16),
        "tiny":   sf(13),
    }


# ─────────────────────────────────────────────────────────
#  Lane helper
# ─────────────────────────────────────────────────────────

def _vehicles_ahead(v: Vehicle, all_vehicles: list) -> list:
    """Return vehicles in the same lane that are strictly ahead of v, nearest-first."""
    v_dist = v.dist_to_stop()
    candidates = [
        u for u in all_vehicles
        if (u is not v
            and u.direction == v.direction
            and not u.passed
            and not getattr(u, 'is_diverted', False)
            and u.dist_to_stop() < v_dist)
    ]
    candidates.sort(key=lambda u: u.dist_to_stop())
    return candidates


# ─────────────────────────────────────────────────────────
#  Main simulation class
# ─────────────────────────────────────────────────────────

class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock  = pygame.time.Clock()
        self.fonts  = init_fonts()

        # Static background rendered once
        self._bg = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        draw_scene(self._bg, self.fonts)

        # Simulation state
        self.vehicles:   list[Vehicle]     = []
        self.controller: TrafficController = TrafficController()
        self._spawn_timers: dict[str, float] = {d: 0.0 for d in DIRECTIONS}

        self.total_passed: int   = 0
        self._prev_time:   float = time.monotonic()
        self._fps_avg:     float = float(FPS)

        # AI Predictor
        self.predictor        = TrafficPredictor()
        self._stats_timer:  float = 0.0
        self._emergency_count: int = 0
        self._prev_emergency: bool = False

        # Accident Detection & Response System
        self.accident_system = AccidentSystem()
        self.accident_system.set_logger(log_ambulance_event)

    # ── Queue helper ─────────────────────────────────────

    def _queues(self) -> dict:
        q: dict[str, list] = {d: [] for d in DIRECTIONS}
        for v in self.vehicles:
            if not v.passed:
                q[v.direction].append(v)
        return q

    # ── Spawning ─────────────────────────────────────────

    def _update_spawn(self, dt: float):
        for d in DIRECTIONS:
            # Block spawning in the accident lane (lane is closed)
            if self.accident_system.blocked_direction == d:
                continue

            self._spawn_timers[d] += dt
            if self._spawn_timers[d] >= SPAWN_INTERVAL:
                self._spawn_timers[d] = 0.0
                if random.random() < SPAWN_CHANCE:
                    v = try_spawn(d, self.vehicles)
                    if v is not None:
                        self.vehicles.append(v)

    # ── Update ───────────────────────────────────────────

    def _update(self, dt: float):
        queues = self._queues()

        # 1. AI signal controller
        self.controller.update(dt, queues)

        # 2. Vehicle physics
        for v in self.vehicles:
            green = self.controller.is_green(v.direction) or v.in_junction
            ahead = _vehicles_ahead(v, self.vehicles)
            v.update(dt, green, ahead)

        # 3. Count passed vehicles
        for v in self.vehicles:
            if v.passed and not hasattr(v, "_counted"):
                v._counted           = True
                self.total_passed   += 1
                self.controller.total_passed = self.total_passed

        # 4. Remove off-screen vehicles
        self.vehicles = [v for v in self.vehicles if not v.is_off_screen()]

        # 5. Spawn new vehicles
        self._update_spawn(dt)

        # 5b. Accident system (auto-detect, emergency dispatch)
        spawned_ambu = self.accident_system.update(dt, self.vehicles)
        if spawned_ambu is not None:
            self.vehicles.append(spawned_ambu)

        # 6. AI Predictor update
        self.predictor.update(dt, queues, self.total_passed)

        # 7. Detect ambulance emergency events
        if self.controller.emergency_active and not self._prev_emergency:
            self._emergency_count += 1
            d = self.controller.emergency_dir or "?"
            log_ambulance_event(f"🚨 EMERGENCY activated – {d.upper()} side")
        elif not self.controller.emergency_active and self._prev_emergency:
            log_ambulance_event("✅ Emergency cleared – normal cycle resumed")
        self._prev_emergency = self.controller.emergency_active

        # 8. Push stats to dashboard every 0.5 s
        self._stats_timer += dt
        if self._stats_timer >= 0.5:
            self._stats_timer = 0.0
            self._push_stats(queues)

    # ── Dashboard stats push ──────────────────────────────

    def _push_stats(self, queues: dict):
        pred = self.predictor.get_stats()
        stats = {
            "mode":            self.controller.current_mode,
            "green_dir":       self.controller.green_dir,
            "phase":           self.controller.phase,
            "countdown":       round(self.controller.countdown(), 1),
            "signals":         {d: self.controller.signal_color(d) for d in DIRECTIONS},
            "queues":          {d: sum(1 for v in queues.get(d, []) if not v.passed)
                                for d in DIRECTIONS},
            "total_passed":    self.total_passed,
            "fps":             round(self._fps_avg, 1),
            "emergency_active": self.controller.emergency_active,
            "emergency_dir":   self.controller.emergency_dir,
            "ambulance_queue": self.controller.pending_ambulance_dirs,
            "emergency_count": self._emergency_count,
            **pred,
            "accident": self.accident_system.get_state(),
        }
        update_stats(stats)

    # ── Draw ─────────────────────────────────────────────

    def _draw(self):
        # Static scene
        self.screen.blit(self._bg, (0, 0))

        # Live signal lights
        draw_signals(self.screen, self.controller)

        # Vehicles sorted by correct depth axis to prevent overlap.
        # Vertical movers (N/S) sort by Y; horizontal (E/W) sort by X.
        # Ambulances always drawn on top within their layer.
        def _depth_key(u):
            if u.direction in ("north", "south"):
                return (0, u.y, 1 if u.vtype == "ambulance" else 0)
            else:
                return (1, u.x, 1 if u.vtype == "ambulance" else 0)

        for v in sorted(self.vehicles, key=_depth_key):
            v.draw(self.screen)

        # HUD
        draw_dashboard(self.screen, self.fonts,
                       self.controller, self._queues(), self.total_passed)
        draw_fps(self.screen, self.fonts, self._fps_avg)
        draw_compass(self.screen, self.fonts)

        # Ambulance banner (with multi-queue support)
        draw_emergency_banner(self.screen, self.fonts, self.controller)

        # Accident overlay effects
        draw_accident_effects(self.screen, self.fonts,
                              self.vehicles, self.accident_system)

        pygame.display.flip()

    # ── Main loop ────────────────────────────────────────

    def run(self):
        running = True
        while running:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_a:
                        # Spawn one ambulance on a random side for demo
                        d    = random.choice(DIRECTIONS)
                        ambu = Vehicle("ambulance", d)
                        self.vehicles.append(ambu)
                        print(f"[DEMO] Ambulance spawned on: {d.upper()}")
                    elif event.key == pygame.K_m:
                        # Spawn ambulances on ALL four sides simultaneously
                        for d in DIRECTIONS:
                            ambu = Vehicle("ambulance", d)
                            self.vehicles.append(ambu)
                        print("[DEMO] Ambulances spawned on ALL four sides!")
                    elif event.key == pygame.K_c:
                        # Trigger a manual accident
                        self.accident_system.trigger_accident(self.vehicles)
                    elif event.key == pygame.K_r:
                        # Reset / clear the active accident
                        self.accident_system.reset_accident()

            # Timing
            now  = time.monotonic()
            dt   = min(now - self._prev_time, 0.033)  # cap at 33 ms (~30 fps min)
            self._prev_time = now
            self._fps_avg   = 0.9 * self._fps_avg + 0.1 * (1.0 / max(dt, 1e-6))

            # Simulate + render
            self._update(dt)
            self._draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit(0)


# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  AI Smart Traffic Signal - Chowk Simulation")
    print("  ESC / Q  -> Quit")
    print("  A        -> Spawn 1 ambulance on a random side")
    print("  M        -> Spawn ambulances on ALL 4 sides (stress test)")
    print("  C        -> Trigger accident (manual)")
    print("  R        -> Reset / clear accident")
    print("=" * 62)
    if DASHBOARD_ENABLED:
        start_dashboard(port=DASHBOARD_PORT, host=DASHBOARD_HOST)
    Simulation().run()
