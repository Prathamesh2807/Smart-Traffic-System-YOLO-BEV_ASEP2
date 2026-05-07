"""
=============================================================
  renderer.py  –  All drawing / rendering functions
=============================================================
"""
import math
import random
import time
import pygame
import numpy as np
from config import *

# ─────────────────────────────────────────────────────────
#  One-time glow surface cache (avoids per-frame allocation)
# ─────────────────────────────────────────────────────────
_GLOW_CACHE: dict[tuple, pygame.Surface] = {}

def _get_glow(color: tuple, size: int = 32, alpha: int = 55) -> pygame.Surface:
    key = (color, size, alpha)
    if key not in _GLOW_CACHE:
        s = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (size // 2, size // 2), size // 2)
        _GLOW_CACHE[key] = s
    return _GLOW_CACHE[key]


# ─────────────────────────────────────────────────────────
#  Building layout (generated once, deterministic seed)
# ─────────────────────────────────────────────────────────
_BUILDINGS: list[dict] = []


def _generate_buildings():
    global _BUILDINGS
    if _BUILDINGS:
        return
    rng       = random.Random(42)
    half_road = ROAD_WIDTH // 2
    margin    = 8

    corners = [
        (margin, margin, CENTER_X - half_road - margin, CENTER_Y - half_road - margin),
        (CENTER_X + half_road + margin, margin, WINDOW_WIDTH - margin, CENTER_Y - half_road - margin),
        (margin, CENTER_Y + half_road + margin, CENTER_X - half_road - margin, WINDOW_HEIGHT - margin),
        (CENTER_X + half_road + margin, CENTER_Y + half_road + margin, WINDOW_WIDTH - margin, WINDOW_HEIGHT - margin),
    ]

    STYLES = ["glass", "classic", "modern", "tower"]

    for (zx1, zy1, zx2, zy2) in corners:
        zone_w, zone_h = zx2 - zx1, zy2 - zy1
        if zone_w <= 0 or zone_h <= 0:
            continue
        rows = rng.randint(2, 3)
        cols = rng.randint(2, 4)
        bw   = max(26, zone_w // cols - 8)
        bh   = max(32, zone_h // rows - 8)
        for r in range(rows):
            for c in range(cols):
                x = zx1 + c * (zone_w // cols) + rng.randint(2, 6)
                y = zy1 + r * (zone_h // rows) + rng.randint(2, 6)
                w = rng.randint(max(20, bw - 10), bw + 8)
                h = rng.randint(max(28, bh - 10), bh + 14)
                w = min(w, zx2 - x - 5)
                h = min(h, zy2 - y - 5)
                if w < 16 or h < 18:
                    continue
                ci    = rng.randrange(len(COLOR_BUILDING))
                style = rng.choice(STYLES)
                win_rows = max(2, h // 13)
                win_cols = max(1, w // 12)
                _BUILDINGS.append({
                    "rect":      pygame.Rect(x, y, w, h),
                    "color":     COLOR_BUILDING[ci],
                    "win_rows":  win_rows,
                    "win_cols":  win_cols,
                    "style":     style,
                    "antenna":   rng.random() < 0.35,
                    "watertower": rng.random() < 0.15,
                    "seed":      rng.randint(0, 9999),
                })


# ─────────────────────────────────────────────────────────
#  Road helpers
# ─────────────────────────────────────────────────────────

def _road_rects():
    half   = ROAD_WIDTH // 2
    v_road = pygame.Rect(CENTER_X - half, 0,  ROAD_WIDTH, WINDOW_HEIGHT)
    h_road = pygame.Rect(0, CENTER_Y - half,  WINDOW_WIDTH, ROAD_WIDTH)
    return [v_road, h_road]


# Pole positions – near-right corner of each approach
_POLE_POSITIONS = {
    "north": (CENTER_X + ROAD_WIDTH // 2 + 16, CENTER_Y - STOP_LINE_DIST - 10),
    "south": (CENTER_X - ROAD_WIDTH // 2 - 16, CENTER_Y + STOP_LINE_DIST + 10),
    "east":  (CENTER_X + STOP_LINE_DIST + 10,  CENTER_Y + ROAD_WIDTH // 2 + 16),
    "west":  (CENTER_X - STOP_LINE_DIST - 10,  CENTER_Y - ROAD_WIDTH // 2 - 16),
}


# ─────────────────────────────────────────────────────────
#  Main static scene draw
# ─────────────────────────────────────────────────────────

def draw_scene(surface: pygame.Surface, fonts: dict):
    surface.fill(COLOR_BG)
    _generate_buildings()
    _draw_buildings(surface)

    # Road strips
    for rect in _road_rects():
        pygame.draw.rect(surface, COLOR_ROAD, rect)

    # Intersection box
    half      = ROAD_WIDTH // 2
    inter_rect = pygame.Rect(CENTER_X - half, CENTER_Y - half, ROAD_WIDTH, ROAD_WIDTH)
    pygame.draw.rect(surface, COLOR_INTERSECTION, inter_rect)

    _draw_curbs(surface)
    _draw_lane_markings(surface)
    _draw_stop_lines(surface)
    _draw_zebra(surface)

    # Signal pole structures (lights are drawn live in draw_signals)
    for direction, (px, py) in _POLE_POSITIONS.items():
        pygame.draw.line(surface, COLOR_POLE, (px, py), (px, py - 62), 4)
        pygame.draw.line(surface, COLOR_POLE, (px - 9, py - 62), (px + 9, py - 62), 4)
        bw2, bh2 = 18, 48
        box_rect = pygame.Rect(px - bw2 // 2, py - 62 - bh2, bw2, bh2)
        pygame.draw.rect(surface, (28, 28, 34), box_rect, border_radius=4)
        pygame.draw.rect(surface, (68, 68, 80), box_rect, 1, border_radius=4)


def _draw_buildings(surface: pygame.Surface):
    rng_win = random.Random(0)  # deterministic window lighting
    for b in _BUILDINGS:
        rect  = b["rect"]
        color = b["color"]
        style = b.get("style", "classic")
        seed  = b.get("seed", 0)
        rng_b = random.Random(seed)

        # ── Deep shadow ──────────────────────────────────
        shadow = rect.move(6, 6)
        pygame.draw.rect(surface, tuple(max(0, c-45) for c in color), shadow, border_radius=5)

        # ── Building body ────────────────────────────────
        pygame.draw.rect(surface, color, rect, border_radius=5)

        if style == "glass":
            # Tinted glass curtain wall — vertical bands
            band_w = max(4, rect.width // 5)
            for bx in range(rect.left + 2, rect.right - 2, band_w):
                bnd_w = min(band_w - 2, rect.right - bx - 2)
                if bnd_w < 2: continue
                glass_col = tuple(min(255, c + 18) for c in color)
                pygame.draw.rect(surface, glass_col,
                    (bx, rect.top + 2, bnd_w, rect.height - 4), border_radius=2)
            # Glass shine
            pygame.draw.rect(surface, tuple(min(255, c+60) for c in color),
                (rect.left+2, rect.top+2, max(3, rect.width//6), rect.height-4), border_radius=2)

        elif style == "tower":
            # Setback tower: darker lower base, lighter top section
            base_h = rect.height * 2 // 3
            pygame.draw.rect(surface, tuple(max(0, c-15) for c in color),
                (rect.left, rect.top+base_h, rect.width, rect.height-base_h), border_radius=3)
            pygame.draw.rect(surface, tuple(min(255, c+28) for c in color),
                (rect.left+2, rect.top, rect.width-4, base_h), border_radius=5)
            # Vertical pillar lines
            for px in range(rect.left+4, rect.right-4, max(5, rect.width//4)):
                pygame.draw.line(surface, tuple(max(0,c-20) for c in color),
                    (px, rect.top+3), (px, rect.top+base_h-2), 1)

        elif style == "modern":
            # Horizontal floor bands
            band_h = max(5, rect.height // max(3, rect.height//10))
            for by in range(rect.top, rect.bottom, band_h*2):
                bnd_h = min(band_h-1, rect.bottom - by)
                if bnd_h < 1: continue
                pygame.draw.rect(surface, tuple(min(255, c+20) for c in color),
                    (rect.left+1, by+1, rect.width-2, bnd_h), border_radius=1)

        # ── Roof ─────────────────────────────────────────
        roof_col = tuple(min(255, c+36) for c in color)
        pygame.draw.rect(surface, roof_col,
            pygame.Rect(rect.left, rect.top, rect.width, 5), border_radius=5)
        # Roof edge highlight
        pygame.draw.line(surface, tuple(min(255,c+70) for c in color),
            (rect.left+2, rect.top+1), (rect.right-3, rect.top+1), 1)

        # ── Windows ──────────────────────────────────────
        wr, wc2 = b["win_rows"], b["win_cols"]
        ww = max(4, rect.width  // wc2 - 4)
        wh = max(4, rect.height // wr  - 5)
        for r in range(wr):
            for c in range(wc2):
                wx = rect.left + c * (rect.width  // wc2) + 3
                wy = rect.top  + r * (rect.height // wr)  + 6
                lit_val = (seed * 7 + wx * 3 + wy * 2) % 10
                if lit_val < 6:
                    # Warm lit window
                    wcol = (255, 238, 160) if lit_val < 3 else (255, 210, 100)
                    pygame.draw.rect(surface, wcol, (wx, wy, ww, wh), border_radius=1)
                    # Window frame
                    pygame.draw.rect(surface, tuple(max(0,c2-30) for c2 in color),
                        (wx, wy, ww, wh), 1, border_radius=1)
                    # Glare dot
                    pygame.draw.rect(surface, (255,255,220), (wx+1, wy+1, ww//3, 1))
                elif lit_val < 8:
                    # Dark / unlit
                    dark = tuple(max(0, c2-80) for c2 in color)
                    pygame.draw.rect(surface, dark, (wx, wy, ww, wh), border_radius=1)
                else:
                    # Blue office window
                    pygame.draw.rect(surface, (55, 110, 180), (wx, wy, ww, wh), border_radius=1)
                    pygame.draw.rect(surface, (90, 155, 215), (wx+1, wy+1, ww//2, 1))

        # ── Outline ──────────────────────────────────────
        pygame.draw.rect(surface, tuple(max(0,c-30) for c in color),
            rect, 1, border_radius=5)

        # ── Rooftop details ──────────────────────────────
        if b.get("antenna"):
            ax = rect.centerx
            ay = rect.top
            pygame.draw.line(surface, (100, 105, 120), (ax, ay), (ax, ay-12), 2)
            pygame.draw.circle(surface, (220, 60, 60), (ax, ay-13), 2)
            # Crossbar
            pygame.draw.line(surface, (90, 95, 110), (ax-4, ay-7), (ax+4, ay-7), 1)

        if b.get("watertower") and rect.width >= 20:
            wtx = rect.left + rect.width//4
            wty = rect.top - 10
            wtw, wth = max(10, rect.width//3), 8
            # Tank
            pygame.draw.ellipse(surface, (75, 68, 58), (wtx, wty, wtw, wth))
            pygame.draw.ellipse(surface, (95, 88, 75), (wtx, wty, wtw, wth-2))
            # Legs
            pygame.draw.line(surface, (65, 62, 55), (wtx+2, wty+wth), (wtx+2, rect.top), 1)
            pygame.draw.line(surface, (65, 62, 55), (wtx+wtw-2, wty+wth), (wtx+wtw-2, rect.top), 1)


def _draw_curbs(surface: pygame.Surface):
    half, cw = ROAD_WIDTH // 2, 5
    c = COLOR_CURB
    pygame.draw.rect(surface, c, (CENTER_X - half - cw, 0, cw, WINDOW_HEIGHT))
    pygame.draw.rect(surface, c, (CENTER_X + half,      0, cw, WINDOW_HEIGHT))
    pygame.draw.rect(surface, c, (0, CENTER_Y - half - cw, WINDOW_WIDTH, cw))
    pygame.draw.rect(surface, c, (0, CENTER_Y + half,      WINDOW_WIDTH, cw))


def _draw_lane_markings(surface: pygame.Surface):
    half  = ROAD_WIDTH // 2
    dash, gap  = 14, 10
    color = COLOR_LANE_MARK
    # Vertical centre line
    y = 0
    while y < CENTER_Y - half:
        pygame.draw.line(surface, color, (CENTER_X, y), (CENTER_X, y + dash), 2)
        y += dash + gap
    y = CENTER_Y + half
    while y < WINDOW_HEIGHT:
        pygame.draw.line(surface, color, (CENTER_X, y), (CENTER_X, y + dash), 2)
        y += dash + gap
    # Horizontal centre line
    x = 0
    while x < CENTER_X - half:
        pygame.draw.line(surface, color, (x, CENTER_Y), (x + dash, CENTER_Y), 2)
        x += dash + gap
    x = CENTER_X + half
    while x < WINDOW_WIDTH:
        pygame.draw.line(surface, color, (x, CENTER_Y), (x + dash, CENTER_Y), 2)
        x += dash + gap


def _draw_stop_lines(surface: pygame.Surface):
    half, dist = ROAD_WIDTH // 2, STOP_LINE_DIST
    c, lw = (215, 215, 215), 3
    pygame.draw.line(surface, c, (CENTER_X, CENTER_Y - dist), (CENTER_X + half, CENTER_Y - dist), lw)
    pygame.draw.line(surface, c, (CENTER_X - half, CENTER_Y + dist), (CENTER_X, CENTER_Y + dist), lw)
    pygame.draw.line(surface, c, (CENTER_X + dist, CENTER_Y), (CENTER_X + dist, CENTER_Y + half), lw)
    pygame.draw.line(surface, c, (CENTER_X - dist, CENTER_Y - half), (CENTER_X - dist, CENTER_Y), lw)


def _draw_zebra(surface: pygame.Surface):
    half  = ROAD_WIDTH // 2
    dist  = STOP_LINE_DIST + 5
    zw, zn = 7, 5
    lt, dk = (215, 215, 195), (75, 78, 90)
    for x0, x1, yb in [(CENTER_X, CENTER_X + half, CENTER_Y - dist),
                       (CENTER_X - half, CENTER_X, CENTER_Y + dist)]:
        for i in range(zn):
            y = yb + i * (zw * 2) - zn * zw
            pygame.draw.rect(surface, lt if i % 2 == 0 else dk, (x0, y, x1 - x0, zw))
    for xb, y0, y1 in [(CENTER_X + dist, CENTER_Y, CENTER_Y + half),
                       (CENTER_X - dist, CENTER_Y - half, CENTER_Y)]:
        for i in range(zn):
            x = xb + i * (zw * 2) - zn * zw
            pygame.draw.rect(surface, lt if i % 2 == 0 else dk, (x, y0, zw, y1 - y0))


# ─────────────────────────────────────────────────────────
#  Live signal lights
# ─────────────────────────────────────────────────────────

def draw_signals(surface: pygame.Surface, controller):
    for direction, (px, py) in _POLE_POSITIONS.items():
        state    = controller.signal_color(direction)
        bw2, bh2 = 18, 48
        box_rect  = pygame.Rect(px - bw2 // 2, py - 62 - bh2, bw2, bh2)

        lights = [
            ("red",    box_rect.top + bh2 // 6,     COLOR_SIGNAL_RED),
            ("yellow", box_rect.top + bh2 // 2,     COLOR_SIGNAL_YELLOW),
            ("green",  box_rect.bottom - bh2 // 6,  COLOR_SIGNAL_GREEN),
        ]
        light_x = box_rect.centerx
        for name, ly, lit_color in lights:
            is_on = (
                (name == "red"    and state == "red")    or
                (name == "yellow" and state == "yellow") or
                (name == "green"  and state == "green")
            )
            draw_color = lit_color if is_on else COLOR_SIGNAL_OFF
            pygame.draw.circle(surface, draw_color, (light_x, ly), 7)
            if is_on:
                glow = _get_glow(lit_color, 32, 55)
                surface.blit(glow, (light_x - 16, ly - 16))


# ─────────────────────────────────────────────────────────
#  Dashboard / HUD
# ─────────────────────────────────────────────────────────
DASH_X, DASH_Y = 10, 10
DASH_W, DASH_H = 230, 520


def draw_dashboard(surface: pygame.Surface, fonts: dict,
                   controller, queues: dict, total_passed: int):
    # Translucent panel
    panel = pygame.Surface((DASH_W, DASH_H), pygame.SRCALPHA)
    panel.fill((16, 20, 30, 220))
    pygame.draw.rect(panel, (*COLOR_DASH_BORDER, 210),
                     (0, 0, DASH_W, DASH_H), 2, border_radius=12)
    surface.blit(panel, (DASH_X, DASH_Y))

    y     = DASH_Y + 14
    lh    = 24
    small = fonts["small"]
    med   = fonts["medium"]

    def txt(text, fx, fy, font, color):
        surface.blit(font.render(text, True, color), (fx, fy))

    def sep():
        nonlocal y
        pygame.draw.line(surface, COLOR_DASH_BORDER,
                         (DASH_X + 6, y), (DASH_X + DASH_W - 6, y), 1)
        y += 7

    # Title
    txt("AI TRAFFIC CONTROL", DASH_X + 10, y, med, COLOR_HIGHLIGHT)
    y += lh + 2
    sep()

    # Mode
    mode_col = COLOR_EMERGENCY if controller.current_mode == "EMERGENCY" else COLOR_SUCCESS
    txt(f"Mode: {controller.current_mode}", DASH_X + 10, y, med, mode_col)
    y += lh

    # Green direction + timer
    gd = controller.green_dir or "—"
    txt(f"Green: {gd.capitalize()}", DASH_X + 10, y, med, COLOR_SIGNAL_GREEN)
    y += lh
    cd    = controller.countdown()
    txt(f"Timer: {cd:.1f}s", DASH_X + 10, y, med, COLOR_TEXT_BRIGHT)
    y += lh + 2
    # Progress bar
    max_t = max(controller.green_time, 1.0)
    bar_w = int((cd / max_t) * (DASH_W - 20))
    pygame.draw.rect(surface, (38, 48, 68), (DASH_X + 10, y, DASH_W - 20, 8), border_radius=4)
    if bar_w > 0:
        bc = COLOR_SIGNAL_GREEN if controller.phase == "green" else COLOR_SIGNAL_YELLOW
        pygame.draw.rect(surface, bc, (DASH_X + 10, y, bar_w, 8), border_radius=4)
    y += 14
    sep()

    # ── Queue Status ────────────────────────────────────
    txt("Queue Status", DASH_X + 10, y, med, COLOR_HIGHLIGHT)
    y += lh
    dir_colors = {
        "north": (100, 178, 255),
        "south": (255, 148, 100),
        "east":  (100, 218, 158),
        "west":  (220, 178, 100),
    }
    for d in DIRECTIONS:
        q       = queues.get(d, [])
        active  = [v for v in q if not v.passed]
        cnt     = len(active)
        ambu_cnt = sum(1 for v in active if v.vtype == "ambulance")
        sig     = controller.signal_color(d)
        sig_c   = {"red": COLOR_SIGNAL_RED,
                   "yellow": COLOR_SIGNAL_YELLOW,
                   "green": COLOR_SIGNAL_GREEN}[sig]

        pygame.draw.circle(surface, sig_c, (DASH_X + 18, y + 8), 6)
        txt(f"{d.capitalize():6s}:", DASH_X + 30, y, small, dir_colors[d])
        count_col = (255, 95, 95) if cnt >= 8 else (255, 218, 95) if cnt >= 4 else COLOR_TEXT_BRIGHT
        txt(f"{cnt:2d} veh", DASH_X + 112, y, small, count_col)

        # Ambulance indicator on same line
        if ambu_cnt > 0:
            pygame.draw.circle(surface, COLOR_AMBU_RED, (DASH_X + 190, y + 8), 4)
            txt(str(ambu_cnt), DASH_X + 196, y + 2, fonts["tiny"], COLOR_AMBU_RED)

        # Density bar
        blen = min(cnt, 12) * (75 // 12)
        b_c  = (175, 55, 55) if cnt >= 8 else (198, 175, 55) if cnt >= 4 else (55, 158, 78)
        pygame.draw.rect(surface, (30, 36, 52), (DASH_X + 10, y + lh - 4, 75, 4), border_radius=2)
        if blen > 0:
            pygame.draw.rect(surface, b_c, (DASH_X + 10, y + lh - 4, blen, 4), border_radius=2)
        y += lh + 5
    sep()

    # ── Ambulance Queue Section ──────────────────────────
    ambu_queue = controller.pending_ambulance_dirs
    txt(f"Ambulance Queue ({len(ambu_queue)})", DASH_X + 10, y, med,
        COLOR_EMERGENCY if ambu_queue else COLOR_TEXT_DIM)
    y += lh

    if ambu_queue:
        for idx, qd in enumerate(ambu_queue):
            prefix    = "► " if qd == controller.emergency_dir else f"{idx+1}. "
            row_color = COLOR_AMBU_RED if qd == controller.emergency_dir else (220, 140, 140)
            txt(f"{prefix}{qd.capitalize()}", DASH_X + 14, y, small, row_color)
            # Remaining timer if this is the active one
            if qd == controller.emergency_dir:
                secs = max(0.0, controller.emergency_timer)
                txt(f"{secs:.1f}s", DASH_X + 145, y, small, (255, 200, 100))
            y += lh - 2
    else:
        txt("  No ambulances", DASH_X + 14, y, small, COLOR_TEXT_DIM)
        y += lh
    sep()

    # Total passed
    txt(f"Total Passed: {total_passed}", DASH_X + 10, y, med, COLOR_TEXT_BRIGHT)


def draw_fps(surface: pygame.Surface, fonts: dict, fps: float):
    surface.blit(fonts["small"].render(f"FPS: {fps:.0f}", True, COLOR_TEXT_DIM),
                 (DASH_X + 10, DASH_Y + DASH_H + 6))


# ─────────────────────────────────────────────────────────
#  Emergency banner (animated top strip)
# ─────────────────────────────────────────────────────────

def draw_emergency_banner(surface: pygame.Surface, fonts: dict, controller):
    """Flashing ambulance banner at the very top of screen."""
    if not controller.emergency_active and not controller.pending_ambulance_dirs:
        return

    t = time.monotonic()
    if int(t * 3) % 2 == 1:
        return   # blink off

    ambu_dirs = controller.pending_ambulance_dirs
    active    = controller.emergency_dir

    bw, bh = 560, 42
    bx = CENTER_X - bw // 2
    by = 8

    banner = pygame.Surface((bw, bh), pygame.SRCALPHA)
    banner.fill((160, 15, 15, 235))
    pygame.draw.rect(banner, (255, 70, 70, 255), (0, 0, bw, bh), 3, border_radius=10)
    surface.blit(banner, (bx, by))

    if len(ambu_dirs) > 1:
        others = [d for d in ambu_dirs if d != active]
        others_str = ", ".join(d.upper() for d in others)
        label_txt = f"AMBULANCE PRIORITY  -  {active.upper()}  |  Waiting: {others_str}"
    elif active:
        label_txt = f"AMBULANCE PRIORITY  -  {active.upper()} SIDE"
    else:
        return

    lbl = fonts["banner"].render(label_txt, True, (255, 255, 255))
    surface.blit(lbl, (bx + bw // 2 - lbl.get_width() // 2,
                       by + bh // 2 - lbl.get_height() // 2))


# ─────────────────────────────────────────────────────────
#  Compass Rose
# ─────────────────────────────────────────────────────────

def draw_compass(surface: pygame.Surface, fonts: dict):
    cx, cy, r = WINDOW_WIDTH - 46, 46, 30   # stays in simulation area (left of panel)
    pygame.draw.circle(surface, (28, 33, 50), (cx, cy), r)
    pygame.draw.circle(surface, COLOR_DASH_BORDER, (cx, cy), r, 1)
    for lbl, (dx, dy) in [("N", (0, -1)), ("S", (0, 1)), ("E", (1, 0)), ("W", (-1, 0))]:
        lx, ly = cx + dx * (r - 8), cy + dy * (r - 8)
        s = fonts["tiny"].render(lbl, True, COLOR_TEXT_BRIGHT)
        surface.blit(s, (lx - s.get_width() // 2, ly - s.get_height() // 2))
    pygame.draw.polygon(surface, COLOR_HIGHLIGHT,
                        [(cx, cy - r + 7), (cx - 5, cy + 2), (cx + 5, cy + 2)])


# ─────────────────────────────────────────────────────────
#  Accident visual effects (additive overlay)
# ─────────────────────────────────────────────────────────

# Smoke particle state — persistent between frames for drift effect
_smoke_particles: list[dict] = []
_smoke_timer: float = 0.0


def _draw_crash_overlay(surface: pygame.Surface, fonts: dict, vehicle):
    """Red flashing outline + CRASH text + smoke for a single crashed vehicle."""
    t = time.monotonic()

    # ── Flashing red outline ────────────────────────────────
    blink = int(t * 5) % 2 == 0
    outline_color = (255, 40, 40) if blink else (255, 120, 40)

    vx, vy = int(vehicle.x), int(vehicle.y)

    if vehicle.direction in ("north", "south"):
        hw, hh = vehicle.width // 2 + 4, vehicle.height // 2 + 4
    else:
        hw, hh = vehicle.height // 2 + 4, vehicle.width // 2 + 4

    outline_rect = pygame.Rect(vx - hw, vy - hh, hw * 2, hh * 2)
    pygame.draw.rect(surface, outline_color, outline_rect, 3, border_radius=5)

    # ── Second glow ring (larger, semi-transparent) ─────────
    glow_surf = pygame.Surface((hw * 2 + 16, hh * 2 + 16), pygame.SRCALPHA)
    glow_col = (*outline_color, 60 if blink else 30)
    pygame.draw.rect(glow_surf, glow_col,
                     (0, 0, hw * 2 + 16, hh * 2 + 16), 4, border_radius=8)
    surface.blit(glow_surf, (vx - hw - 8, vy - hh - 8))

    # ── "🔥 CRASH" text above vehicle ───────────────────────
    crash_col = (255, 70, 70) if blink else (255, 180, 60)
    crash_lbl = fonts["medium"].render("CRASH", True, crash_col)
    surface.blit(crash_lbl, (vx - crash_lbl.get_width() // 2, vy - hh - 22))

    # ── Smoke effect — translucent gray circles drifting up ──
    global _smoke_particles, _smoke_timer
    _smoke_timer += 0.016  # ~60 fps dt approximation

    # Spawn new particles periodically
    if _smoke_timer >= 0.15:
        _smoke_timer = 0.0
        for _ in range(2):
            _smoke_particles.append({
                "x": vx + random.randint(-8, 8),
                "y": vy - hh,
                "r": random.randint(5, 12),
                "alpha": random.randint(90, 140),
                "vx": random.uniform(-0.4, 0.4),
                "vy": random.uniform(-1.5, -0.6),
                "life": 0.0,
                "max_life": random.uniform(0.8, 1.8),
            })

    # Update & draw particles
    alive = []
    for p in _smoke_particles:
        p["life"] += 0.016
        if p["life"] >= p["max_life"]:
            continue
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["r"] += 0.08  # grow slightly
        frac = p["life"] / p["max_life"]
        alpha = int(p["alpha"] * (1.0 - frac))

        smoke = pygame.Surface((int(p["r"]) * 2, int(p["r"]) * 2), pygame.SRCALPHA)
        gray = 130 + int(50 * frac)
        pygame.draw.circle(smoke, (gray, gray, gray, max(0, alpha)),
                           (int(p["r"]), int(p["r"])), int(p["r"]))
        surface.blit(smoke, (int(p["x"] - p["r"]), int(p["y"] - p["r"])))
        alive.append(p)

    _smoke_particles = alive


def draw_accident_effects(surface: pygame.Surface, fonts: dict,
                          vehicles: list, accident_system):
    """
    Draw all accident-related visual effects.
    Called from Simulation._draw() AFTER all normal drawing.
    """
    global _smoke_particles

    if not accident_system.accident_active:
        _smoke_particles.clear()
        return

    # ── Draw crash overlay on crashed vehicles ──────────────
    for v in vehicles:
        if getattr(v, "is_crashed", False):
            _draw_crash_overlay(surface, fonts, v)

    # ── Banner positioning: stacked below emergency banner ──
    # Emergency banner occupies y=8..50, so accident banners start at y=56
    t = time.monotonic()
    blink = int(t * 3) % 2 == 0
    next_y = 56  # starting Y for accident banners

    # ── Banner 1: ACCIDENT DETECTED ─────────────────────────
    bw, bh = 500, 40
    bx = CENTER_X - bw // 2

    banner1 = pygame.Surface((bw, bh), pygame.SRCALPHA)
    banner1.fill((175, 18, 18, 240))
    pygame.draw.rect(banner1, (255, 55, 55, 255), (0, 0, bw, bh), 3, border_radius=10)
    surface.blit(banner1, (bx, next_y))

    lbl = fonts["banner"].render("ACCIDENT DETECTED  -  LANE BLOCKED", True, (255, 255, 255))
    surface.blit(lbl, (bx + bw // 2 - lbl.get_width() // 2,
                       next_y + bh // 2 - lbl.get_height() // 2))
    next_y += bh + 6

    # ── Banner 2: Lane closure / diversion status ───────────
    direction = accident_system.blocked_direction or "?"
    divert_to = (accident_system.diversion_direction or "?").upper()

    if accident_system._clearing_active:
        bw2, bh2 = 520, 36
        bx2 = CENTER_X - bw2 // 2
        banner2 = pygame.Surface((bw2, bh2), pygame.SRCALPHA)
        banner2.fill((150, 90, 10, 235))
        pygame.draw.rect(banner2, (255, 180, 40, 240), (0, 0, bw2, bh2), 2, border_radius=8)
        surface.blit(banner2, (bx2, next_y))

        clear_txt = f"{direction.upper()} Lane CLOSED  -  Diverting to {divert_to}  ({accident_system._vehicles_cleared} cleared)"
        lbl2 = fonts["medium"].render(clear_txt, True, (255, 245, 200))
        surface.blit(lbl2, (bx2 + bw2 // 2 - lbl2.get_width() // 2,
                            next_y + bh2 // 2 - lbl2.get_height() // 2))
    else:
        bw2, bh2 = 500, 36
        bx2 = CENTER_X - bw2 // 2
        banner2 = pygame.Surface((bw2, bh2), pygame.SRCALPHA)
        banner2.fill((15, 110, 55, 235))
        pygame.draw.rect(banner2, (55, 200, 100, 240), (0, 0, bw2, bh2), 2, border_radius=8)
        surface.blit(banner2, (bx2, next_y))

        done_txt = f"{direction.upper()} Lane Cleared  -  {accident_system._vehicles_cleared} vehicles diverted to {divert_to}"
        lbl2 = fonts["medium"].render(done_txt, True, (200, 255, 220))
        surface.blit(lbl2, (bx2 + bw2 // 2 - lbl2.get_width() // 2,
                            next_y + bh2 // 2 - lbl2.get_height() // 2))
    next_y += bh2 + 6

    # ── Banner 3: Ambulance dispatch status ─────────────────
    if accident_system._ambulance_dispatched:
        bw3, bh3 = 360, 32
        bx3 = CENTER_X - bw3 // 2
        banner3 = pygame.Surface((bw3, bh3), pygame.SRCALPHA)
        banner3.fill((15, 80, 160, 230))
        pygame.draw.rect(banner3, (60, 155, 255, 230), (0, 0, bw3, bh3), 2, border_radius=8)
        surface.blit(banner3, (bx3, next_y))

        amb_lbl = fonts["medium"].render("Emergency Services En Route", True, (210, 235, 255))
        surface.blit(amb_lbl, (bx3 + bw3 // 2 - amb_lbl.get_width() // 2,
                               next_y + bh3 // 2 - amb_lbl.get_height() // 2))
    else:
        remaining = max(0.0, accident_system._dispatch_delay
                        - accident_system._ambulance_dispatch_timer)
        if remaining > 0:
            disp_txt = fonts["small"].render(
                f"Dispatching emergency services in {remaining:.0f}s...",
                True, (190, 210, 230))
            surface.blit(disp_txt, (CENTER_X - disp_txt.get_width() // 2, next_y + 4))


# ─────────────────────────────────────────────────────────
#  Side Panel  (in-window dashboard)
# ─────────────────────────────────────────────────────────

# Event log buffer (latest events shown in panel)
_event_log: list[str] = []
_event_log_max = 12


def log_panel_event(msg: str):
    """Called from anywhere to push an event into the side-panel log."""
    _event_log.append(msg)
    if len(_event_log) > _event_log_max:
        _event_log.pop(0)


def draw_side_panel(surface: pygame.Surface, fonts: dict,
                    controller, queues: dict,
                    total_passed: int, fps: float,
                    predictor, accident_system,
                    emergency_count: int):
    """Compact dark panel on the right side with all dashboard info."""
    PX  = WINDOW_WIDTH
    PW  = PANEL_WIDTH
    PH  = WINDOW_HEIGHT
    PAD = 10
    IND = PX + PAD

    # ── Background ──────────────────────────────────────────
    panel = pygame.Surface((PW, PH), pygame.SRCALPHA)
    panel.fill((12, 16, 26, 250))
    pygame.draw.line(panel, (60, 90, 140), (0, 0), (0, PH), 2)
    surface.blit(panel, (PX, 0))

    med   = fonts["medium"]
    small = fonts["small"]
    tiny  = fonts["tiny"]
    y = 8

    def txt(text, fx, fy, font, color):
        surface.blit(font.render(text, True, color), (fx, fy))

    def header(label, color=COLOR_HIGHLIGHT):
        nonlocal y
        pygame.draw.rect(surface, (*color, 35), (IND - 2, y, PW - PAD * 2 + 4, 18))
        txt(label, IND + 2, y + 1, med, color)
        y += 20

    def sep():
        nonlocal y
        pygame.draw.line(surface, (35, 45, 65),
                         (IND, y), (PX + PW - PAD, y), 1)
        y += 5

    # ── HEADER ───────────────────────────────────────────────
    txt("LIVE DASHBOARD", IND, y, fonts["big"], (100, 170, 255))
    txt(f"FPS {fps:.0f}", PX + PW - PAD - 42, y + 4, tiny, COLOR_TEXT_DIM)
    y += 24
    sep()

    # ── SIGNAL STATUS ────────────────────────────────────────
    header("SIGNAL STATUS", (100, 170, 255))

    mode_col = COLOR_EMERGENCY if controller.current_mode == "EMERGENCY" else COLOR_SUCCESS
    txt(f"Mode: {controller.current_mode}", IND, y, small, mode_col)
    y += 16

    gd = controller.green_dir or "---"
    cd = controller.countdown()
    txt(f"Green: {gd.capitalize()}", IND, y, small, COLOR_SIGNAL_GREEN)
    txt(f"{cd:.1f}s", IND + 130, y, small, COLOR_TEXT_BRIGHT)
    y += 16

    # Phase progress bar
    max_t = max(controller.green_time, 1.0)
    bar_w = PW - PAD * 2
    fill = int((cd / max_t) * bar_w)
    pygame.draw.rect(surface, (28, 35, 50), (IND, y, bar_w, 5), border_radius=2)
    if fill > 0:
        bc = COLOR_SIGNAL_GREEN if controller.phase == "green" else COLOR_SIGNAL_YELLOW
        pygame.draw.rect(surface, bc, (IND, y, fill, 5), border_radius=2)
    y += 10
    sep()

    # ── QUEUE STATUS ─────────────────────────────────────────
    header("QUEUE STATUS", (100, 200, 160))
    txt(f"Passed: {total_passed}", IND + 160, y - 18, tiny, COLOR_SUCCESS)

    dir_colors = {
        "north": (100, 178, 255), "south": (255, 148, 100),
        "east":  (100, 218, 158), "west":  (220, 178, 100),
    }
    max_q = max((len([v for v in queues.get(d, []) if not v.passed]) for d in DIRECTIONS), default=1)
    max_q = max(max_q, 1)

    for d in DIRECTIONS:
        q = queues.get(d, [])
        active = [v for v in q if not v.passed]
        cnt = len(active)
        ambu_cnt = sum(1 for v in active if v.vtype == "ambulance")
        sig = controller.signal_color(d)
        sig_c = {"red": COLOR_SIGNAL_RED, "yellow": COLOR_SIGNAL_YELLOW,
                 "green": COLOR_SIGNAL_GREEN}[sig]

        # Signal dot + direction initial + count
        pygame.draw.circle(surface, sig_c, (IND + 5, y + 7), 4)
        txt(f"{d[0].upper()}", IND + 14, y, small, dir_colors[d])
        cnt_col = (255, 85, 85) if cnt >= 8 else (255, 210, 85) if cnt >= 4 else COLOR_TEXT_BRIGHT
        txt(f"{cnt:2d}", IND + 30, y, small, cnt_col)
        if ambu_cnt:
            pygame.draw.circle(surface, COLOR_AMBU_RED, (IND + 55, y + 7), 3)

        # Density bar
        bx = IND + 62
        bw = PW - PAD - 62 - PAD
        bfill = int((cnt / max_q) * bw) if max_q else 0
        b_col = (175, 55, 55) if cnt >= 8 else (198, 175, 55) if cnt >= 4 else (55, 158, 78)
        pygame.draw.rect(surface, (22, 30, 45), (bx, y + 3, bw, 8), border_radius=3)
        if bfill > 0:
            pygame.draw.rect(surface, b_col, (bx, y + 3, bfill, 8), border_radius=3)
        y += 17
    sep()

    # ── AMBULANCE ────────────────────────────────────────────
    ambu_q = controller.pending_ambulance_dirs
    a_col = COLOR_EMERGENCY if ambu_q else COLOR_TEXT_DIM
    header(f"AMBULANCE ({len(ambu_q)})", a_col)
    txt(f"Total: {emergency_count}", IND + 200, y - 18, tiny, COLOR_TEXT_DIM)

    if ambu_q:
        for idx, qd in enumerate(ambu_q[:3]):
            active_dir = qd == controller.emergency_dir
            col = COLOR_AMBU_RED if active_dir else (200, 130, 130)
            mark = ">>>" if active_dir else f" {idx+1}."
            txt(f"{mark} {qd.capitalize()}", IND + 2, y, small, col)
            if active_dir:
                secs = max(0.0, controller.emergency_timer)
                txt(f"{secs:.1f}s", IND + 140, y, small, (255, 200, 100))
            y += 15
    else:
        txt("None active", IND + 2, y, tiny, COLOR_TEXT_DIM)
        y += 13
    y += 2
    sep()

    # ── ACCIDENT STATUS ──────────────────────────────────────
    acc = accident_system
    if acc.accident_active:
        blink = int(time.monotonic() * 3) % 2 == 0
        card_c = (55, 15, 15, 210) if blink else (40, 12, 12, 210)
        cw, ch = PW - PAD * 2, 44
        card = pygame.Surface((cw, ch), pygame.SRCALPHA)
        card.fill(card_c)
        pygame.draw.rect(card, (255, 60, 60, 190), (0, 0, cw, ch), 2, border_radius=6)
        surface.blit(card, (IND, y))

        txt("ACCIDENT", IND + 8, y + 3, med, (255, 75, 75))
        blocked = (acc.blocked_direction or "?").upper()
        divert  = (acc.diversion_direction or "?").upper()
        txt(f"{blocked} lane | {acc.accident_duration:.0f}s", IND + 8, y + 19, small, (255, 180, 140))
        info = f"Divert -> {divert}"
        if acc._clearing_active:
            info += f" ({acc._vehicles_cleared} cleared)"
        else:
            info += f" ({acc._vehicles_cleared} done)"
        if acc._ambulance_dispatched:
            info += " AMB"
        txt(info, IND + 8, y + 33, tiny, (200, 200, 220))
        y += ch + 4
    else:
        header("ACCIDENT", COLOR_SUCCESS)
        txt("No accident", IND + 2, y, tiny, COLOR_TEXT_DIM)
        y += 13
    sep()

    # ── AI PREDICTIONS ───────────────────────────────────────
    pred_s = predictor.get_stats()
    preds  = pred_s.get("predictions", {})
    tput   = pred_s.get("throughput", 0)
    conf   = pred_s.get("avg_conf", 0)
    header("AI PREDICTIONS", (160, 145, 255))
    txt(f"{tput:.0f} veh/min | {conf:.0%} conf", IND + 2, y, tiny, COLOR_TEXT_DIM)
    y += 13

    if preds:
        txt("      +30s  +60s  +90s", IND + 36, y, tiny, (90, 100, 120))
        y += 12
        for d in DIRECTIONS:
            dp = preds.get(d, {})
            f30 = dp.get("forecast_30", 0)
            f60 = dp.get("forecast_60", 0)
            f90 = dp.get("forecast_90", 0)
            txt(f"{d[0].upper()}", IND + 4, y, tiny, dir_colors.get(d, COLOR_TEXT_BRIGHT))
            txt(f"  {f30:4.1f}  {f60:4.1f}  {f90:4.1f}", IND + 40, y, tiny, COLOR_TEXT_BRIGHT)
            y += 12
    else:
        txt("Collecting data...", IND + 2, y, tiny, COLOR_TEXT_DIM)
        y += 12
    y += 2
    sep()

    # ── EVENT LOG ────────────────────────────────────────────
    header("EVENT LOG", (160, 185, 210))
    max_vis = min(6, max(1, (PH - y - 40) // 12))
    if _event_log:
        for msg in reversed(_event_log[-max_vis:]):
            display = msg[:38] + ".." if len(msg) > 40 else msg
            txt(display, IND + 2, y, tiny, (145, 160, 180))
            y += 12
    else:
        txt("No events yet", IND + 2, y, tiny, (80, 90, 110))

    # ── CONTROLS ─────────────────────────────────────────────
    ctrl_y = PH - 30
    pygame.draw.line(surface, (35, 45, 65), (IND, ctrl_y - 4), (PX + PW - PAD, ctrl_y - 4), 1)
    txt("C:Crash R:Reset A:Ambu M:All ESC:Quit", IND, ctrl_y, tiny, (85, 100, 125))
    txt("Web: localhost:5000", IND, ctrl_y + 13, tiny, (85, 100, 125))
