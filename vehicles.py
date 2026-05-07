"""
=============================================================
  vehicles.py  –  Vehicle class with redesigned visuals
=============================================================
  Each vehicle is drawn on a canonical "facing up" surface
  and then rotated to match its travel direction.
  This gives clean, detail-rich visuals for all 5 types.
=============================================================
"""
import math
import random
import pygame
import numpy as np
from config import (
    VEHICLE_TYPES, VEHICLE_WEIGHTS, AMBULANCE_CHANCE,
    CENTER_X, CENTER_Y, ROAD_WIDTH, LANE_WIDTH, STOP_LINE_DIST,
    SAFE_FOLLOW_GAP, DECEL_DISTANCE, MIN_SPAWN_GAP, WINDOW_WIDTH, WINDOW_HEIGHT,
    COLOR_AMBU_BLUE, COLOR_AMBU_RED,
)

# ─────────────────────────────────────────────────────────
#  Rotation angles — canonical surface faces UP (south→north)
#    south approach → vehicle moves north  → facing up   →   0°
#    north approach → vehicle moves south  → facing down → 180°
#    east  approach → vehicle moves west   → facing left →  90° CCW
#    west  approach → vehicle moves east   → facing right→ -90° CCW
# ─────────────────────────────────────────────────────────
_ROTATE_ANGLE: dict[str, int] = {
    "south":   0,
    "north": 180,
    "east":   90,
    "west":  -90,
}

# Surface cache: (vtype, flash_state) → canonical pygame.Surface
_SURF_CACHE: dict[tuple, pygame.Surface] = {}


# ─────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────

def _incoming_lane_centre(direction: str) -> tuple[int, int]:
    """Centre of the incoming lane at the screen edge."""
    half = LANE_WIDTH // 2
    match direction:
        case "north": return CENTER_X + half, 0
        case "south": return CENTER_X - half, WINDOW_HEIGHT
        case "east":  return WINDOW_WIDTH, CENTER_Y + half
        case "west":  return 0, CENTER_Y - half
        case _: raise ValueError(direction)


def _stop_line_coord(direction: str) -> int:
    match direction:
        case "north": return CENTER_Y - STOP_LINE_DIST
        case "south": return CENTER_Y + STOP_LINE_DIST
        case "east":  return CENTER_X + STOP_LINE_DIST
        case "west":  return CENTER_X - STOP_LINE_DIST
        case _: raise ValueError(direction)


def _direction_vector(direction: str) -> tuple[int, int]:
    return {"north": (0, 1), "south": (0, -1),
            "east": (-1, 0), "west": (1, 0)}[direction]


# ─────────────────────────────────────────────────────────
#  Canonical vehicle surface builders
#  Convention: nose at TOP (y=pad), tail at BOTTOM (y=pad+bl)
# ─────────────────────────────────────────────────────────
PAD = 3   # px padding on each side of the canonical surface


def _tint(color: tuple, delta: int) -> tuple:
    return tuple(max(0, min(255, c + delta)) for c in color)


def _make_canonical(vtype: str, flash_on: bool) -> pygame.Surface:
    """Build a canonical (facing-up) surface for the given vehicle type."""
    props = VEHICLE_TYPES[vtype]
    bw, bl = props["width"], props["height"]
    color  = props["color"]

    surf = pygame.Surface((bw + PAD * 2, bl + PAD * 2), pygame.SRCALPHA)

    if vtype == "bike":
        _draw_bike(surf, bw, bl, color)
    elif vtype == "car":
        _draw_car(surf, bw, bl, color)
    elif vtype == "bus":
        _draw_bus(surf, bw, bl, color)
    elif vtype == "truck":
        _draw_truck(surf, bw, bl, color)
    elif vtype == "ambulance":
        _draw_ambulance(surf, bw, bl, color, flash_on)
    return surf


def _get_surface(vtype: str, direction: str, flash_on: bool) -> pygame.Surface:
    """Return a cached, rotated surface for the given vehicle state."""
    # Include direction in key so each rotation is cached separately
    key = (vtype, direction, flash_on)
    if key not in _SURF_CACHE:
        canonical = _make_canonical(vtype, flash_on)
        angle     = _ROTATE_ANGLE[direction]
        if angle == 0:
            _SURF_CACHE[key] = canonical
        else:
            _SURF_CACHE[key] = pygame.transform.rotate(canonical, angle)
    return _SURF_CACHE[key]


# ── BIKE ────────────────────────────────────────────────
def _draw_bike(surf: pygame.Surface, bw: int, bl: int, color: tuple):
    cx  = PAD + bw // 2
    wr  = max(5, bw // 2)
    ny  = PAD + wr + 1
    ty  = PAD + bl - wr - 1
    wc  = (22, 22, 28)
    rim = (110, 115, 130)

    # Rear wheel with rim
    pygame.draw.circle(surf, wc, (cx, ty), wr)
    pygame.draw.circle(surf, rim, (cx, ty), wr - 1, 2)
    pygame.draw.circle(surf, (180, 185, 200), (cx, ty), 2)
    # Spokes
    for ang in [0, 60, 120]:
        rad = math.radians(ang)
        pygame.draw.line(surf, rim,
            (cx, ty),
            (int(cx + math.cos(rad)*(wr-2)), int(ty + math.sin(rad)*(wr-2))), 1)

    # Front wheel with rim
    pygame.draw.circle(surf, wc, (cx, ny), wr)
    pygame.draw.circle(surf, rim, (cx, ny), wr - 1, 2)
    pygame.draw.circle(surf, (180, 185, 200), (cx, ny), 2)
    for ang in [0, 60, 120]:
        rad = math.radians(ang)
        pygame.draw.line(surf, rim,
            (cx, ny),
            (int(cx + math.cos(rad)*(wr-2)), int(ny + math.sin(rad)*(wr-2))), 1)

    frame_top = ny + wr
    frame_bot = ty - wr
    # Frame tubes
    pygame.draw.line(surf, _tint(color, -10), (cx, frame_top), (cx - 2, frame_bot), 3)
    pygame.draw.line(surf, _tint(color, 20),  (cx, frame_top), (cx + 1, frame_bot), 1)

    # Rider body
    rider_h = (frame_bot - frame_top) * 2 // 3
    rider_w = bw - 3
    rider_y = frame_top + (frame_bot - frame_top - rider_h) // 3
    pygame.draw.ellipse(surf, color, (cx - rider_w//2, rider_y, rider_w, rider_h))
    # Highlight on rider
    pygame.draw.ellipse(surf, _tint(color, 50),
        (cx - rider_w//2 + 1, rider_y + 1, rider_w//2, rider_h//3))
    pygame.draw.ellipse(surf, (18, 18, 22), (cx - rider_w//2, rider_y, rider_w, rider_h), 1)

    # Helmet with visor
    hcol = _tint(color, 55)
    pygame.draw.circle(surf, hcol, (cx, rider_y + 1), bw//2 - 1)
    pygame.draw.circle(surf, _tint(hcol, 40), (cx - 1, rider_y - 1), bw//4)
    pygame.draw.arc(surf, (80, 200, 255), (cx - bw//2 + 2, rider_y - 1, bw - 4, 5), 0, math.pi, 2)
    pygame.draw.circle(surf, (18, 18, 22), (cx, rider_y + 1), bw//2 - 1, 1)

    # Handlebar
    hb_y = frame_top + 1
    pygame.draw.line(surf, (190, 195, 210), (cx - bw//2+1, hb_y), (cx + bw//2-1, hb_y), 2)

    # LED Headlight
    pygame.draw.circle(surf, (255, 255, 160), (cx, PAD + 1), 3)
    pygame.draw.circle(surf, (255, 255, 255), (cx, PAD + 1), 1)


# ── CAR ─────────────────────────────────────────────────
def _draw_car(surf: pygame.Surface, bw: int, bl: int, color: tuple):
    x0, y0 = PAD, PAD

    hood_h  = bl * 20 // 100
    ws_h    = bl * 20 // 100
    roof_h  = bl * 24 // 100
    rws_h   = bl * 16 // 100
    trunk_h = bl - hood_h - ws_h - roof_h - rws_h

    glass   = (45, 90, 148)
    outline = (15, 15, 20)

    # Drop shadow
    pygame.draw.rect(surf, _tint(color, -70), (x0+3, y0+3, bw, bl), border_radius=7)
    # Body
    pygame.draw.rect(surf, color, (x0, y0, bw, bl), border_radius=7)
    # Side panel highlight (left edge)
    pygame.draw.rect(surf, _tint(color, 45), (x0+1, y0+hood_h, 2, ws_h+roof_h), border_radius=1)
    # Hood
    pygame.draw.rect(surf, _tint(color, -20), (x0, y0, bw, hood_h), border_radius=7)
    pygame.draw.rect(surf, _tint(color, 30), (x0+2, y0+1, bw-4, 3), border_radius=2)

    # Windshield
    ws_y = y0 + hood_h
    pygame.draw.rect(surf, glass, (x0+2, ws_y, bw-4, ws_h), border_radius=3)
    pygame.draw.rect(surf, _tint(glass, 70), (x0+3, ws_y+1, (bw-4)//3, ws_h-2), border_radius=2)
    # Wipers
    pygame.draw.line(surf, (30,35,45), (x0+3, ws_y+ws_h-2), (x0+bw//2, ws_y+2), 1)

    # Roof
    roof_y = ws_y + ws_h
    pygame.draw.rect(surf, _tint(color, 18), (x0+1, roof_y, bw-2, roof_h), border_radius=3)
    pygame.draw.rect(surf, _tint(color, 55), (x0+2, roof_y+1, bw-4, 2), border_radius=1)

    # Rear window
    rws_y = roof_y + roof_h
    pygame.draw.rect(surf, _tint(glass, -20), (x0+2, rws_y, bw-4, rws_h), border_radius=3)

    # Trunk
    trunk_y = rws_y + rws_h
    pygame.draw.rect(surf, _tint(color, -22), (x0, trunk_y, bw, trunk_h+1), border_radius=7)

    # Body outline
    pygame.draw.rect(surf, outline, (x0, y0, bw, bl), 1, border_radius=7)

    # Wheels with chrome rims
    ww, wh = max(5, bw//4), max(5, bl//8)
    for wx, wy in [(x0, y0+5), (x0+bw-ww, y0+5),
                   (x0, y0+bl-5-wh), (x0+bw-ww, y0+bl-5-wh)]:
        pygame.draw.ellipse(surf, (18, 18, 22), (wx, wy, ww, wh))
        pygame.draw.ellipse(surf, (80, 85, 100), (wx+1, wy+1, ww-2, wh-2))
        pygame.draw.ellipse(surf, (140, 145, 165), (wx+2, wy+1, ww-4, wh-2), 1)

    # LED Headlights
    pygame.draw.rect(surf, (255, 252, 180), (x0+1, y0+1, bw-2, 3), border_radius=2)
    pygame.draw.rect(surf, (255, 255, 255), (x0+2, y0+1, 4, 2))
    pygame.draw.rect(surf, (255, 255, 255), (x0+bw-6, y0+1, 4, 2))
    # DRL strip
    pygame.draw.line(surf, (220, 230, 255), (x0+2, y0+4), (x0+bw-2, y0+4), 1)

    # Tail lights (LED bar style)
    pygame.draw.rect(surf, (180, 20, 20), (x0+1, y0+bl-3, bw-2, 3), border_radius=2)
    pygame.draw.rect(surf, (255, 60, 60), (x0+2, y0+bl-3, 5, 2))
    pygame.draw.rect(surf, (255, 60, 60), (x0+bw-7, y0+bl-3, 5, 2))


# ── BUS ─────────────────────────────────────────────────
def _draw_bus(surf: pygame.Surface, bw: int, bl: int, color: tuple):
    x0, y0 = PAD, PAD
    outline = (15, 15, 20)

    # Drop shadow
    pygame.draw.rect(surf, _tint(color, -65), (x0+3, y0+3, bw, bl), border_radius=6)
    # Body
    pygame.draw.rect(surf, color, (x0, y0, bw, bl), border_radius=6)
    # Side shine stripe
    pygame.draw.rect(surf, _tint(color, 40), (x0+1, y0+8, 2, bl-16), border_radius=1)
    # Roof
    pygame.draw.rect(surf, _tint(color, 35), (x0, y0, bw, 6), border_radius=6)
    pygame.draw.rect(surf, _tint(color, 60), (x0+2, y0+1, bw-4, 2), border_radius=1)

    # Destination LED sign
    dest_h = max(7, bl // 9)
    pygame.draw.rect(surf, (18, 18, 22), (x0+2, y0+2, bw-4, dest_h), border_radius=3)
    pygame.draw.rect(surf, (255, 210, 0), (x0+3, y0+3, bw-6, dest_h-2), border_radius=2)
    pygame.draw.rect(surf, (255, 240, 80), (x0+4, y0+4, (bw-8)//2, 2))

    # Windshield (panoramic)
    ws_y = y0 + dest_h + 2
    ws_h = bl // 6
    pygame.draw.rect(surf, (40, 80, 138), (x0+1, ws_y, bw-2, ws_h), border_radius=3)
    pygame.draw.rect(surf, (80, 140, 200), (x0+2, ws_y+1, (bw-4)//2, ws_h-2), border_radius=2)
    # Wiper
    pygame.draw.line(surf, (20,25,35), (x0+2, ws_y+ws_h-2), (x0+bw//2+2, ws_y+2), 1)

    # Passenger windows (rows x cols)
    win_start_y = ws_y + ws_h + 3
    win_area_h  = bl - (win_start_y - y0) - 12
    rows, cols  = 5, 2
    wp = 2
    w_w = (bw - 4 - (cols-1)*wp) // cols
    w_h = (win_area_h - (rows-1)*wp) // rows
    for r in range(rows):
        for c in range(cols):
            wx = x0 + 2 + c*(w_w+wp)
            wy = win_start_y + r*(w_h+wp)
            pygame.draw.rect(surf, (38, 78, 128), (wx, wy, w_w, w_h), border_radius=2)
            pygame.draw.rect(surf, (80, 145, 210), (wx+1, wy+1, w_w//2, 2))
            pygame.draw.rect(surf, (60, 120, 180), (wx, wy, w_w, w_h), 1, border_radius=2)

    # Decorative waist stripe
    stripe_y = win_start_y + (rows*(w_h+wp))//2 - 1
    pygame.draw.rect(surf, _tint(color, -40), (x0, stripe_y, bw, 3))
    pygame.draw.rect(surf, _tint(color, 20),  (x0, stripe_y, bw, 1))

    # Wheels with chrome
    wr = max(7, bw//4)
    for wx, wy in [(x0+1, y0+bl//5-wr), (x0+bw-1, y0+bl//5-wr),
                   (x0+1, y0+bl-bl//5), (x0+bw-1, y0+bl-bl//5)]:
        pygame.draw.circle(surf, (18, 18, 22), (wx, wy), wr)
        pygame.draw.circle(surf, (75, 80, 95), (wx, wy), wr-2, 2)
        pygame.draw.circle(surf, (160, 165, 185), (wx, wy), 2)

    # LED headlights
    pygame.draw.rect(surf, (255, 252, 180), (x0+1, y0+dest_h+1, bw-2, 3), border_radius=1)
    pygame.draw.rect(surf, (255, 255, 220), (x0+2, y0+dest_h+1, 5, 2))
    pygame.draw.rect(surf, (255, 255, 220), (x0+bw-7, y0+dest_h+1, 5, 2))

    pygame.draw.rect(surf, outline, (x0, y0, bw, bl), 1, border_radius=6)

    # LED tail bar
    pygame.draw.rect(surf, (160, 20, 20), (x0+1, y0+bl-4, bw-2, 3), border_radius=1)
    pygame.draw.rect(surf, (255, 50, 50), (x0+2, y0+bl-4, 5, 2))
    pygame.draw.rect(surf, (255, 50, 50), (x0+bw-7, y0+bl-4, 5, 2))


# ── TRUCK ────────────────────────────────────────────────
def _draw_truck(surf: pygame.Surface, bw: int, bl: int, color: tuple):
    x0, y0  = PAD, PAD
    outline  = (12, 12, 16)

    cab_h     = bl * 30 // 100
    gap_h     = 4
    trailer_y = y0 + cab_h + gap_h
    trailer_h = bl - cab_h - gap_h
    cab_color = _tint(color, -30)
    trl_color = color

    # ── Trailer ─────────────────────────────────────────
    pygame.draw.rect(surf, _tint(trl_color, -65), (x0+3, trailer_y+3, bw, trailer_h), border_radius=4)
    pygame.draw.rect(surf, trl_color, (x0, trailer_y, bw, trailer_h), border_radius=4)
    # Shine stripe on trailer
    pygame.draw.rect(surf, _tint(trl_color, 50), (x0+1, trailer_y+2, 2, trailer_h-4), border_radius=1)
    # Horizontal rib lines
    for i in range(1, 4):
        vy = trailer_y + i * (trailer_h // 4)
        pygame.draw.line(surf, _tint(trl_color, -30), (x0+2, vy), (x0+bw-2, vy), 1)
    # Safety reflector stripe
    stripe_y = trailer_y + trailer_h * 3 // 4
    for sx in range(x0, x0+bw, 6):
        col = (255, 180, 0) if (sx//6) % 2 == 0 else (200, 30, 30)
        pygame.draw.rect(surf, col, (sx, stripe_y, 3, 4))
    pygame.draw.rect(surf, outline, (x0, trailer_y, bw, trailer_h), 1, border_radius=4)
    # Tail lights LED
    pygame.draw.rect(surf, (160, 20, 20), (x0+1, y0+bl-4, bw-2, 3), border_radius=1)
    pygame.draw.rect(surf, (255, 55, 55), (x0+2, y0+bl-3, 5, 1))
    pygame.draw.rect(surf, (255, 55, 55), (x0+bw-7, y0+bl-3, 5, 1))

    # ── Cab ─────────────────────────────────────────────
    cab_inset = 2
    cab_x = x0 + cab_inset
    cab_w = bw - 2 * cab_inset
    pygame.draw.rect(surf, _tint(cab_color, -55), (cab_x+2, y0+2, cab_w, cab_h), border_radius=6)
    pygame.draw.rect(surf, cab_color, (cab_x, y0, cab_w, cab_h), border_radius=6)
    # Cab highlight
    pygame.draw.rect(surf, _tint(cab_color, 45), (cab_x+1, y0+2, 2, cab_h-4), border_radius=1)
    pygame.draw.rect(surf, _tint(cab_color, 30), (cab_x+2, y0+1, cab_w-4, 2), border_radius=1)

    # Windshield
    ws_y = y0 + 3
    ws_h = cab_h * 42 // 100
    pygame.draw.rect(surf, (42, 82, 138), (cab_x+2, ws_y, cab_w-4, ws_h), border_radius=3)
    pygame.draw.rect(surf, (80, 130, 178), (cab_x+3, ws_y+1, (cab_w-4)//3, ws_h-2), border_radius=2)
    pygame.draw.line(surf, (25,30,40), (cab_x+3, ws_y+ws_h-2), (cab_x+cab_w//2, ws_y+2), 1)

    # Exhaust stacks
    stk_y = y0 + ws_h + 4
    stk_h = cab_h - ws_h - 6
    for stk_x in [cab_x+1, cab_x+cab_w-4]:
        pygame.draw.rect(surf, (60, 62, 70), (stk_x, stk_y, 3, stk_h), border_radius=1)
        pygame.draw.rect(surf, (90, 92, 105), (stk_x+1, stk_y, 1, stk_h))

    # LED headlight bar
    pygame.draw.rect(surf, (255, 252, 180), (cab_x+1, y0+1, cab_w-2, 3), border_radius=2)
    pygame.draw.rect(surf, (255, 255, 220), (cab_x+2, y0+1, 5, 2))
    pygame.draw.rect(surf, (255, 255, 220), (cab_x+cab_w-7, y0+1, 5, 2))
    pygame.draw.rect(surf, outline, (cab_x, y0, cab_w, cab_h), 1, border_radius=6)

    # ── Wheels (6) with chrome rims ─────────────────────
    wr = max(5, bw//5)
    for wx in [x0, x0+bw]:
        pygame.draw.circle(surf, (18,18,22), (wx, y0+cab_h-4), wr)
        pygame.draw.circle(surf, (80,85,100), (wx, y0+cab_h-4), wr-2, 2)
        pygame.draw.circle(surf, (155,160,180), (wx, y0+cab_h-4), 2)
    axle1_y = trailer_y + trailer_h * 35 // 100
    axle2_y = trailer_y + trailer_h * 70 // 100
    for axle_y in [axle1_y, axle2_y]:
        for wx in [x0, x0+bw]:
            pygame.draw.circle(surf, (18,18,22), (wx, axle_y), wr)
            pygame.draw.circle(surf, (80,85,100), (wx, axle_y), wr-2, 2)
            pygame.draw.circle(surf, (155,160,180), (wx, axle_y), 2)


# ── AMBULANCE ────────────────────────────────────────────
def _draw_ambulance(surf: pygame.Surface, bw: int, bl: int,
                    color: tuple, flash_on: bool):
    x0, y0 = PAD, PAD
    white   = (245, 245, 252)
    outline = (160, 165, 175)
    red_c   = (210, 30, 30)

    # Drop shadow
    pygame.draw.rect(surf, (145, 148, 158), (x0+3, y0+3, bw, bl), border_radius=6)
    # Body — white with subtle warm tint
    pygame.draw.rect(surf, white, (x0, y0, bw, bl), border_radius=6)
    # Side shine
    pygame.draw.rect(surf, (255, 255, 255), (x0+1, y0+4, 2, bl-8), border_radius=1)

    # ── LED Light bar ───────────────────────────────────
    lb_h = max(6, bl // 7)
    left_col  = COLOR_AMBU_BLUE if flash_on else COLOR_AMBU_RED
    right_col = COLOR_AMBU_RED  if flash_on else COLOR_AMBU_BLUE
    # Light bar housing
    pygame.draw.rect(surf, (30, 32, 40), (x0, y0, bw, lb_h), border_radius=4)
    # Left lamp
    pygame.draw.rect(surf, left_col,  (x0+1,      y0+1, bw//2-1, lb_h-2), border_radius=3)
    # Right lamp
    pygame.draw.rect(surf, right_col, (x0+bw//2,  y0+1, bw//2-1, lb_h-2), border_radius=3)
    # Lens highlights
    pygame.draw.rect(surf, _tint(left_col, 80),  (x0+2,      y0+2, (bw//2-3)//2, 2), border_radius=1)
    pygame.draw.rect(surf, _tint(right_col, 80), (x0+bw//2+1, y0+2, (bw//2-3)//2, 2), border_radius=1)
    # Glow halos
    for gcol, gx in [(left_col, x0+bw//4), (right_col, x0+3*bw//4)]:
        glow = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*gcol, 90), (10, 10), 10)
        surf.blit(glow, (gx-10, y0+lb_h//2-10))

    # Windshield
    ws_y = y0 + lb_h
    ws_h = bl * 17 // 100
    pygame.draw.rect(surf, (44, 88, 148), (x0+2, ws_y, bw-4, ws_h), border_radius=3)
    pygame.draw.rect(surf, (85, 138, 188), (x0+3, ws_y+1, (bw-4)//3, ws_h-2), border_radius=2)
    pygame.draw.line(surf, (25,30,40), (x0+3, ws_y+ws_h-2), (x0+bw//2, ws_y+2), 1)

    # Chevron emergency stripe
    stripe_y1 = ws_y + ws_h + 2
    stripe_h  = max(6, bl // 9)
    pygame.draw.rect(surf, red_c, (x0, stripe_y1, bw, stripe_h))
    # Chevron pattern in stripe
    mid_x = x0 + bw//2
    for i in range(3):
        cx2 = mid_x - i*(bw//5)
        pygame.draw.polygon(surf, _tint(red_c, 30),
            [(cx2, stripe_y1), (cx2+bw//10, stripe_y1+stripe_h//2),
             (cx2, stripe_y1+stripe_h), (cx2-bw//10, stripe_y1+stripe_h//2)])

    # Star-of-life cross
    cx_body  = x0 + bw // 2
    cross_y  = stripe_y1 + stripe_h + 4
    cross_sz = max(7, min(bw-6, (bl - (cross_y-y0)) // 2 - 3))
    arm_thick = max(3, cross_sz//3)
    # White circle
    pygame.draw.circle(surf, (255, 255, 255), (cx_body, cross_y+cross_sz//2), cross_sz//2+4)
    pygame.draw.circle(surf, red_c, (cx_body, cross_y+cross_sz//2), cross_sz//2+4, 1)
    # Cross arms
    pygame.draw.rect(surf, red_c, (cx_body-arm_thick//2, cross_y, arm_thick, cross_sz))
    pygame.draw.rect(surf, red_c, (cx_body-cross_sz//2, cross_y+cross_sz//2-arm_thick//2, cross_sz, arm_thick))

    # Outline
    pygame.draw.rect(surf, outline, (x0, y0, bw, bl), 1, border_radius=6)

    # LED headlights
    pygame.draw.rect(surf, (220, 235, 255), (x0+1, y0+lb_h, bw-2, 3), border_radius=1)
    pygame.draw.rect(surf, (255, 255, 255), (x0+2, y0+lb_h, 5, 2))
    pygame.draw.rect(surf, (255, 255, 255), (x0+bw-7, y0+lb_h, 5, 2))

    # LED tail lights
    pygame.draw.rect(surf, red_c, (x0+1, y0+bl-4, bw-2, 3), border_radius=1)
    pygame.draw.rect(surf, (255, 80, 80), (x0+2, y0+bl-4, 5, 2))
    pygame.draw.rect(surf, (255, 80, 80), (x0+bw-7, y0+bl-4, 5, 2))

    # Wheels with chrome rims
    ww, wh = max(5, bw//4), max(4, bl//9)
    for wx, wy in [(x0, y0+8), (x0+bw-ww, y0+8),
                   (x0, y0+bl-8-wh), (x0+bw-ww, y0+bl-8-wh)]:
        pygame.draw.ellipse(surf, (18, 18, 22), (wx, wy, ww, wh))
        pygame.draw.ellipse(surf, (85, 90, 108), (wx+1, wy+1, ww-2, wh-2))
        pygame.draw.ellipse(surf, (150, 155, 175), (wx+2, wy+1, ww-4, wh-2), 1)


# ─────────────────────────────────────────────────────────
#  Vehicle class
# ─────────────────────────────────────────────────────────

class Vehicle:
    _id_counter = 0

    def __init__(self, vtype: str, direction: str):
        Vehicle._id_counter += 1
        self.id        = Vehicle._id_counter
        self.vtype     = vtype
        self.direction = direction

        props          = VEHICLE_TYPES[vtype]
        self.width     = props["width"]
        self.height    = props["height"]
        self.speed     = props["speed"]
        self.color     = props["color"]
        self.label     = props["label"]
        self.weight    = props["weight"]

        self.dx, self.dy = _direction_vector(direction)

        # Travel-axis body length
        if direction in ("north", "south"):
            self.body_len   = self.height
            self.body_width = self.width
        else:
            self.body_len   = self.width
            self.body_width = self.height

        # Spawn at edge of screen (incoming lane)
        sx, sy    = _incoming_lane_centre(direction)
        self.x    = float(sx - self.dx * self.body_len // 2)
        self.y    = float(sy - self.dy * self.body_len // 2)

        # State
        self.passed        = False
        self.in_junction   = False
        self.current_speed = float(self.speed)
        self._target_speed = float(self.speed)

        # Stop-line constants
        self._stop_coord = _stop_line_coord(direction)
        self._stop_axis  = 1 if direction in ("north", "south") else 0

        # Ambulance flash timer
        self._flash_t = 0.0
        self.flash_on = False

        # Accident system flags (set externally by AccidentSystem)
        self.is_crashed  = False
        # ── Diversion state (3-phase waypoint turn) ──────────
        self.is_diverted       = False
        self._divert_phase     = 0      # 0=approach, 1=arc turn, 2=exit
        self._divert_wp        = [(0.0, 0.0)]  # approach waypoint (turn-start)
        self._divert_arc_center = (0.0, 0.0)   # center of circular arc
        self._divert_arc_r     = 40.0           # arc radius (px)
        self._divert_arc_a_start = 0.0          # arc start angle (radians)
        self._divert_arc_a_end   = 0.0          # arc end angle (radians)
        self._divert_arc_span    = 0.0          # signed angular span
        self._divert_arc_t       = 0.0          # arc progress [0..1]
        self._divert_exit_dx   = 0.0            # exit straight direction X
        self._divert_exit_dy   = 0.0            # exit straight direction Y
        self._divert_angle     = 0.0            # current heading for sprite draw

    # ── Nose (leading edge in direction of travel) ──────

    @property
    def nose_x(self) -> float:
        return self.x + self.dx * self.body_len / 2

    @property
    def nose_y(self) -> float:
        return self.y + self.dy * self.body_len / 2

    # ── Distance to stop line (positive = approaching) ──

    def dist_to_stop(self) -> float:
        if self._stop_axis == 1:
            if self.direction == "north": return self._stop_coord - self.nose_y
            else:                         return self.nose_y - self._stop_coord
        else:
            if self.direction == "east":  return self.nose_x - self._stop_coord
            else:                         return self._stop_coord - self.nose_x

    # ── Off-screen check ────────────────────────────────

    def is_off_screen(self) -> bool:
        m = 120
        return (self.x < -m or self.x > WINDOW_WIDTH  + m or
                self.y < -m or self.y > WINDOW_HEIGHT + m)

    # ── Physics update ──────────────────────────────────

    def update(self, dt: float, green: bool, vehicles_ahead: "list[Vehicle]"):
        # ── Accident: crashed vehicles are immobile ──────────
        if self.is_crashed:
            self.current_speed = 0.0
            self._target_speed = 0.0
            return

        # ── Diverted vehicles: 3-phase waypoint turn ─────────
        if self.is_diverted:
            spd = float(self.speed) * 0.75
            self.current_speed = min(self.current_speed + 100.0 * dt, spd)

            phase = self._divert_phase

            # ── Phase 0: APPROACH ── drive straight to turn-start point ──
            if phase == 0:
                tx, ty = self._divert_wp[0]
                dx_r = tx - self.x
                dy_r = ty - self.y
                dist_r = math.hypot(dx_r, dy_r)
                if dist_r <= max(6.0, self.current_speed * dt * 1.5):
                    self.x, self.y = tx, ty
                    self._divert_phase = 1
                    self._divert_arc_t = 0.0
                else:
                    nx, ny = dx_r / dist_r, dy_r / dist_r
                    self.x += nx * self.current_speed * dt
                    self.y += ny * self.current_speed * dt
                    self._divert_angle = math.atan2(-ny, nx)  # update for draw

            # ── Phase 1: TURN ── circular arc through intersection ────────
            elif phase == 1:
                cx_arc, cy_arc = self._divert_arc_center
                r_arc           = self._divert_arc_r
                a_start         = self._divert_arc_a_start
                a_end           = self._divert_arc_a_end
                arc_span        = self._divert_arc_span   # signed

                arc_len      = abs(arc_span) * r_arc
                arc_speed    = max(self.current_speed * 0.85, 30.0)
                dt_angle     = (arc_speed / max(r_arc, 1)) * dt
                self._divert_arc_t = min(
                    self._divert_arc_t + dt_angle / max(abs(arc_span), 0.001), 1.0)
                t = self._divert_arc_t

                ang = a_start + arc_span * t
                self.x = cx_arc + r_arc * math.cos(ang)
                self.y = cy_arc + r_arc * math.sin(ang)

                # Tangent direction = perpendicular to radius (in arc direction)
                tangent_ang = ang + math.pi / 2 * (1 if arc_span > 0 else -1)
                self._divert_angle = tangent_ang

                if t >= 1.0:
                    self._divert_phase = 2

            # ── Phase 2: EXIT ── straight out on diversion lane ──────────
            else:
                edx, edy = self._divert_exit_dx, self._divert_exit_dy
                self.x += edx * self.current_speed * dt
                self.y += edy * self.current_speed * dt
                self._divert_angle = math.atan2(-edy, edx)

            return


        # Ambulance flash
        if self.vtype == "ambulance":
            self._flash_t += dt
            if self._flash_t >= 0.28:
                self._flash_t = 0.0
                self.flash_on = not self.flash_on
                _SURF_CACHE.pop(("ambulance", self.direction, not self.flash_on), None)

        dist = self.dist_to_stop()

        # ── Target speed from signal ──────────────────────────────────
        if self.in_junction or self.passed:
            self._target_speed = float(self.speed)
        elif not green:
            if dist <= 0:
                self._target_speed = 0.0
            elif dist < DECEL_DISTANCE:
                self._target_speed = self.speed * max(0.0, dist / DECEL_DISTANCE)
            else:
                self._target_speed = float(self.speed)
        else:
            self._target_speed = float(self.speed)

        # ── Car-following model ───────────────────────────────────────
        # Only consider the single nearest leader (closest gap)
        nearest_gap = float("inf")
        nearest_leader = None
        for leader in vehicles_ahead:
            g = self._gap_to(leader)
            if g < nearest_gap:
                nearest_gap = g
                nearest_leader = leader

        if nearest_leader is not None:
            gap = nearest_gap
            if gap <= SAFE_FOLLOW_GAP:
                self._target_speed = 0.0
            elif gap < float(self.speed) * 0.8 + SAFE_FOLLOW_GAP * 3:
                # Proportional slow-down: linearly ramp between 0 and full speed
                frac = max(0.0, (gap - SAFE_FOLLOW_GAP) /
                           max(float(self.speed) * 0.8 + SAFE_FOLLOW_GAP * 2, 1.0))
                self._target_speed = min(self._target_speed, self.speed * frac)

        # ── Smooth acceleration / braking ─────────────────────────────
        accel = 180.0
        decel = 600.0   # hard braking — prevents running through stopped vehicles
        if self.current_speed < self._target_speed:
            self.current_speed = min(self._target_speed, self.current_speed + accel * dt)
        elif self.current_speed > self._target_speed:
            self.current_speed = max(self._target_speed, self.current_speed - decel * dt)

        # ── Move ──────────────────────────────────────────────────────
        move = self.current_speed * dt
        self.x += self.dx * move
        self.y += self.dy * move

        # ── Hard positional clamp (prevents any visual penetration) ───
        # After moving, ensure we haven't entered the leader's body.
        if nearest_leader is not None:
            gap_after = self._gap_to(nearest_leader)
            if gap_after < SAFE_FOLLOW_GAP:
                # Push ourselves back so gap == SAFE_FOLLOW_GAP
                push = SAFE_FOLLOW_GAP - gap_after
                self.x -= self.dx * push
                self.y -= self.dy * push
                self.current_speed = 0.0

        # ── Junction tracking ─────────────────────────────────────────
        if not self.in_junction and dist <= 0 and green:
            self.in_junction = True
        if self.in_junction:
            d2c = math.hypot(self.x - CENTER_X, self.y - CENTER_Y)
            if d2c > ROAD_WIDTH * 0.85:
                self.in_junction = False
                self.passed      = True

    def _gap_to(self, leader: "Vehicle") -> float:
        """Bumper-to-bumper gap to the vehicle ahead (positive = space, negative = overlap)."""
        if self.direction in ("north", "south"):
            # tail = rear edge of leader = centre minus half-length in travel direction
            tail_y = leader.y - leader.dy * (leader.body_len / 2)
            return (tail_y - self.nose_y) * self.dy
        else:
            tail_x = leader.x - leader.dx * (leader.body_len / 2)
            return (tail_x - self.nose_x) * self.dx

    # ── Draw ────────────────────────────────────────────

    def draw(self, surface: pygame.Surface):
        if self.is_diverted:
            # _divert_angle is math-convention (0=right, π/2=up)
            # canonical surface faces UP (north).
            # pygame rotate is CCW. We want: heading right → rotate -90°, up → 0°, etc.
            canonical = _make_canonical(self.vtype, self.flash_on)
            rot_deg = math.degrees(self._divert_angle) - 90.0
            rotated = pygame.transform.rotate(canonical, rot_deg)
            rect = rotated.get_rect(center=(int(self.x), int(self.y)))
            surface.blit(rotated, rect)
        else:
            vsurface = _get_surface(self.vtype, self.direction, self.flash_on)
            rect     = vsurface.get_rect(center=(int(self.x), int(self.y)))
            surface.blit(vsurface, rect)


# ─────────────────────────────────────────────────────────
#  Spawn helpers
# ─────────────────────────────────────────────────────────

# Maximum ambulances allowed on screen during normal spawning.
# User-triggered spawns (A / M keys) bypass this limit.
MAX_AMBULANCES_NORMAL = 2


def pick_vehicle_type(existing: "list[Vehicle] | None" = None) -> str:
    """
    Choose a random vehicle type.
    If 'existing' is provided, caps ambulances at MAX_AMBULANCES_NORMAL.
    """
    if random.random() < AMBULANCE_CHANCE:
        # Check ambulance cap (only when existing list is provided)
        if existing is not None:
            ambu_count = sum(1 for v in existing
                            if v.vtype == "ambulance" and not v.passed)
            if ambu_count >= MAX_AMBULANCES_NORMAL:
                pass   # skip — fall through to normal vehicle pick
            else:
                return "ambulance"
        else:
            return "ambulance"
    r = random.random()
    acc = 0.0
    for vt, w in VEHICLE_WEIGHTS.items():
        acc += w
        if r < acc:
            return vt
    return "car"


def try_spawn(direction: str, existing: "list[Vehicle]") -> "Vehicle | None":
    spawn_x, spawn_y = _incoming_lane_centre(direction)
    for v in existing:
        if v.direction == direction and not v.passed:
            d = math.hypot(v.x - spawn_x, v.y - spawn_y)
            if d < MIN_SPAWN_GAP + v.body_len:
                return None
    return Vehicle(pick_vehicle_type(existing), direction)
