"""
=============================================================
  signals.py  –  AI Traffic Signal Controller
=============================================================
  Multi-ambulance queue:
  - Every approaching ambulance (from ANY direction) is queued.
  - They are served FIFO: current → next → next …
  - If more ambulances queue up while one is being served,
    they wait and are served immediately after.
  - No starvation: at most 4 ambulances can ever be queued.
=============================================================
"""
import time
from collections import deque
from config import (
    DIRECTIONS, MIN_GREEN_TIME, MAX_GREEN_TIME, YELLOW_TIME,
    AMBULANCE_GREEN_HOLD, AMBULANCE_EXTEND_HOLD, AMBULANCE_DETECT_DIST,
    MIN_RED_WAIT, VEHICLE_TYPES,
)


class SignalState:
    RED    = "red"
    YELLOW = "yellow"
    GREEN  = "green"


class TrafficController:
    """
    AI Traffic Signal Controller with multi-ambulance priority queue.

    Ambulance Queue Logic:
      1. Every frame, scan all vehicles for ambulances within detection range.
      2. If an ambulance from direction D is detected and D is not yet in
         _ambu_queue → append D (one entry per unique direction at a time).
      3. If no emergency is currently active → immediately activate the first
         entry in the queue.
      4. When the active ambulance clears its direction:
           a. Remove that direction from the queue.
           b. If more entries remain → activate the next one immediately
              (skip yellow/red-all transition for speed).
           c. If queue is empty → transition yellow → resume normal cycle.
      5. If the same direction still has more ambulances waiting after the
         first one clears → extend the hold by AMBULANCE_EXTEND_HOLD seconds
         instead of immediately ending emergency on that side.
    """

    def __init__(self):
        # Per-direction signal face
        self.signal: dict[str, str] = {d: SignalState.RED for d in DIRECTIONS}

        # Current active green
        self.green_dir:   str | None = None
        self.phase:       str        = "red_all"
        self.phase_timer: float      = 2.0
        self.green_time:  float      = 0.0

        # ── Multi-ambulance queue ──────────────────────────────────
        # Ordered list of directions awaiting ambulance green.
        # Each direction appears at most once in this list.
        self._ambu_queue:  list[str]        = []
        self.emergency_active: bool         = False
        self.emergency_dir:    str | None   = None
        self.emergency_timer:  float        = 0.0

        # Fairness / anti-starvation
        self._served_queue: deque[str] = deque(maxlen=len(DIRECTIONS))
        self._last_red_ts: dict[str, float] = {d: 0.0 for d in DIRECTIONS}

        # Stats / UI
        self.total_passed:  int = 0
        self.current_mode:  str = "Normal"

    # ── Public API ───────────────────────────────────────────────

    def signal_color(self, direction: str) -> str:
        return self.signal[direction]

    def is_green(self, direction: str) -> bool:
        return self.signal[direction] == SignalState.GREEN

    def countdown(self) -> float:
        return max(0.0, self.phase_timer)

    @property
    def pending_ambulance_dirs(self) -> list[str]:
        """All directions currently in the ambulance queue (including active)."""
        return list(self._ambu_queue)

    @property
    def queued_count(self) -> int:
        """Number of directions waiting for ambulance green."""
        return len(self._ambu_queue)

    # ── Main update ──────────────────────────────────────────────

    def update(self, dt: float, queues: "dict[str, list]"):
        """
        dt     – seconds since last frame
        queues – {direction: [Vehicle …]} active (non-passed) vehicles
        """
        # ── 1. Scan for ambulances, update queue ──────────────────
        self._scan_ambulances(queues)

        # ── 2. Handle active emergency timer ─────────────────────
        if self.emergency_active:
            self.emergency_timer -= dt
            if self.emergency_timer <= 0:
                self._on_emergency_timer_expired(queues)
            return   # Do NOT run normal phase logic while emergency is active

        # ── 3. If queue has entries but no active emergency → start ──
        if self._ambu_queue and not self.emergency_active:
            self._activate_emergency(self._ambu_queue[0])
            return

        # ── 4. Normal signal phase transitions ────────────────────
        self.phase_timer -= dt
        if self.phase_timer <= 0:
            if self.phase == "green":
                self._start_yellow()
            elif self.phase in ("yellow", "red_all"):
                self._start_red_and_elect(queues)

    # ── Ambulance queue engine ───────────────────────────────────

    def _scan_ambulances(self, queues: dict):
        """Detect approaching ambulances and add directions to queue."""
        for d, vehicles in queues.items():
            for v in vehicles:
                if (v.vtype == "ambulance"
                        and not v.in_junction
                        and not v.passed
                        and v.dist_to_stop() < AMBULANCE_DETECT_DIST
                        and d not in self._ambu_queue):
                    self._ambu_queue.append(d)

        # Remove directions where ALL ambulances have passed / cleared
        cleared = []
        for d in self._ambu_queue:
            has_waiting = any(
                v.vtype == "ambulance" and not v.passed and not v.in_junction
                for v in queues.get(d, [])
            )
            if not has_waiting:
                cleared.append(d)
        for d in cleared:
            self._ambu_queue.remove(d)
            if self.emergency_dir == d and not cleared:
                # Will handle in _on_emergency_timer_expired
                pass

        # Update mode label
        if self._ambu_queue or self.emergency_active:
            self.current_mode = "EMERGENCY"
        else:
            self.current_mode = "Normal"

    def _on_emergency_timer_expired(self, queues: dict):
        """
        Called when the green hold timer runs out for the current ambulance.
        Checks if more ambulances are still on this side; if so, extends hold.
        Otherwise moves to next queued direction or resumes normal cycle.
        """
        cur = self.emergency_dir

        # Count remaining ambulances on the current-emergency side
        remaining = [
            v for v in queues.get(cur, [])
            if v.vtype == "ambulance" and not v.passed and not v.in_junction
        ]
        if remaining:
            # More ambulances still waiting on the same side – extend hold
            self.emergency_timer = AMBULANCE_EXTEND_HOLD
            return

        # Current side is cleared
        if cur in self._ambu_queue:
            self._ambu_queue.remove(cur)

        # Is there a next queued direction?
        if self._ambu_queue:
            # Serve next ambulance immediately (no yellow for speed)
            next_dir = self._ambu_queue[0]
            self._activate_emergency(next_dir)
        else:
            # No more ambulances – end emergency, transition to yellow
            self.emergency_active = False
            self.emergency_dir    = None
            self.current_mode     = "Normal"
            self._start_yellow()

    def _activate_emergency(self, direction: str):
        """Force <direction> to GREEN immediately (ambulance override)."""
        for d in DIRECTIONS:
            self.signal[d] = SignalState.RED
        self.signal[direction]  = SignalState.GREEN
        self.green_dir          = direction
        self.phase              = "green"
        self.phase_timer        = AMBULANCE_GREEN_HOLD
        self.green_time         = AMBULANCE_GREEN_HOLD
        self.emergency_active   = True
        self.emergency_dir      = direction
        self.emergency_timer    = AMBULANCE_GREEN_HOLD
        self.current_mode       = "EMERGENCY"

    # ── Normal signal cycle ──────────────────────────────────────

    def _activate_green(self, direction: str, duration: float):
        """Standard green activation (non-emergency)."""
        for d in DIRECTIONS:
            self.signal[d] = SignalState.RED
        self.signal[direction] = SignalState.GREEN
        self.green_dir         = direction
        self.phase             = "green"
        self.phase_timer       = duration
        self.green_time        = duration
        self._served_queue.append(direction)
        self._last_red_ts.update(
            {d: time.monotonic() for d in DIRECTIONS if d != direction}
        )

    def _start_yellow(self):
        if self.green_dir:
            self.signal[self.green_dir] = SignalState.YELLOW
        self.phase       = "yellow"
        self.phase_timer = YELLOW_TIME

    def _start_red_and_elect(self, queues: dict):
        for d in DIRECTIONS:
            self.signal[d] = SignalState.RED
        if self.green_dir:
            self._last_red_ts[self.green_dir] = time.monotonic()
        self.green_dir = None

        next_dir, green_dur = self._elect(queues)
        if next_dir:
            self._activate_green(next_dir, green_dur)
        else:
            self.phase       = "red_all"
            self.phase_timer = 2.0

    def _elect(self, queues: dict) -> "tuple[str | None, float]":
        """
        Choose next green direction using weighted density scoring
        with fairness constraints.
        """
        now = time.monotonic()

        candidates = {
            d: [v for v in vs if not v.passed and not v.in_junction]
            for d, vs in queues.items()
        }
        waiting = {d for d, vs in candidates.items() if vs}
        if not waiting:
            return None, 0.0

        # Fairness: de-prioritise recently served sides
        if len(waiting) > 1:
            recent = set(list(self._served_queue)[-2:]) if len(self._served_queue) >= 2 else set()
            eligible = waiting - recent or waiting
        else:
            eligible = waiting

        # Minimum red wait
        truly_eligible = {
            d for d in eligible
            if (now - self._last_red_ts.get(d, 0.0)) >= MIN_RED_WAIT
               or d not in self._served_queue
        } or eligible

        scores = {
            d: sum(VEHICLE_TYPES[v.vtype]["weight"] for v in candidates[d])
               + (now - self._last_red_ts.get(d, now)) * 0.1
            for d in truly_eligible
        }
        best = max(scores, key=scores.__getitem__)
        return best, self._calc_green_time(candidates[best])

    def _calc_green_time(self, vehicles: list) -> float:
        if not vehicles:
            return MIN_GREEN_TIME
        base = sum(VEHICLE_TYPES[v.vtype]["weight"] for v in vehicles)
        dist_bonus = sum(
            max(0.0, v.dist_to_stop()) / max(v.speed, 1.0)
            for v in vehicles
        ) / len(vehicles) * 0.5
        return float(max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, base + dist_bonus)))
