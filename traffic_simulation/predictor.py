"""
=============================================================
  predictor.py  –  Local AI Traffic Predictor
=============================================================
  Pure-numpy time-series prediction engine.
  No API keys, no internet connection required.

  Algorithm per direction:
    1. Collect queue-length samples every SAMPLE_INTERVAL seconds
       into a sliding window of HISTORY_WINDOW points.
    2. Compute Exponential Weighted Moving Average (EWMA) for
       real-time smoothing.
    3. Fit a linear trend (numpy.polyfit) over the last 30 samples.
    4. Project the trend forward for PREDICT_HORIZON seconds.
    5. Compute a confidence score from regression residuals.
    6. Track vehicle throughput (vehicles / minute).
=============================================================
"""
import time
import numpy as np
from collections import deque
from config import DIRECTIONS, PREDICT_HORIZON, HISTORY_WINDOW

SAMPLE_INTERVAL = 0.5   # seconds between data samples


# ─────────────────────────────────────────────────────────
#  Per-direction predictor
# ─────────────────────────────────────────────────────────

class _DirectionPredictor:
    """Sliding-window predictor for a single approach direction."""

    ALPHA = 0.25           # EWMA smoothing factor  (0 = very smooth, 1 = raw)
    TREND_POINTS = 30      # samples used for linear trend fitting

    def __init__(self, window: int):
        self._window  = window
        self._times   = deque(maxlen=window)   # sim-time stamps
        self._values  = deque(maxlen=window)   # raw queue counts
        self._ewma    = 0.0
        self._n       = 0

    # ── Public interface ────────────────────────────────

    def add(self, value: float, sim_t: float):
        self._times.append(sim_t)
        self._values.append(float(value))
        # Update EWMA
        if self._n == 0:
            self._ewma = float(value)
        else:
            self._ewma = self.ALPHA * float(value) + (1.0 - self.ALPHA) * self._ewma
        self._n += 1

    def forecast(self, horizons: list) -> dict:
        """Return forecast dict for all requested horizon seconds."""
        result = {
            "smooth":     round(self._ewma, 2),
            "confidence": 0.0,
            "trend":      0.0,
        }
        for h in horizons:
            result[f"forecast_{h}"] = round(max(0.0, self._ewma), 2)

        if self._n < 6:          # not enough data yet
            return result

        vals  = np.array(self._values, dtype=np.float64)
        times = np.array(self._times,  dtype=np.float64)

        # Use at most TREND_POINTS most recent samples
        k   = min(self.TREND_POINTS, len(vals))
        v_k = vals[-k:]
        t_k = times[-k:] - times[-k]   # normalise to start at 0

        try:
            coeffs    = np.polyfit(t_k, v_k, 1)       # [slope, intercept]
            slope     = float(coeffs[0])

            predicted = np.polyval(coeffs, t_k)
            residuals = v_k - predicted
            std_res   = float(np.std(residuals))
            val_range = float(np.ptp(v_k)) if np.ptp(v_k) > 0 else 1.0
            confidence = max(0.05, min(0.99, 1.0 - std_res / val_range))

            result["trend"]      = round(slope, 4)
            result["confidence"] = round(confidence, 2)

            t_last = float(t_k[-1])
            for h in horizons:
                proj = float(np.polyval(coeffs, t_last + h))
                result[f"forecast_{h}"] = round(max(0.0, proj), 2)

        except np.linalg.LinAlgError:
            pass   # fall back to EWMA forecasts already set above

        return result

    def history(self, n: int) -> list:
        """Return the last n raw samples (padded with 0 if fewer)."""
        vals = list(self._values)
        if len(vals) >= n:
            return [round(v, 1) for v in vals[-n:]]
        pad = [0.0] * (n - len(vals))
        return pad + [round(v, 1) for v in vals]


# ─────────────────────────────────────────────────────────
#  Master predictor (all 4 directions)
# ─────────────────────────────────────────────────────────

class TrafficPredictor:
    """
    Master predictor that aggregates all four approach directions.

    Usage (called from Simulation._update every tick):
        predictor.update(dt, queues, total_passed)
        stats = predictor.get_stats()   # dict ready for dashboard
    """

    THROUGHPUT_WINDOW = 120   # throughput samples kept (~60 s)

    def __init__(self):
        self._dir_pred: dict[str, _DirectionPredictor] = {
            d: _DirectionPredictor(HISTORY_WINDOW) for d in DIRECTIONS
        }
        self._timer      = 0.0
        self._sim_time   = 0.0

        # Throughput tracking
        self._tput_buf   = deque(maxlen=self.THROUGHPUT_WINDOW)
        self._last_total = 0

        # Last snapshot for external query
        self._last_stats: dict = {}

    # ── Called every simulation tick ────────────────────

    def update(self, dt: float, queues: dict, total_passed: int):
        self._sim_time += dt
        self._timer    += dt

        # Throughput delta per SAMPLE_INTERVAL
        delta = max(0, total_passed - self._last_total)
        self._last_total = total_passed

        if self._timer >= SAMPLE_INTERVAL:
            self._timer = 0.0
            self._tput_buf.append(delta)

            for d in DIRECTIONS:
                q     = queues.get(d, [])
                count = float(sum(1 for v in q if not v.passed))
                self._dir_pred[d].add(count, self._sim_time)

    # ── Stats snapshot for dashboard ────────────────────

    def get_stats(self) -> dict:
        predictions = {}
        history     = {}
        for d in DIRECTIONS:
            predictions[d] = self._dir_pred[d].forecast(PREDICT_HORIZON)
            history[d]     = self._dir_pred[d].history(60)

        # Throughput: vehicles / minute
        total_delta  = sum(self._tput_buf)
        window_secs  = len(self._tput_buf) * SAMPLE_INTERVAL
        throughput   = (total_delta / max(window_secs, 1.0)) * 60.0

        # Overall prediction confidence (mean across directions)
        conf_vals = [predictions[d]["confidence"] for d in DIRECTIONS]
        avg_conf  = round(float(np.mean(conf_vals)), 2) if conf_vals else 0.0

        return {
            "predictions":  predictions,
            "throughput":   round(throughput, 1),
            "avg_conf":     avg_conf,
            "sim_time":     round(self._sim_time, 1),
            "history":      history,
        }
