"""
=============================================================
  dashboard_server.py  –  Flask Cloud Monitoring Dashboard
=============================================================
  Runs on 0.0.0.0 (accessible from any device on the same
  WiFi network) at port DASHBOARD_PORT (default 5000).

  Routes:
    GET /           – serves dashboard.html
    GET /stream     – Server-Sent Events (live JSON every 500 ms)
    GET /api/stats  – one-shot REST snapshot

  Thread safety:
    _stats and _ambu_log are written by the pygame/simulation
    thread and read by the Flask request threads.
    A single threading.Lock protects both.
=============================================================
"""

import json
import time
import socket
import threading
import logging

from flask import Flask, Response, render_template
from config import DASHBOARD_PORT

# Silence Flask request logs to keep terminal clean
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────
#  Shared state (simulation thread → Flask threads)
# ─────────────────────────────────────────────────────────

_lock:      threading.Lock = threading.Lock()
_stats:     dict           = {}
_ambu_log:  list           = []   # [{time, event}, …]
_MAX_LOG  = 40


def update_stats(new_stats: dict):
    """Called from the simulation thread every ~500 ms."""
    global _stats
    with _lock:
        _stats = new_stats


def log_ambulance_event(event: str):
    """Record a timestamped ambulance event (called from sim thread)."""
    with _lock:
        ts = time.strftime("%H:%M:%S")
        _ambu_log.append({"time": ts, "event": event})
        if len(_ambu_log) > _MAX_LOG:
            _ambu_log.pop(0)


def _snapshot() -> dict:
    with _lock:
        snap = dict(_stats)
        snap["ambulance_log"] = list(_ambu_log)
    return snap


# ─────────────────────────────────────────────────────────
#  Flask application
# ─────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — pushes JSON every 500 ms."""
    def _generate():
        while True:
            data = json.dumps(_snapshot(), default=str)
            yield f"data: {data}\n\n"
            time.sleep(0.5)

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@app.route("/api/stats")
def api_stats():
    """One-shot REST endpoint (useful for testing / external tools)."""
    data = json.dumps(_snapshot(), default=str)
    return data, 200, {"Content-Type": "application/json"}


# ─────────────────────────────────────────────────────────
#  Startup helper
# ─────────────────────────────────────────────────────────

def get_local_ip() -> str:
    """Return the LAN IP so users know the WiFi URL."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "localhost"


def start_dashboard(port: int = DASHBOARD_PORT, host: str = "0.0.0.0") -> threading.Thread:
    """
    Launch Flask in a background daemon thread.
    Returns the thread object (usually not needed).
    """
    ip = get_local_ip()
    print("=" * 62)
    print("  📊 Cloud Dashboard started!")
    print(f"     Local  → http://localhost:{port}")
    print(f"     WiFi   → http://{ip}:{port}   (same network)")
    print("=" * 62)

    t = threading.Thread(
        target=lambda: app.run(
            host=host,
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True,
        ),
        daemon=True,
        name="DashboardServer",
    )
    t.start()
    return t
