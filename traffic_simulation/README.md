# 🚦 AI Smart Traffic Signal – Chowk Simulation

A **real-time 4-way city intersection simulation** with AI-controlled traffic signals,
vehicle physics, and ambulance emergency pre-emption.  Built for final-year engineering
project presentations.

---

## 📦 Requirements

```
Python 3.10+
pygame  2.x
numpy   (any recent version)
```

Install with:

```bash
pip install pygame numpy
```

---

## ▶️  How to Run

```bash
cd traffic_simulation
python main.py
```

| Key       | Action                              |
|-----------|-------------------------------------|
| `ESC / Q` | Quit the simulation                 |
| `A`       | Manually spawn an ambulance (demo)  |

---

## 📁 File Structure

```
traffic_simulation/
│
├── main.py        ← Entry point – game loop, orchestration
├── config.py      ← ALL settings (edit here!)
├── vehicles.py    ← Vehicle class, spawning helpers
├── signals.py     ← AI Traffic Controller (green-time AI logic)
└── renderer.py    ← Drawing: roads, buildings, signals, HUD
```

---

## ⚙️  Customisation (config.py)

| Setting             | Default | Meaning                                    |
|---------------------|---------|--------------------------------------------|
| `MAX_GREEN_TIME`    | 30 s    | Maximum green signal cap                   |
| `MIN_GREEN_TIME`    | 5 s     | Minimum green – never too short            |
| `YELLOW_TIME`       | 3 s     | Duration of yellow phase                   |
| `SPAWN_INTERVAL`    | 1.8 s   | Seconds between spawn attempts per lane    |
| `SPAWN_CHANCE`      | 0.85    | Probability a vehicle spawns per interval  |
| `AMBULANCE_CHANCE`  | 0.04    | Probability a vehicle is an ambulance      |
| `AMBULANCE_GREEN_HOLD` | 12 s | Green held for ambulance clearance        |
| `FPS`               | 60      | Target frame rate                          |
| `VEHICLE_TYPES`     | —       | Each type: speed, size, weight, colour     |

---

## 🤖 AI Signal Logic

### Green-Time Calculation
```
green_time = Σ weight(vehicle_i)  +  0.5 × avg(dist_i / speed_i)
```
Capped: `MIN_GREEN_TIME ≤ green_time ≤ MAX_GREEN_TIME`

| Vehicle  | Weight (sec) | Speed (px/s) |
|----------|-------------|--------------|
| Bike     | 1           | 120          |
| Car      | 2           | 90           |
| Bus      | 3           | 60           |
| Truck    | 4           | 50           |
| Ambulance| bypass      | 150          |

### Fairness (Anti-Starvation)
- Tracks last 2 served directions using a deque.
- A recently served direction is de-prioritised when other sides are waiting.
- Falls back to all-eligible if everyone was recently served.

### Ambulance Priority
1. Ambulance detected within 200 px of stop-line → immediate green.
2. All other signals forced RED.
3. Green held for `AMBULANCE_GREEN_HOLD` seconds (default 12 s).
4. Auto-resumes normal cycle once ambulance has cleared.

---

## 🎨 Visual Features

- Semi-3D top-down city view with procedurally generated buildings
- Lit window patterns on buildings
- Dashed lane markings, curbs, zebra crossings
- Traffic signal poles with red/yellow/green glow halos
- Vehicle headlights, ambulance cross + flashing light bar
- Live HUD dashboard with density bars and countdown timer
- Flashing red emergency banner during ambulance mode
- Compass rose (top-right)
- Pseudo-depth sorting (vehicles drawn by Y for depth)
