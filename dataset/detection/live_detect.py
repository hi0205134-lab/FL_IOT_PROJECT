import serial
import pickle
import numpy as np
import json
import os
import sys
import time
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.model import SimpleNN

# ── Settings ──────────────────────────────────────────
SERIAL_PORT  = "COM5"
BAUD_RATE    = 115200
THRESHOLD    = 0.5
RAILWAY_URL  = "https://fliotproject-production.up.railway.app"

print("=" * 55)
print("  FL IoT IDS — Live Real-Time Detection")
print("=" * 55)

# ── Check Railway is reachable ────────────────────────
print("Checking Railway connection...")
try:
    r = requests.get(f"{RAILWAY_URL}/status", timeout=5)
    print(f"[OK] Railway connected: {RAILWAY_URL}")
except Exception:
    print("[WARN] Railway not reachable — results won't go to cloud dashboard")

# ── Load trained global model ─────────────────────────
for f in ["models/global_model.pkl", "models/scaler.pkl"]:
    if not os.path.exists(f):
        print(f"[ERROR] {f} not found!")
        sys.exit(1)

global_weights = pickle.load(open("models/global_model.pkl", "rb"))
scaler         = pickle.load(open("models/scaler.pkl", "rb"))

# ── NEW scaler structure — per node ───────────────────
node_scalers = scaler["nodes"]   # {'node1': {'min':..,'max':..}, ...}

print(f"[OK] Global FL model loaded")
print(f"[OK] Per-node scalers loaded:")
for node, ns in node_scalers.items():
    print(f"     {node}: min={ns['min']:.2f}cm  max={ns['max']:.2f}cm")

model = SimpleNN()
model.set_weights(global_weights)

# ── Connect to Gateway ESP32 ──────────────────────────
print(f"\nConnecting to Gateway on {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
    time.sleep(2)
    print(f"[OK] Connected to {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"[ERROR] Cannot open {SERIAL_PORT}: {e}")
    print("Fix: Close Arduino Serial Monitor first")
    sys.exit(1)

# ── Track status of ALL nodes ─────────────────────────
node_status    = {"node1": "NORMAL", "node2": "NORMAL", "node3": "NORMAL"}
anomaly_counts = {"node1": 0, "node2": 0, "node3": 0}
recent_alerts  = []
total_count    = 0
anomaly_count  = 0

print("\n[LIVE] Reading from sensors... (Ctrl+C to stop)")
print("-" * 60)
print(f"{'Status':<10} {'Node':<8} {'Distance':>10} {'Score':>8}  {'Time'}")
print("-" * 60)

try:
    while True:
        raw = ser.readline()
        if not raw:
            continue

        line = raw.decode("utf-8", errors="ignore").strip()
        if not line or "," not in line:
            continue
        if line.startswith("[") or line.startswith("Gateway"):
            continue

        parts = line.split(",")
        if len(parts) != 2:
            continue

        node_id = parts[0].strip()
        if node_id not in ["node1", "node2", "node3"]:
            continue

        try:
            distance = float(parts[1].strip())
        except ValueError:
            continue

        if distance < 2.0 or distance > 400.0:
            continue

        # ── Per-node normalization ─────────────────────
        ns     = node_scalers.get(node_id, scaler["global"])
        scaled = float(np.clip(
            (distance - ns["min"]) / (ns["max"] - ns["min"]), 0.0, 1.0))
        X    = np.array([[scaled]], dtype=np.float32)
        prob = float(model.predict_proba(X)[0])

        # ── CORRECT label logic ────────────────────────
        # label=1 in dataset = NORMAL, label=0 = ANOMALY
        # so prob > 0.5 means NORMAL
        label      = "NORMAL" if prob > THRESHOLD else "ANOMALY"
        is_anomaly = label == "ANOMALY"

        # ── Update node status ─────────────────────────
        node_status[node_id] = label
        if is_anomaly:
            anomaly_counts[node_id] += 1

        now       = datetime.now()
        timestamp = now.strftime("%H:%M:%S")
        total_count += 1
        if is_anomaly:
            anomaly_count += 1

        # ── Print to terminal ──────────────────────────
        tag = "[ANOMALY]" if is_anomaly else "[OK]     "
        print(f"{tag} {node_id:<8} {distance:>8.2f} cm  score={prob:.4f}  {timestamp}")

        # ── Show all node statuses ─────────────────────
        n1 = node_status["node1"]
        n2 = node_status["node2"]
        n3 = node_status["node3"]
        print(f"  ALL NODES → node1:{n1}  node2:{n2}  node3:{n3}")

        # ── Coordinated attack check ───────────────────
        if is_anomaly:
            recent_alerts.append({"node": node_id, "time": now})

        recent_alerts = [
            a for a in recent_alerts
            if (now - a["time"]).seconds <= 5
        ]
        triggered_nodes = set(a["node"] for a in recent_alerts)
        coordinated = len(triggered_nodes) >= 2

        if coordinated:
            print()
            print("=" * 55)
            print("  *** COORDINATED ATTACK DETECTED! ***")
            print(f"  Nodes: {', '.join(sorted(triggered_nodes))}")
            print(f"  Time : {timestamp}")
            print("=" * 55)
            print()

        # ── Send ALERT to Railway ──────────────────────
        try:
            alert_payload = {
                "timestamp":   timestamp,
                "node":        node_id,
                "distance":    distance,
                "score":       prob,
                "label":       label,
                "coordinated": coordinated,
                "all_nodes":   node_status.copy()
            }
            requests.post(f"{RAILWAY_URL}/alert", json=alert_payload, timeout=3)
        except:
            pass

        # ── Send LOG to Railway — for sample count ─────
        try:
            log_payload = {
                "node":     node_id,
                "distance": distance,
                "label":    0 if is_anomaly else 1
            }
            requests.post(f"{RAILWAY_URL}/log", json=log_payload, timeout=3)
        except:
            pass

        # ── Print stats every 10 readings ─────────────
        if total_count % 10 == 0:
            print(f"  --- Stats: {total_count} total | {anomaly_count} anomalies | {total_count - anomaly_count} normal ---")

except KeyboardInterrupt:
    print(f"\n[STOPPED]")
    print(f"Total readings : {total_count}")
    print(f"Anomalies      : {anomaly_count}")
    print(f"Normal         : {total_count - anomaly_count}")
    print(f"Anomaly counts per node: {anomaly_counts}")

finally:
    ser.close()
    print("Serial port closed.")