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
    print("       Detection will still work locally")

# ── Load trained global model ─────────────────────────
for f in ["models/global_model.pkl", "models/scaler.pkl"]:
    if not os.path.exists(f):
        print(f"[ERROR] {f} not found!")
        print("Run: python server/run_federated_training.py first")
        sys.exit(1)

global_weights = pickle.load(open("models/global_model.pkl", "rb"))
scaler         = pickle.load(open("models/scaler.pkl", "rb"))
d_min = scaler["min"]
d_max = scaler["max"]

model = SimpleNN()
model.set_weights(global_weights)
print(f"[OK] Global FL model loaded")
print(f"[OK] Scaler: min={d_min:.2f}  max={d_max:.2f}")

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

# ── Send result to Railway cloud ──────────────────────
def send_to_cloud(node_id, distance, score, is_anomaly, timestamp):
    payload = {
        "node":          node_id,
        "distance":      round(distance, 2),
        "anomaly_score": score,
        "is_anomaly":    bool(is_anomaly),
        "timestamp":     timestamp
    }
    try:
        requests.post(f"{RAILWAY_URL}/alert", json=payload, timeout=5)
    except Exception:
        pass  # never crash if cloud unreachable

print("\n[LIVE] Reading from sensors... (Ctrl+C to stop)")
print("-" * 60)
print(f"{'Status':<10} {'Node':<8} {'Distance':>10} {'Score':>8}  {'Time'}")
print("-" * 60)

# ── Track coordinated attack ──────────────────────────
recent_alerts = []
total_count   = 0
anomaly_count = 0

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

        # ── Run FL model detection ─────────────────────
        scaled        = np.clip((distance - d_min) / (d_max - d_min), 0.0, 1.0)
        X             = np.array([[scaled]], dtype=np.float32)
        prob_normal   = float(model.predict_proba(X)[0])
        anomaly_score = round(1.0 - prob_normal, 4)
        is_anomaly    = anomaly_score > THRESHOLD
        now           = datetime.now()
        timestamp     = now.isoformat()

        total_count += 1
        if is_anomaly:
            anomaly_count += 1

        # ── Print to terminal ──────────────────────────
        status = "ANOMALY" if is_anomaly else "Normal"
        tag    = "[ANOMALY]" if is_anomaly else "[OK]     "
        print(f"{tag} {node_id:<8} {distance:>8.2f} cm  {anomaly_score:>7.4f}  {now.strftime('%H:%M:%S')}")

        # ── Send ALL readings to Railway ───────────────
        send_to_cloud(node_id, distance, anomaly_score, is_anomaly, timestamp)

        # ── Print alert JSON if anomaly ────────────────
        if is_anomaly:
            alert = {
                "ClientID":      node_id,
                "Source_IP":     f"192.168.1.{10 + int(node_id[-1])}",
                "Anomaly_Score": anomaly_score,
                "Is_Anomaly":    True,
                "Raw_Distance":  round(distance, 2),
                "Timestamp":     timestamp
            }
            print(f"   Alert JSON: {json.dumps(alert)}")
            recent_alerts.append({"node": node_id, "time": now})

        # ── Coordinated attack check ───────────────────
        recent_alerts = [
            a for a in recent_alerts
            if (now - a["time"]).seconds <= 5
        ]
        triggered_nodes = set(a["node"] for a in recent_alerts)
        if len(triggered_nodes) >= 2:
            print()
            print("=" * 55)
            print("COORDINATED ATTACK DETECTED!")
            print(f"   Nodes: {', '.join(sorted(triggered_nodes))}")
            print(f"   Time : {now.strftime('%H:%M:%S')}")
            print(f"   Sent to Railway dashboard automatically")
            print("=" * 55)
            print()
            recent_alerts = []

        # ── Print running stats every 10 readings ─────
        if total_count % 10 == 0:
            print(f"   --- Stats: {total_count} total | {anomaly_count} anomalies | {total_count-anomaly_count} normal ---")

except KeyboardInterrupt:
    print(f"\n[STOPPED]")
    print(f"Total readings : {total_count}")
    print(f"Anomalies      : {anomaly_count}")
    print(f"Normal         : {total_count - anomaly_count}")

finally:
    ser.close()
    print("Serial port closed.")