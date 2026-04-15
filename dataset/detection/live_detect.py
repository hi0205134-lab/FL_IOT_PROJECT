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
 
# Settings
SERIAL_PORT  = 'COM5'
BAUD_RATE    = 115200
THRESHOLD    = 0.5
RAILWAY_URL  = 'https://fliotproject-production.up.railway.app'
 
# ── TIME-BASED ACCESS CONTROL ─────────────────────────────────
# 8am to 8pm : ALL readings are NORMAL regardless of distance
# Before 8am and after 8pm : FL model checks anomaly
# Change WORK_START_HOUR and WORK_END_HOUR freely for demo.
# No retraining needed after changing these two values.
WORK_START_HOUR = 8    # 0-23
WORK_END_HOUR   = 20   # 0-23  (20 = 8:00 PM)
# ──────────────────────────────────────────────────────────────
 
def is_working_hours():
    h = datetime.now().hour
    return WORK_START_HOUR <= h < WORK_END_HOUR
 
print('=' * 55)
print('  FL IoT IDS — Live Real-Time Detection')
print(f'  Normal hours: {WORK_START_HOUR:02d}:00 to {WORK_END_HOUR:02d}:00')
print('=' * 55)
 
# Check Railway
try:
    r = requests.get(f'{RAILWAY_URL}/status', timeout=5)
    print(f'[OK] Railway connected: {RAILWAY_URL}')
except Exception:
    print('[WARN] Railway not reachable — results stay in terminal only')
 
# Load model
for f in ['models/global_model.pkl', 'models/scaler.pkl']:
    if not os.path.exists(f):
        print(f'[ERROR] {f} not found!')
        sys.exit(1)
 
global_weights = pickle.load(open('models/global_model.pkl', 'rb'))
scaler         = pickle.load(open('models/scaler.pkl', 'rb'))
node_scalers   = scaler['nodes']
model          = SimpleNN()
model.set_weights(global_weights)
print('[OK] Global FL model loaded')
for node, ns in node_scalers.items():
    print(f'  {node}: min={ns["min"]:.2f}cm  max={ns["max"]:.2f}cm')
 
# Connect to Gateway
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
    time.sleep(2)
    print(f'[OK] Connected to {SERIAL_PORT}')
except serial.SerialException as e:
    print(f'[ERROR] Cannot open {SERIAL_PORT}: {e}')
    print('Fix: Close Arduino Serial Monitor first')
    sys.exit(1)
 
node_status    = {'node1': 'NORMAL', 'node2': 'NORMAL', 'node3': 'NORMAL'}
anomaly_counts = {'node1': 0, 'node2': 0, 'node3': 0}
recent_alerts  = []
total_count    = 0
anomaly_count  = 0
 
print('\n[LIVE] Reading from sensors... (Ctrl+C to stop)')
print('-' * 60)
 
try:
    while True:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode('utf-8', errors='ignore').strip()
        if not line or ',' not in line:
            continue
        if line.startswith('[') or line.startswith('Gateway'):
            continue
        parts = line.split(',')
        if len(parts) != 2:
            continue
        node_id = parts[0].strip()
        if node_id not in ['node1', 'node2', 'node3']:
            continue
        try:
            distance = float(parts[1].strip())
        except ValueError:
            continue
        if distance < 2.0 or distance > 400.0:
            continue
 
        now       = datetime.now()
        timestamp = now.strftime('%H:%M:%S')
        total_count += 1
 
        # ── TIME CHECK: 8am-8pm = always Normal ───────────────
        if is_working_hours():
            label      = 'NORMAL'
            score      = 0.0
            is_anomaly = False
            reason     = f'Working hours ({WORK_START_HOUR}:00-{WORK_END_HOUR}:00)'
        else:
            # ── FL MODEL CHECK ─────────────────────────────────
            ns         = node_scalers.get(node_id, scaler['global'])
            scaled     = float(np.clip((distance - ns['min']) / (ns['max'] - ns['min']), 0.0, 1.0))
            X          = np.array([[scaled]], dtype=np.float32)
            prob       = float(model.predict_proba(X)[0])
            label      = 'NORMAL' if prob > THRESHOLD else 'ANOMALY'
            is_anomaly = (label == 'ANOMALY')
            score      = round(1.0 - prob, 4)
            reason     = 'FL model'
 
        node_status[node_id] = label
        if is_anomaly:
            anomaly_count += 1
            anomaly_counts[node_id] += 1
 
        tag = '[ANOMALY]' if is_anomaly else '[OK]     '
        print(f'{tag} {node_id:<8} {distance:>8.2f} cm  score={score:.4f}  {timestamp}  {reason}')
        n1 = node_status['node1']
        n2 = node_status['node2']
        n3 = node_status['node3']
        print(f'  ALL NODES → node1:{n1}  node2:{n2}  node3:{n3}')
 
        if is_anomaly:
            alert_payload = {
                'ClientID':      node_id,
                'Source_IP':     f'192.168.1.{10 + int(node_id[-1])}',
                'Anomaly_Score': score,
                'Is_Anomaly':    True,
                'Raw_Distance':  round(distance, 2),
                'Timestamp':     now.isoformat()
            }
            print(f'  Alert JSON: {json.dumps(alert_payload)}')
            recent_alerts.append({'node': node_id, 'time': now})
 
        recent_alerts   = [a for a in recent_alerts if (now - a['time']).seconds <= 5]
        triggered_nodes = set(a['node'] for a in recent_alerts)
        coordinated     = len(triggered_nodes) >= 2
        if coordinated:
            print()
            print('  *** COORDINATED ATTACK DETECTED! ***')
            print(f'  Nodes: {sorted(triggered_nodes)}')
            print(f'  Time : {timestamp}')
            print()
            recent_alerts = []
 
        try:
            requests.post(f'{RAILWAY_URL}/alert', json={
                'timestamp':   now.isoformat(),
                'node':        node_id,
                'distance':    distance,
                'score':       score,
                'label':       label,
                'coordinated': coordinated,
                'all_nodes':   node_status.copy()
            }, timeout=3)
        except:
            pass
 
        if total_count % 10 == 0:
            print(f'  --- Stats: {total_count} total | {anomaly_count} anomalies | {total_count - anomaly_count} normal ---')
 
except KeyboardInterrupt:
    print(f'\n[STOPPED]')
    print(f'Total readings : {total_count}')
    print(f'Anomalies      : {anomaly_count}')
    print(f'Normal         : {total_count - anomaly_count}')
    print(f'Per node       : {anomaly_counts}')
finally:
    ser.close()
    print('Serial port closed.')
