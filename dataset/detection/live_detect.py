import serial, pickle, numpy as np, requests, os, sys, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.model import SimpleNN

SERIAL_PORT = 'COM5'
BAUD_RATE   = 115200
THRESHOLD   = 0.5
FLASK_URL   = 'https://fliotproject-production.up.railway.app'

for f in ['models/global_model.pkl', 'models/scaler.pkl']:
    if not os.path.exists(f):
        print(f'[ERROR] {f} not found!'); sys.exit(1)

global_weights = pickle.load(open('models/global_model.pkl', 'rb'))
scaler         = pickle.load(open('models/scaler.pkl', 'rb'))
node_scalers   = scaler['nodes']

model = SimpleNN()
model.set_weights(global_weights)
print('[OK] FL model loaded')

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
    time.sleep(2)
    print(f'[OK] Connected to {SERIAL_PORT}')
except Exception as e:
    print(f'[ERROR] {e}'); sys.exit(1)

# Track status of ALL nodes
node_status    = {'node1': 'NORMAL', 'node2': 'NORMAL', 'node3': 'NORMAL'}
anomaly_counts = {'node1': 0, 'node2': 0, 'node3': 0}

print('Live detection started...')
print('-' * 60)

try:
    while True:
        raw = ser.readline()
        if not raw: continue
        line = raw.decode('utf-8', errors='ignore').strip()
        if not line or ',' not in line: continue
        if line.startswith('[') or line.startswith('Gateway'): continue
        parts = line.split(',')
        if len(parts) != 2: continue
        node_id = parts[0].strip()
        if node_id not in ['node1', 'node2', 'node3']: continue
        try: distance = float(parts[1].strip())
        except: continue
        if distance < 2.0 or distance > 400.0: continue

        # Per-node normalization
        ns     = node_scalers.get(node_id, scaler['global'])
        scaled = float(np.clip(
            (distance - ns['min']) / (ns['max'] - ns['min']), 0.0, 1.0))
        X    = np.array([[scaled]], dtype=np.float32)
        prob = float(model.predict_proba(X)[0])
        label = 'ANOMALY' if prob > THRESHOLD else 'NORMAL'

        # Update this node's status
        node_status[node_id] = label
        if label == 'ANOMALY':
            anomaly_counts[node_id] += 1

        ts = datetime.now().strftime('%H:%M:%S')

        # Check coordinated attack
        anomaly_nodes = [n for n, s in node_status.items() if s == 'ANOMALY']
        coordinated   = len(anomaly_nodes) >= 2

        # Print result
        tag = '[ANOMALY]' if label == 'ANOMALY' else '[NORMAL ]'
        print(f'{tag} {node_id} | {distance:.2f}cm | score={prob:.4f} | {ts}')

        # Always show all node statuses so every node knows others
        n1 = node_status['node1']
        n2 = node_status['node2']
        n3 = node_status['node3']
        print(f'  ALL NODES → node1:{n1}  node2:{n2}  node3:{n3}')

        if coordinated:
            print(f'  *** COORDINATED ATTACK! Nodes: {", ".join(anomaly_nodes)} ***')

        # Send to Railway cloud
        try:
            payload = {
                'timestamp':   ts,
                'node':        node_id,
                'distance':    distance,
                'score':       prob,
                'label':       label,
                'coordinated': coordinated,
                'all_nodes':   node_status.copy()
            }
            requests.post(f'{FLASK_URL}/alert', json=payload, timeout=3)
        except:
            pass

except KeyboardInterrupt:
    print(f'\nStopped.')
    print(f'Anomaly counts: {anomaly_counts}')
finally:
    ser.close()