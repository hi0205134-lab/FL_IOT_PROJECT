import pickle, numpy as np, json, os, sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.model import SimpleNN
for f in ['models/global_model.pkl', 'models/scaler.pkl']:
    if not os.path.exists(f):
        print(f'[ERROR] {f} not found. Run training first!')
        sys.exit(1)
 
global_weights = pickle.load(open('models/global_model.pkl', 'rb'))
scaler         = pickle.load(open('models/scaler.pkl', 'rb'))
node_scalers   = scaler['nodes']   # per-node — fixes KeyError crash
 
model = SimpleNN()                  # create model first
model.set_weights(global_weights)
THRESHOLD = 0.5
 
print('[OK] Global FL model loaded')
for node, ns in node_scalers.items():
    print(f'  {node}: min={ns["min"]:.2f}  max={ns["max"]:.2f}')
print('-' * 55)
 
def detect(node_id, raw_distance):
    ns     = node_scalers.get(node_id, scaler['global'])
    scaled = np.clip((raw_distance - ns['min']) / (ns['max'] - ns['min']), 0.0, 1.0)
    X      = np.array([[scaled]], dtype=np.float32)
    prob   = float(model.predict_proba(X)[0])
    is_a   = prob < THRESHOLD
    score  = round(1.0 - prob, 4)
    alert  = {
        'ClientID':      node_id,
        'Source_IP':     f'192.168.1.{10 + int(node_id[-1])}',
        'Anomaly_Score': score,
        'Is_Anomaly':    bool(is_a),
        'Raw_Distance':  round(raw_distance, 2),
        'Timestamp':     datetime.now().isoformat()
    }
    tag = 'ANOMALY' if is_a else 'OK'
    print(f'[{tag}] {node_id} | dist={raw_distance:.2f}cm | score={score:.4f}')
    if is_a:
        print(f'  Alert: {json.dumps(alert)}')
    return alert
 
tests = [
    ('node1', 25.0), ('node1', 32.0),
    ('node2',  8.0), ('node2', 12.0),
    ('node3', 45.0), ('node3', 60.0),
    ('node1',  4.0), ('node1',250.0),
    ('node2',180.0), ('node3',300.0),
    ('node1',200.0), ('node2',200.0), ('node3',200.0),
]
alerts    = [detect(n, d) for n, d in tests]
anomalies = [a for a in alerts if a['Is_Anomaly']]
triggered = set(a['ClientID'] for a in anomalies)
print(f'\nTotal anomalies: {len(anomalies)}')
if len(triggered) >= 2:
    print(f'COORDINATED ATTACK DETECTED — Nodes: {triggered}')
