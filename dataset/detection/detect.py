import pickle, numpy as np, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.model import SimpleNN

for f in ['models/global_model.pkl', 'models/scaler.pkl']:
    if not os.path.exists(f):
        print(f'[ERROR] {f} not found. Run training first!'); sys.exit(1)

global_weights = pickle.load(open('models/global_model.pkl', 'rb'))
scaler         = pickle.load(open('models/scaler.pkl', 'rb'))
node_scalers   = scaler['nodes']

model = SimpleNN()
model.set_weights(global_weights)
THRESHOLD = 0.5

print('[OK] Global FL model loaded')
print('Running detection tests...')
print('-' * 60)

def detect(node_id, raw_distance):
    ns     = node_scalers.get(node_id, scaler['global'])
    scaled = float(np.clip(
        (raw_distance - ns['min']) / (ns['max'] - ns['min']), 0.0, 1.0))
    X    = np.array([[scaled]], dtype=np.float32)
    prob = float(model.predict_proba(X)[0])
    label = 'ANOMALY' if prob > THRESHOLD else 'NORMAL'
    print(f'{node_id} | {raw_distance:7.2f}cm | {label:7s} | score={prob:.4f}')
    return label, prob

print('\n--- node1 tests (normal: 15-40cm) ---')
detect('node1', 25.0)
detect('node1', 10.0)
detect('node1', 200.0)

print('\n--- node2 tests (normal: 2-20cm) ---')
detect('node2', 10.0)
detect('node2', 29.0)
detect('node2', 35.0)

print('\n--- node3 tests (normal: 15-75cm) ---')
detect('node3', 50.0)
detect('node3', 5.0)
detect('node3', 120.0)