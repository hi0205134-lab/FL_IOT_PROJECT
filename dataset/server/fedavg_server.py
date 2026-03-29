import numpy as np, pickle, os, sys

CLIENT_FILES = ['models/client1.pkl','models/client2.pkl','models/client3.pkl']
GLOBAL_FILE  = 'models/global_model.pkl'

print('=== FedAvg Aggregation ===')
clients = []
for f in CLIENT_FILES:
    if not os.path.exists(f):
        print(f'[ERROR] {f} not found'); sys.exit(1)
    d = pickle.load(open(f, 'rb'))
    clients.append(d)
    m = d['metrics']
    print(f"Client{d['client_id']}: {d['sample_size']} samples | Acc={m['accuracy']:.4f} F1={m['f1']:.4f}")

total = sum(c['sample_size'] for c in clients)
gw    = {k: np.zeros_like(v) for k,v in clients[0]['weights'].items()}
for c in clients:
    frac = c['sample_size'] / total
    for k in gw:
        gw[k] += frac * c['weights'][k]

os.makedirs('models', exist_ok=True)
pickle.dump(gw, open(GLOBAL_FILE, 'wb'))
print(f'[OK] Global model saved: {GLOBAL_FILE}')
