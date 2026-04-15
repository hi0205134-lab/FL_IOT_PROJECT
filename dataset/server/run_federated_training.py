import subprocess, pickle, numpy as np, pandas as pd, os, sys
from sklearn.metrics import accuracy_score, f1_score, recall_score

# ✅ Fix sys.path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)

ROUNDS   = 15
EPOCHS   = 100
LR       = '0.01'
LOG_FILE = 'models/fl_convergence_log.csv'

# ✅ Check datasets
for i in [1,2,3]:
    if not os.path.exists(f'dataset/client{i}.csv'):
        print(f'[ERROR] dataset/client{i}.csv missing!')
        sys.exit(1)

os.makedirs('models', exist_ok=True)

print(f'=== FEDERATED LEARNING — {ROUNDS} ROUNDS ===\n')


# ✅ Fixed evaluate_global()
def evaluate_global():
    from clients.model import SimpleNN   # safe import after path fix
    
    model_path = 'models/global_model.pkl'
    if not os.path.exists(model_path):
        return None

    # ✅ Create model before setting weights
    model = SimpleNN()

    # ✅ Safe file loading
    with open(model_path, 'rb') as f:
        weights = pickle.load(f)

    model.set_weights(weights)

    all_y, all_p = [], []

    for i in [1,2,3]:
        df = pd.read_csv(f'dataset/client{i}.csv')

        X = df['distance'].values.reshape(-1,1).astype(float)
        y = df['label'].values.astype(float)

        preds = model.predict(X)

        all_y.extend(y)
        all_p.extend(preds)

    return {
        'accuracy': accuracy_score(all_y, all_p),
        'f1': f1_score(all_y, all_p, zero_division=0),
        'recall': recall_score(all_y, all_p, zero_division=0)
    }


# ✅ Training Loop
log_rows = []

for rnd in range(1, ROUNDS+1):
    print(f'--- ROUND {rnd}/{ROUNDS} ---')

    # 🔁 Train clients
    for cid in [1,2,3]:
        print(f'  Client {cid}...', end=' ', flush=True)

        r = subprocess.run(
            [sys.executable, 'dataset/clients/train_client.py',
             '--client', str(cid),
             '--epochs', str(EPOCHS),
             '--lr', LR],
            capture_output=True,
            text=True
        )

        if r.returncode != 0:
            print(f'FAILED\n{r.stderr[-300:]}')
            sys.exit(1)

        print('done')

    # 🧠 FedAvg aggregation
    r = subprocess.run(
        [sys.executable, 'dataset/server/fedavg_server.py'],
        capture_output=True,
        text=True
    )

    if r.returncode != 0:
        print('FedAvg FAILED')
        sys.exit(1)

    # 📊 Evaluate global model
    m = evaluate_global()

    if m:
        print(f"  Acc:{m['accuracy']:.4f}  F1:{m['f1']:.4f}  Recall:{m['recall']:.4f}")

        log_rows.append({
            'round': rnd,
            'accuracy': round(m['accuracy'], 4),
            'f1': round(m['f1'], 4),
            'recall': round(m['recall'], 4)
        })


# ✅ Save log
pd.DataFrame(log_rows).to_csv(LOG_FILE, index=False)

print('\nTraining complete!')

# ✅ Final comparison
if log_rows:
    f = log_rows[0]
    l = log_rows[-1]

    print(f"Round 1  → Acc:{f['accuracy']:.4f}  F1:{f['f1']:.4f}  Recall:{f['recall']:.4f}")
    print(f"Round 15 → Acc:{l['accuracy']:.4f}  F1:{l['f1']:.4f}  Recall:{l['recall']:.4f}")
    print(f"F1 improvement: +{l['f1'] - f['f1']:.4f}")