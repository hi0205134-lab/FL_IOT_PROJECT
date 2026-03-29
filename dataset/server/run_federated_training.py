import subprocess, pickle, numpy as np, pandas as pd, os, sys
from sklearn.metrics import accuracy_score, f1_score, recall_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROUNDS   = 15
EPOCHS   = 100
LR       = '0.01'
LOG_FILE = 'models/fl_convergence_log.csv'

for i in [1,2,3]:
    if not os.path.exists(f'dataset/client{i}.csv'):
        print(f'[ERROR] dataset/client{i}.csv missing!'); sys.exit(1)

os.makedirs('models', exist_ok=True)
print(f'=== FEDERATED LEARNING — {ROUNDS} ROUNDS ===\n')

def evaluate_global():
    from clients.model import SimpleNN
    if not os.path.exists('models/global_model.pkl'): return None
    model = SimpleNN()
    model.set_weights(pickle.load(open('models/global_model.pkl','rb')))
    all_y, all_p = [], []
    for i in [1,2,3]:
        df = pd.read_csv(f'dataset/client{i}.csv')
        X  = df['distance'].values.reshape(-1,1).astype(float)
        y  = df['label'].values.astype(float)
        all_y.extend(y); all_p.extend(model.predict(X))
    return {'accuracy':accuracy_score(all_y,all_p),
            'f1':      f1_score(all_y,all_p,zero_division=0),
            'recall':  recall_score(all_y,all_p,zero_division=0)}

log_rows = []
for rnd in range(1, ROUNDS+1):
    print(f'--- ROUND {rnd}/{ROUNDS} ---')
    for cid in [1,2,3]:
        print(f'  Client {cid}...', end=' ', flush=True)
        r = subprocess.run(
            [sys.executable,'dataset/clients/train_client.py',
             '--client',str(cid),'--epochs',str(EPOCHS),'--lr',LR],
            capture_output=True, text=True)
        if r.returncode != 0: print(f'FAILED\n{r.stderr[-300:]}'); sys.exit(1)
        print('done')
    r = subprocess.run([sys.executable,'dataset/server/fedavg_server.py'],
                       capture_output=True, text=True)
    if r.returncode != 0: print('FedAvg FAILED'); sys.exit(1)
    m = evaluate_global()
    if m:
        print(f"  Acc:{m['accuracy']:.4f}  F1:{m['f1']:.4f}  Recall:{m['recall']:.4f}")
        log_rows.append({'round':rnd,
                         'accuracy':round(m['accuracy'],4),
                         'f1':round(m['f1'],4),
                         'recall':round(m['recall'],4)})
pd.DataFrame(log_rows).to_csv(LOG_FILE, index=False)
print('\nTraining complete!')
if log_rows:
    f=log_rows[0]; l=log_rows[-1]
    print(f"Round 1  → Acc:{f['accuracy']:.4f}  F1:{f['f1']:.4f}  Recall:{f['recall']:.4f}")
    print(f"Round 15 → Acc:{l['accuracy']:.4f}  F1:{l['f1']:.4f}  Recall:{l['recall']:.4f}")
    print(f"F1 improvement: +{l['f1']-f['f1']:.4f}")