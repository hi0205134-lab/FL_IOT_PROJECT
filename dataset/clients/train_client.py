import numpy as np, pandas as pd, pickle, os, sys, argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.model import SimpleNN

parser = argparse.ArgumentParser()
parser.add_argument('--client', type=int, required=True, choices=[1,2,3])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr',     type=float, default=0.01)
args = parser.parse_args()

CSV_FILE    = f'dataset/client{args.client}.csv'
MODEL_FILE  = f'models/client{args.client}.pkl'
GLOBAL_FILE = 'models/global_model.pkl'

print(f'=== Training Client {args.client} ===')

df = pd.read_csv(CSV_FILE)

X  = df['distance'].values.reshape(-1, 1).astype(np.float32)
y  = df['label'].values.astype(np.float32)

print(f'Samples:{len(X)} | Normal:{int(y.sum())} | Anomaly:{int(len(y)-y.sum())}')

model = SimpleNN(input_size=1, hidden_size=16, output_size=1)

if os.path.exists(GLOBAL_FILE):
    model.set_weights(pickle.load(open(GLOBAL_FILE, 'rb')))
    print('[OK] Loaded global weights')
else:
    print('[INFO] Training from scratch')

model.train(X, y, epochs=args.epochs, lr=args.lr, verbose=True)

preds = model.predict(X)

acc   = accuracy_score(y, preds)
f1    = f1_score(y, preds, zero_division=0)
rec   = recall_score(y, preds, zero_division=0)
cm    = confusion_matrix(y, preds)

print(f'Accuracy:{acc:.4f}  F1:{f1:.4f}  Recall:{rec:.4f}')
print(f'TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}')

os.makedirs('models', exist_ok=True)

pickle.dump({
    'weights':model.get_weights(),
    'sample_size':len(X),
    'client_id':args.client,
    'metrics':{
        'accuracy':float(acc),
        'f1':float(f1),
        'recall':float(rec)
    }
}, open(MODEL_FILE,'wb'))

print(f'Saved: {MODEL_FILE}')