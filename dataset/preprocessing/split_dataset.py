import pandas as pd
import os
import sys

if not os.path.exists('dataset/normalized_dataset.csv'):
    print('[ERROR] Run normalize.py first!')
    sys.exit(1)

df = pd.read_csv('dataset/normalized_dataset.csv')
print(f'Loaded {len(df)} rows')

for i, node in enumerate(['node1', 'node2', 'node3'], start=1):
    s   = df[df['node'] == node].copy().reset_index(drop=True)
    out = f'dataset/client{i}.csv'
    s.to_csv(out, index=False)
    print(f'client{i}.csv → {len(s)} rows | Normal:{(s["label"]==1).sum()} Anomaly:{(s["label"]==0).sum()}')

print('Split complete!')