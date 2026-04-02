import pandas as pd
import numpy as np
import pickle
import os
import sys

INPUT  = 'dataset/clean_dataset.csv'
OUTPUT = 'dataset/normalized_dataset.csv'

if not os.path.exists(INPUT):
    print('[ERROR] clean_dataset.csv not found!'); sys.exit(1)

df = pd.read_csv(INPUT)
print(f'Loaded {len(df)} rows')

global_min = df['distance'].min()
global_max = df['distance'].max()

df['distance_raw'] = df['distance'].copy()
node_scalers = {}

for node in ['node1', 'node2', 'node3']:
    mask        = df['node'] == node
    # Use NORMAL readings only (label=1) to set min/max
    normal_mask = mask & (df['label'] == 1)
    n_min = df.loc[normal_mask, 'distance'].min()
    n_max = df.loc[normal_mask, 'distance'].max()

    # Scale and CLIP between 0 and 1
    scaled = (df.loc[mask, 'distance'] - n_min) / (n_max - n_min)
    df.loc[mask, 'distance'] = np.clip(scaled, 0.0, 1.0)

    node_scalers[node] = {'min': n_min, 'max': n_max}
    print(f'{node}: normal_min={n_min:.2f}cm  normal_max={n_max:.2f}cm')

print(f'After normalization: min={df["distance"].min():.4f}  max={df["distance"].max():.4f}')

os.makedirs('models', exist_ok=True)
pickle.dump({
    'global': {'min': global_min, 'max': global_max},
    'nodes':  node_scalers
}, open('models/scaler.pkl', 'wb'))

print('Scaler saved: models/scaler.pkl')
df.to_csv(OUTPUT, index=False)
print(f'Saved: {OUTPUT}')