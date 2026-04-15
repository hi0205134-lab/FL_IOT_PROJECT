import pandas as pd
import pickle
import os
import sys
INPUT  = 'dataset/clean_dataset.csv'
OUTPUT = 'dataset/normalized_dataset.csv'
if not os.path.exists(INPUT):
    print('[ERROR] clean_dataset.csv not found!')
    sys.exit(1)
 
df = pd.read_csv(INPUT)
print(f'Loaded {len(df)} rows')
 
global_min = df['distance'].min()
global_max = df['distance'].max()
df['distance_raw'] = df['distance'].copy()
 
node_scalers = {}
for node in ['node1', 'node2', 'node3']:
    mask  = df['node'] == node
    n_min = df.loc[mask, 'distance'].min()
    n_max = df.loc[mask, 'distance'].max()
    df.loc[mask, 'distance'] = (df.loc[mask, 'distance'] - n_min) / (n_max - n_min)
    node_scalers[node] = {'min': n_min, 'max': n_max}
    print(f'  {node}: min={n_min:.2f}  max={n_max:.2f}')
 
print(f'After: min={df["distance"].min():.4f}  max={df["distance"].max():.4f}')
 
os.makedirs('models', exist_ok=True)
pickle.dump({
    'global': {'min': global_min, 'max': global_max},
    'nodes':  node_scalers
}, open('models/scaler.pkl', 'wb'))
 
print('Scaler saved: models/scaler.pkl')
df.to_csv(OUTPUT, index=False)
print(f'Saved: {OUTPUT}')
