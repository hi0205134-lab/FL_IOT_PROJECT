import pandas as pd
import pickle
import os
import sys

INPUT  = 'dataset/clean_dataset.csv'
OUTPUT = 'dataset/normalized_dataset.csv'

if not os.path.exists(INPUT):
    print('[ERROR] clean_dataset.csv not found in dataset/ folder!')
    sys.exit(1)

df    = pd.read_csv(INPUT)
print(f'Loaded {len(df)} rows')

d_min = df['distance'].min()
d_max = df['distance'].max()
print(f'Before: min={d_min:.4f}  max={d_max:.4f}  mean={df["distance"].mean():.4f}')

df['distance_raw'] = df['distance'].copy()
df['distance']     = (df['distance'] - d_min) / (d_max - d_min)
print(f'After : min={df["distance"].min():.4f}  max={df["distance"].max():.4f}')

os.makedirs('models', exist_ok=True)
pickle.dump({'min': d_min, 'max': d_max}, open('models/scaler.pkl', 'wb'))
print('Scaler saved: models/scaler.pkl')

df.to_csv(OUTPUT, index=False)
print(f'Saved: {OUTPUT}')