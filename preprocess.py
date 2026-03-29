import pandas as pd
import os, sys

INPUT  = "dataset/sensor_dataset.xlsx"
OUTPUT = "dataset/clean_dataset.csv"

if not os.path.exists(INPUT):
    print('[ERROR] No dataset found. Collect data first!')
    sys.exit(1)

df = pd.read_excel(INPUT, engine='openpyxl')
print(f'Loaded {len(df)} rows')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
df.dropna(subset=['distance'], inplace=True)

for node in sorted(df['node'].unique()):
    s = df[df['node']==node]
    print(f'  {node}: {len(s)} samples | {s["distance"].min():.2f} - {s["distance"].max():.2f} cm')

df['label'] = 1
for node in df['node'].unique():
    mask = df['node'] == node
    vals = df.loc[mask, 'distance']
    lo, hi = vals.quantile(0.05), vals.quantile(0.95)
    df.loc[mask & ((df['distance'] < lo) | (df['distance'] > hi)), 'label'] = 0

print(f'Normal: {(df["label"]==1).sum()}  Anomaly: {(df["label"]==0).sum()}')
df.to_csv(OUTPUT, index=False)
print(f'Saved: {OUTPUT}')