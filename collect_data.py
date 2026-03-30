import serial
import requests
import time
import sys

SERIAL_PORT = 'COM5'     # Gateway ESP32 port - change if different
BAUD_RATE   = 115200
FLASK_URL   = 'https://fliotproject-production.up.railway.app/log'
TARGET      = 3000       # stop after 3000 samples

print('Checking Railway API...')
try:
    r = requests.get('https://fliotproject-production.up.railway.app/status', timeout=10)
    print(f"[OK] Railway API running. Current rows: {r.json().get('total',0)}")
except:
    print('[ERROR] Railway API not reachable! Check your internet connection.')
    sys.exit(1)

print(f'Connecting to {SERIAL_PORT}...')
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
    time.sleep(2)
    print(f'[OK] Connected to {SERIAL_PORT}')
except serial.SerialException as e:
    print(f'[ERROR] Cannot open {SERIAL_PORT}: {e}')
    print('Fix: Close Arduino Serial Monitor — it blocks the COM port')
    sys.exit(1)

print('Waiting for data from Gateway...')
print('-' * 55)
count = 0

try:
    while count < TARGET:
        raw = ser.readline()
        if not raw: continue
        line = raw.decode('utf-8', errors='ignore').strip()
        if not line or ',' not in line: continue
        if line.startswith('[') or line.startswith('Gateway'): continue
        parts = line.split(',')
        if len(parts) != 2: continue
        node_id = parts[0].strip()
        if node_id not in ['node1','node2','node3']: continue
        try: distance = float(parts[1].strip())
        except: continue
        if distance < 2.0 or distance > 400.0: continue
        count += 1
        payload = {'sample':count, 'node':node_id, 'distance':distance, 'label':1}
        try:
            resp = requests.post(FLASK_URL, json=payload, timeout=10)
            if resp.status_code == 200:
                total = resp.json().get('total', count)
                print(f'[OK] Sample:{count:4d} | {node_id} | {distance:7.2f} cm  (total:{total})')
            else:
                count -= 1
        except requests.exceptions.ConnectionError:
            print('[ERROR] Railway API not reachable! Check internet.')
            count -= 1
            time.sleep(2)
except KeyboardInterrupt:
    print(f'\nStopped. Samples collected: {count}')
finally:
    ser.close()
    print('Done! Check: https://fliotproject-production.up.railway.app/status')