import serial, requests, time, sys

SERIAL_PORT = 'COM5'
BAUD_RATE   = 115200
TARGET      = 4500

# Change to your Railway URL after deployment
FLASK_URL = 'https://your-app.up.railway.app/log'

print('Checking Railway API...')
try:
    check = FLASK_URL.replace('/log','/status')
    r = requests.get(check, timeout=10)
    print(f"[OK] API running. Rows: {r.json().get('total',0)}")
except Exception as e:
    print(f'[ERROR] Cannot reach API: {e}'); sys.exit(1)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
    time.sleep(2)
    print(f'[OK] Connected to {SERIAL_PORT}')
except serial.SerialException as e:
    print(f'[ERROR] {e}'); sys.exit(1)

print('Sending to Railway cloud...')
print('-'*55)
count = 0
try:
    while count < TARGET:
        raw = ser.readline()
        if not raw: continue
        line = raw.decode('utf-8',errors='ignore').strip()
        if not line or ',' not in line: continue
        if line.startswith('[') or line.startswith('Gateway'): continue
        parts = line.split(',')
        if len(parts)!=2: continue
        node_id=parts[0].strip()
        if node_id not in ['node1','node2','node3']: continue
        try: distance=float(parts[1].strip())
        except: continue
        if distance<2.0 or distance>400.0: continue
        count+=1
        payload={'sample':count,'node':node_id,'distance':distance,'label':1}
        try:
            resp=requests.post(FLASK_URL,json=payload,timeout=10)
            if resp.status_code==200:
                total=resp.json().get('total',count)
                print(f'[OK] Sample:{count:4d} | {node_id} | {distance:.2f}cm (total:{total})')
            else: count-=1
        except requests.exceptions.ConnectionError:
            print('[ERROR] Cannot reach Railway'); count-=1; time.sleep(3)
except KeyboardInterrupt:
    print(f'\nStopped. Collected:{count}')
finally:
    ser.close()
    print('Download: your-app.up.railway.app/download')
