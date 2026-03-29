from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)
FILE = 'dataset/sensor_dataset.xlsx'
os.makedirs('dataset', exist_ok=True)

@app.route('/log', methods=['POST'])
def log():
    data = request.json
    if not data:
        return jsonify({'error': 'No data'}), 400
    try:
        row = {
            'sample':   int(data['sample']),
            'node':     str(data['node']),
            'distance': float(data['distance']),
            'label':    int(data.get('label', 1))
        }
        if os.path.exists(FILE):
            df = pd.read_excel(FILE, engine='openpyxl')
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_excel(FILE, index=False, engine='openpyxl')
        total = len(df)
        print(f"[STORED] Sample:{row['sample']:4d} | {row['node']} | {row['distance']:.2f} cm  (Total: {total})")
        return jsonify({'status': 'stored', 'total': total})
    except Exception as e:
        print(f'[ERROR] {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    if os.path.exists(FILE):
        df = pd.read_excel(FILE, engine='openpyxl')
        return jsonify({'total': len(df),
                        'by_node': df.groupby('node').size().to_dict()})
    return jsonify({'total': 0})

if __name__ == '__main__':
    print('Flask started. Excel will be saved to:', os.path.abspath(FILE))
    app.run(host='0.0.0.0', port=5000, debug=False)