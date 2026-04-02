from flask import Flask, request, jsonify, send_file
import pandas as pd
import os

app = Flask(__name__)
FILE       = 'dataset/sensor_dataset.xlsx'
ALERT_FILE = 'dataset/alerts.xlsx'
os.makedirs('dataset', exist_ok=True)

@app.route('/log', methods=['POST'])
def log():
    data = request.json
    if not data: return jsonify({'error': 'No data'}), 400
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
        print(f"[STORED] {row['sample']:4d} | {row['node']} | {row['distance']:.2f}cm (Total:{total})")
        return jsonify({'status': 'stored', 'total': total})
    except Exception as e:
        print(f'[ERROR] {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/alert', methods=['POST'])
def alert():
    data = request.json
    if not data: return jsonify({'error': 'No data'}), 400
    try:
        row = {
            'timestamp':   str(data.get('timestamp', '')),
            'node':        str(data['node']),
            'distance':    float(data['distance']),
            'score':       float(data['score']),
            'label':       str(data['label']),
            'coordinated': bool(data.get('coordinated', False)),
            'node1_status': str(data.get('all_nodes', {}).get('node1', 'UNKNOWN')),
            'node2_status': str(data.get('all_nodes', {}).get('node2', 'UNKNOWN')),
            'node3_status': str(data.get('all_nodes', {}).get('node3', 'UNKNOWN')),
        }
        if os.path.exists(ALERT_FILE):
            df = pd.read_excel(ALERT_FILE, engine='openpyxl')
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_excel(ALERT_FILE, index=False, engine='openpyxl')
        return jsonify({'status': 'alert stored'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/alerts', methods=['GET'])
def get_alerts():
    if os.path.exists(ALERT_FILE):
        df = pd.read_excel(ALERT_FILE, engine='openpyxl')
        return jsonify({'alerts': df.tail(50).to_dict(orient='records')})
    return jsonify({'alerts': []})

@app.route('/status', methods=['GET'])
def status():
    if os.path.exists(FILE):
        df = pd.read_excel(FILE, engine='openpyxl')
        return jsonify({'total': len(df), 'by_node': df.groupby('node').size().to_dict()})
    return jsonify({'total': 0})

@app.route('/download', methods=['GET'])
def download():
    if os.path.exists(FILE):
        return send_file(FILE, as_attachment=True, download_name='sensor_dataset.xlsx')
    return jsonify({'error': 'No data yet'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'Flask started on port {port}')
    app.run(host='0.0.0.0', port=port, debug=False)