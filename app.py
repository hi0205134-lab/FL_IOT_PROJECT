from flask import Flask, request, jsonify, send_file
import pandas as pd
import os

app = Flask(__name__)
FILE = 'dataset/sensor_dataset.xlsx'
os.makedirs('dataset', exist_ok=True)

@app.route('/log', methods=['POST'])
def log():
    data = request.json
    if not data: return jsonify({'error':'No data'}), 400
    try:
        row = {'sample':int(data['sample']),'node':str(data['node']),
               'distance':float(data['distance']),'label':int(data.get('label',1))}
        if os.path.exists(FILE):
            df = pd.read_excel(FILE, engine='openpyxl')
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_excel(FILE, index=False, engine='openpyxl')
        total = len(df)
        print(f"[STORED] {row['sample']:4d} | {row['node']} | {row['distance']:.2f}cm (Total:{total})")
        return jsonify({'status':'stored','total':total})
    except Exception as e:
        print(f'[ERROR] {e}')
        return jsonify({'error':str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    if os.path.exists(FILE):
        df = pd.read_excel(FILE, engine='openpyxl')
        return jsonify({'total':len(df),'by_node':df.groupby('node').size().to_dict()})
    return jsonify({'total':0})

@app.route('/download', methods=['GET'])
def download():
    if os.path.exists(FILE):
        return send_file(FILE, as_attachment=True, download_name='sensor_dataset.xlsx')
    return jsonify({'error':'No data yet'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'Flask started on port {port}')
    app.run(host='0.0.0.0', port=port, debug=False)
