from flask import Flask, request, jsonify, send_file
import pandas as pd
import json
import os

app = Flask(__name__)

EXCEL_FILE = "dataset/sensor_dataset.xlsx"
ALERT_FILE = "dataset/live_alerts.json"
os.makedirs("dataset", exist_ok=True)


# ── Route 1: Store sensor data during collection ──────
@app.route("/log", methods=["POST"])
def log():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    try:
        row = {
            "sample":   int(data["sample"]),
            "node":     str(data["node"]),
            "distance": float(data["distance"]),
            "label":    int(data.get("label", 1))
        }
        if os.path.exists(EXCEL_FILE):
            df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")
        total = len(df)
        print(f"[STORED] {row['sample']:4d} | {row['node']} | {row['distance']:.2f}cm (Total:{total})")
        return jsonify({"status": "stored", "total": total})
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# ── Route 2: Check collection status ─────────────────
@app.route("/status", methods=["GET"])
def status():
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
        return jsonify({
            "total":   len(df),
            "by_node": df.groupby("node").size().to_dict()
        })
    return jsonify({"total": 0})


# ── Route 3: Download collected Excel file ────────────
@app.route("/download", methods=["GET"])
def download():
    if os.path.exists(EXCEL_FILE):
        return send_file(EXCEL_FILE, as_attachment=True,
                         download_name="sensor_dataset.xlsx")
    return jsonify({"error": "No data yet"}), 404


# ── Route 4: Receive live detection results ───────────
@app.route("/alert", methods=["POST"])
def alert():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    try:
        if os.path.exists(ALERT_FILE):
            with open(ALERT_FILE, "r") as f:
                alerts = json.load(f)
        else:
            alerts = []
        alerts.append(data)
        alerts = alerts[-200:]  # keep last 200 readings only
        with open(ALERT_FILE, "w") as f:
            json.dump(alerts, f)
        status_str = "ANOMALY" if data.get("is_anomaly") else "Normal"
        print(f"[ALERT] {data.get('node')} | {data.get('distance')}cm | {status_str} | score={data.get('anomaly_score')}")
        return jsonify({"status": "stored"})
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# ── Route 5: Get live alerts for dashboard ────────────
@app.route("/alerts", methods=["GET"])
def get_alerts():
    if os.path.exists(ALERT_FILE):
        with open(ALERT_FILE, "r") as f:
            return jsonify(json.load(f))
    return jsonify([])


# ── Route 6: Clear all live alerts ───────────────────
@app.route("/clear_alerts", methods=["POST"])
def clear_alerts():
    if os.path.exists(ALERT_FILE):
        os.remove(ALERT_FILE)
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Flask started on port {port}")
    print(f"Routes: /log  /status  /download  /alert  /alerts  /clear_alerts")
    app.run(host="0.0.0.0", port=port, debug=False)