import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAILWAY_URL = "https://fliotproject-production.up.railway.app"

st.set_page_config(
    page_title="FL IoT IDS Dashboard",
    page_icon="shield",
    layout="wide"
)

st.title("Federated Learning — IoT Intrusion Detection System")
st.caption("Privacy-Preserving C-IDS | 3 ESP32 Nodes | LoRa E32 | HC-SR04")

tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset",
    "Model Metrics",
    "FL Convergence",
    "Live Detection"
])

# ════════════════════════════════════════════════════════
# TAB 1 — DATASET
# ════════════════════════════════════════════════════════
with tab1:
    st.header("Collected Sensor Dataset")
    f = "dataset/clean_dataset.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples", len(df))
        c2.metric("Node 1",        len(df[df["node"] == "node1"]))
        c3.metric("Node 2",        len(df[df["node"] == "node2"]))
        c4.metric("Node 3",        len(df[df["node"] == "node3"]))

        col1, col2 = st.columns(2)
        col1.metric("Normal  (label=1)", int((df["label"] == 1).sum()))
        col2.metric("Anomaly (label=0)", int((df["label"] == 0).sum()))

        st.subheader("Last 20 rows")
        st.dataframe(df.tail(20), use_container_width=True)

        st.subheader("Average distance by node")
        pivot = df.groupby("node")["distance"].mean().reset_index()
        st.bar_chart(pivot.set_index("node"))
    else:
        st.warning("clean_dataset.csv not found in dataset/ folder.")

# ════════════════════════════════════════════════════════
# TAB 2 — MODEL METRICS
# ════════════════════════════════════════════════════════
with tab2:
    st.header("FL Client Model Metrics")
    rows = []
    for i in range(1, 4):
        p = f"models/client{i}.pkl"
        if os.path.exists(p):
            d = pickle.load(open(p, "rb"))
            m = d["metrics"]
            rows.append({
                "Client":   f"Node {i}",
                "Samples":  d["sample_size"],
                "Accuracy": round(m["accuracy"], 4),
                "F1 Score": round(m["f1"],       4),
                "Recall":   round(m["recall"],   4)
            })
    if rows:
        df_m = pd.DataFrame(rows)
        st.dataframe(df_m.set_index("Client"), use_container_width=True)
        st.bar_chart(df_m.set_index("Client")[["Accuracy", "F1 Score", "Recall"]])
    else:
        st.info("Run training first: python server/run_federated_training.py")

# ════════════════════════════════════════════════════════
# TAB 3 — FL CONVERGENCE
# ════════════════════════════════════════════════════════
with tab3:
    st.header("Federated Learning Convergence — 15 Rounds")
    lf = "models/fl_convergence_log.csv"
    if os.path.exists(lf):
        log = pd.read_csv(lf).set_index("round")
        c1, c2, c3 = st.columns(3)
        c1.metric("Round 1 F1",   f"{log['f1'].iloc[0]:.4f}")
        c2.metric("Final F1",     f"{log['f1'].iloc[-1]:.4f}",
                  delta=f"+{log['f1'].iloc[-1]-log['f1'].iloc[0]:.4f}")
        c3.metric("Final Recall", f"{log['recall'].iloc[-1]:.4f}")
        st.subheader("F1 and Recall per Round")
        st.line_chart(log[["f1", "recall"]])
        st.subheader("All Round Metrics")
        st.dataframe(log.round(4), use_container_width=True)
    else:
        st.info("Run training first.")

# ════════════════════════════════════════════════════════
# TAB 4 — LIVE DETECTION (fetches from Railway)
# ════════════════════════════════════════════════════════
with tab4:
    st.header("Live Anomaly Detection")
    st.caption(f"Fetching from: {RAILWAY_URL}/alerts")

    # ── Auto-refresh every 10 seconds ─────────────────
    import time
    placeholder = st.empty()

    # ── Fetch live alerts from Railway ────────────────
    try:
        resp   = requests.get(f"{RAILWAY_URL}/alerts", timeout=5)
        alerts = resp.json()
    except Exception as e:
        alerts = []
        st.warning(f"Cannot reach Railway: {e}")
        st.info("Make sure live_detect.py is running with hardware connected.")

    if alerts:
        latest = alerts[-1]

        # ── Status banner ──────────────────────────────
        if latest.get("is_anomaly"):
            st.error(
                f"ANOMALY DETECTED — "
                f"{latest['node']} | "
                f"Distance: {latest['distance']} cm | "
                f"Score: {latest['anomaly_score']} | "
                f"{latest['timestamp'][:19]}"
            )
        else:
            st.success(
                f"Normal — "
                f"{latest['node']} | "
                f"Distance: {latest['distance']} cm | "
                f"Score: {latest['anomaly_score']} | "
                f"{latest['timestamp'][:19]}"
            )

        # ── Summary metrics ────────────────────────────
        total  = len(alerts)
        anoms  = sum(1 for a in alerts if a.get("is_anomaly"))
        normal = total - anoms

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Readings", total)
        c2.metric("Normal",         normal)
        c3.metric("Anomalies",      anoms)
        c4.metric("Anomaly Rate",   f"{anoms/total*100:.1f}%")

        # ── Coordinated attack check ───────────────────
        recent = alerts[-10:]
        anom_nodes = set(
            a["node"] for a in recent if a.get("is_anomaly")
        )
        if len(anom_nodes) >= 2:
            st.error(
                f"COORDINATED ATTACK DETECTED! "
                f"Nodes triggered: {', '.join(sorted(anom_nodes))}"
            )

        # ── Live readings table ────────────────────────
        st.subheader("Last 20 Live Readings")
        df_alerts = pd.DataFrame(alerts[-20:][::-1])
        df_alerts["status"] = df_alerts["is_anomaly"].apply(
            lambda x: "ANOMALY" if x else "Normal"
        )
        st.dataframe(
            df_alerts[["timestamp", "node", "distance",
                        "anomaly_score", "status"]],
            use_container_width=True
        )

        # ── Anomaly score trend chart ──────────────────
        st.subheader("Anomaly Score Trend")
        chart_data = pd.DataFrame(alerts[-50:])
        chart_data.index = range(len(chart_data))
        st.line_chart(chart_data["anomaly_score"])

        # ── Per node breakdown ─────────────────────────
        st.subheader("Readings per Node")
        for node in ["node1", "node2", "node3"]:
            node_alerts = [a for a in alerts if a["node"] == node]
            if node_alerts:
                n_anom = sum(1 for a in node_alerts if a.get("is_anomaly"))
                st.text(
                    f"{node}: {len(node_alerts)} readings | "
                    f"Normal: {len(node_alerts)-n_anom} | "
                    f"Anomaly: {n_anom}"
                )

        # ── Clear button ───────────────────────────────
        if st.button("Clear All Live Alerts"):
            try:
                requests.post(f"{RAILWAY_URL}/clear_alerts", timeout=5)
                st.success("Cleared. Refresh page.")
            except Exception:
                st.error("Could not reach Railway to clear alerts.")

    else:
        st.info(
            "No live data yet. "
            "Connect hardware and run: python detection/live_detect.py"
        )

    # ── Manual detection (slider) ──────────────────────
    st.divider()
    st.subheader("Manual Detection Test (no hardware needed)")

    if os.path.exists("models/global_model.pkl") and \
       os.path.exists("models/scaler.pkl"):
        from clients.model import SimpleNN
        gw    = pickle.load(open("models/global_model.pkl", "rb"))
        sc    = pickle.load(open("models/scaler.pkl",        "rb"))
        model = SimpleNN()
        model.set_weights(gw)
        st.success("Global FL model loaded")

        node = st.selectbox("Select Node", ["node1", "node2", "node3"])
        dist = st.slider("Distance (cm)", 1.0, 400.0, 30.0, step=0.5)

        if st.button("Run Detection", type="primary"):
            scaled = np.clip(
                (dist - sc["min"]) / (sc["max"] - sc["min"]), 0.0, 1.0
            )
            X     = np.array([[scaled]], dtype=np.float32)
            score = round(1.0 - float(model.predict_proba(X)[0]), 4)
            if score > 0.5:
                st.error(f"ANOMALY DETECTED | Score: {score}")
            else:
                st.success(f"Normal | Score: {score}")
            st.json({
                "ClientID":      node,
                "Anomaly_Score": score,
                "Is_Anomaly":    score > 0.5,
                "Distance_cm":   dist,
                "Timestamp":     datetime.now().isoformat()
            })
    else:
        st.warning("Train the model first.")

    # ── Refresh button ─────────────────────────────────
    if st.button("Refresh Live Data"):
        st.rerun()

    st.caption("Click 'Refresh Live Data' to fetch latest from Railway")