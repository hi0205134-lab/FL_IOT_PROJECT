import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAILWAY_URL = "https://fliotproject-production.up.railway.app"

st.set_page_config(
    page_title="FL IoT IDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Federated Learning — IoT Intrusion Detection System")
st.caption("Privacy-Preserving C-IDS | 3 ESP32 Nodes | LoRa E32 | HC-SR04")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset",
    "🤖 Model Metrics",
    "📈 FL Convergence",
    "📡 Live Detection"
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
        c2.metric("Node 1", len(df[df["node"] == "node1"]))
        c3.metric("Node 2", len(df[df["node"] == "node2"]))
        c4.metric("Node 3", len(df[df["node"] == "node3"]))

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
# TAB 4 — LIVE DETECTION
# ════════════════════════════════════════════════════════
with tab4:
    st.header("📡 Live Anomaly Detection")
    st.caption(f"Fetching from: {RAILWAY_URL}/alerts")

    # ── Fetch live alerts from Railway ────────────────
    alerts = []
    try:
        resp   = requests.get(f"{RAILWAY_URL}/alerts", timeout=5)
        data   = resp.json()
        # ── FIXED: handle both response formats ───────
        # Your new app.py returns {"alerts": [...]}
        # Old app.py returned a list directly
        if isinstance(data, list):
            alerts = data
        elif isinstance(data, dict):
            alerts = data.get("alerts", [])
    except Exception as e:
        alerts = []
        st.warning(f"Cannot reach Railway: {e}")
        st.info("Make sure live_detect.py is running with hardware connected.")

    # ── FIXED: safe check before accessing alerts ─────
    if alerts and len(alerts) > 0:
        latest = alerts[-1]

        # ── Status banner ──────────────────────────────
        # FIXED: handle both old format (is_anomaly) 
        # and new format (label field)
        label      = latest.get("label", "")
        is_anomaly = latest.get("is_anomaly", label == "ANOMALY")
        score      = latest.get("anomaly_score", latest.get("score", 0))
        distance   = latest.get("distance", 0)
        node       = latest.get("node", "unknown")
        timestamp  = str(latest.get("timestamp", ""))[:19]

        if is_anomaly:
            st.error(
                f"🚨 ANOMALY DETECTED — "
                f"{node} | "
                f"Distance: {distance} cm | "
                f"Score: {score} | "
                f"{timestamp}"
            )
        else:
            st.success(
                f"✅ Normal — "
                f"{node} | "
                f"Distance: {distance} cm | "
                f"Score: {score} | "
                f"{timestamp}"
            )

        # ── Summary metrics ────────────────────────────
        total  = len(alerts)
        # FIXED: handle both is_anomaly and label fields
        anoms  = sum(1 for a in alerts if
                     a.get("is_anomaly", a.get("label") == "ANOMALY"))
        normal = total - anoms

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Readings", total)
        c2.metric("Normal",         normal)
        c3.metric("Anomalies",      anoms)
        c4.metric("Anomaly Rate",   f"{anoms/total*100:.1f}%" if total > 0 else "0%")

        # ── Current node status ────────────────────────
        st.subheader("Current Node Status")
        col1, col2, col3 = st.columns(3)
        for col, node_key, loc in zip(
            [col1, col2, col3],
            ["node1_status", "node2_status", "node3_status"],
            ["Front Door", "Window", "Hallway"]
        ):
            status = latest.get(node_key, "UNKNOWN")
            icon   = "🔴" if status == "ANOMALY" else "🟢"
            col.metric(f"{icon} {loc}", status)

        # ── Coordinated attack check ───────────────────
        recent = alerts[-10:]
        anom_nodes = set(
            a["node"] for a in recent if
            a.get("is_anomaly", a.get("label") == "ANOMALY")
        )
        if len(anom_nodes) >= 2:
            st.error(
                f"🚨 COORDINATED ATTACK DETECTED! "
                f"Nodes triggered: {', '.join(sorted(anom_nodes))}"
            )

        # ── Live readings table ────────────────────────
        st.subheader("Last 20 Live Readings")
        df_alerts = pd.DataFrame(alerts[-20:][::-1])

        # FIXED: handle both label formats
        if "label" in df_alerts.columns:
            df_alerts["status"] = df_alerts["label"]
        elif "is_anomaly" in df_alerts.columns:
            df_alerts["status"] = df_alerts["is_anomaly"].apply(
                lambda x: "ANOMALY" if x else "NORMAL"
            )

        # show available columns only
        show_cols = ["timestamp", "node", "distance", "score", "status", "coordinated"]
        show_cols = [c for c in show_cols if c in df_alerts.columns]

        def highlight(row):
            if row.get("status") == "ANOMALY":
                return ["background-color: #ffcccc"] * len(row)
            return ["background-color: #ccffcc"] * len(row)

        st.dataframe(
            df_alerts[show_cols].style.apply(highlight, axis=1),
            use_container_width=True
        )

        # ── Anomaly score trend chart ──────────────────
        st.subheader("Score Trend")
        chart_data = pd.DataFrame(alerts[-50:])
        # FIXED: handle both score column names
        score_col = "score" if "score" in chart_data.columns else "anomaly_score"
        if score_col in chart_data.columns:
            st.line_chart(chart_data[score_col])

        # ── Per node breakdown ─────────────────────────
        st.subheader("Per-Node Summary")
        col1, col2, col3 = st.columns(3)
        for col, node_name in zip([col1, col2, col3],
                                   ["node1", "node2", "node3"]):
            node_alerts = [a for a in alerts if a["node"] == node_name]
            if node_alerts:
                n_anom = sum(1 for a in node_alerts if
                             a.get("is_anomaly", a.get("label") == "ANOMALY"))
                pct    = n_anom / len(node_alerts) * 100
                icon   = "🔴" if n_anom > 0 else "🟢"
                col.metric(
                    f"{icon} {node_name}",
                    f"{n_anom} anomalies",
                    f"{pct:.1f}% of {len(node_alerts)} readings"
                )
            else:
                col.metric(f"🟢 {node_name}", "No data yet")

    else:
        st.info("No live data yet. Connect hardware and run: python detection/live_detect.py")

    # ── Manual detection (slider) ──────────────────────
    st.divider()
    st.subheader("🔧 Manual Detection Test (no hardware needed)")

    if os.path.exists("models/global_model.pkl") and \
       os.path.exists("models/scaler.pkl"):
        from clients.model import SimpleNN
        gw    = pickle.load(open("models/global_model.pkl", "rb"))
        sc    = pickle.load(open("models/scaler.pkl", "rb"))
        model = SimpleNN()
        model.set_weights(gw)
        st.success("✅ Global FL model loaded")

        node = st.selectbox("Select Node", ["node1", "node2", "node3"])
        dist = st.slider("Distance (cm)", 1.0, 400.0, 30.0, step=0.5)

        if st.button("Run Detection", type="primary"):
            # FIXED: use per-node scaler not global
            node_scalers = sc.get("nodes", {})
            ns = node_scalers.get(node, sc.get("global", sc))
            scaled = float(np.clip(
                (dist - ns["min"]) / (ns["max"] - ns["min"]), 0.0, 1.0
            ))
            X    = np.array([[scaled]], dtype=np.float32)
            prob = float(model.predict_proba(X)[0])
            # FIXED: prob > 0.5 = NORMAL (label=1 is normal in dataset)
            is_anom = prob <= 0.5
            score   = round(prob, 4)
            if is_anom:
                st.error(f"🚨 ANOMALY DETECTED | Score: {score}")
            else:
                st.success(f"✅ Normal | Score: {score}")
            st.json({
                "Node":       node,
                "Distance":   dist,
                "Score":      score,
                "Is_Anomaly": is_anom,
                "Timestamp":  datetime.now().isoformat()
            })
    else:
        st.warning("Train the model first.")

    # ── Auto refresh ───────────────────────────────────
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Live Data"):
            st.rerun()
    with col2:
        auto = st.checkbox("Auto-refresh every 10 seconds", value=False)

    if auto:
        import time
        st.caption("⏱ Auto-refreshing every 10 seconds...")
        time.sleep(10)
        st.rerun()

    st.caption("Data fetched from Railway cloud API in real-time")