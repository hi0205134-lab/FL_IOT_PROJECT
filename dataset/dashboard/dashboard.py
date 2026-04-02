import streamlit as st
import pandas as pd
import pickle
import os
import requests
import time

st.set_page_config(page_title="FL IoT IDS", layout="wide", page_icon="🛡️")

RAILWAY_URL = 'https://fliotproject-production.up.railway.app'

st.title("🛡️ FL IoT Intrusion Detection System")
st.caption("Privacy-Preserving Federated Learning | 3 ESP32 Nodes | Real-time Detection")

tabs = st.tabs(["📡 Live Detection", "📊 FL Results", "📈 Convergence", "ℹ️ About"])

# ── TAB 1: LIVE DETECTION ──────────────────────────────────────────────
with tabs[0]:

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_sec  = st.selectbox("Refresh every", [5, 10, 30], index=1)
        st.markdown("---")
        st.markdown("**Node Ranges:**")
        st.markdown("node1 (Door): 15–40 cm")
        st.markdown("node2 (Window): 2–20 cm")
        st.markdown("node3 (Hallway): 15–75 cm")

    st.subheader("📡 Real-time Node Status")

    # Fetch status
    try:
        r    = requests.get(f'{RAILWAY_URL}/status', timeout=5)
        data = r.json()
        total   = data.get('total', 0)
        by_node = data.get('by_node', {})
        api_ok  = True
    except:
        total   = 0
        by_node = {}
        api_ok  = False

    if not api_ok:
        st.error("Cannot reach Railway API. Check your internet connection.")

    # Node status cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", total)
    c2.metric("node1 (Front Door)", by_node.get('node1', 0))
    c3.metric("node2 (Window)",     by_node.get('node2', 0))
    c4.metric("node3 (Hallway)",    by_node.get('node3', 0))

    st.markdown("---")

    # Fetch alerts
    try:
        r2     = requests.get(f'{RAILWAY_URL}/alerts', timeout=5)
        alerts = r2.json().get('alerts', [])
    except:
        alerts = []

    if alerts:
        df_alerts = pd.DataFrame(alerts)

        # Check coordinated attack from latest alerts
        latest = alerts[-5:]
        anomaly_nodes = list(set([
            a['node'] for a in latest if a.get('label') == 'ANOMALY'
        ]))

        if len(anomaly_nodes) >= 2:
            st.error(f"🚨 COORDINATED ATTACK DETECTED!")
            st.error(f"Nodes triggered: {', '.join(anomaly_nodes)}")
            st.warning("Multiple zones breached simultaneously!")
        elif len(anomaly_nodes) == 1:
            st.warning(f"⚠️ ANOMALY detected at {anomaly_nodes[0]}")
        else:
            st.success("✅ All nodes NORMAL")

        # All node status from latest reading
        if 'node1_status' in df_alerts.columns:
            st.subheader("Current Node Status")
            last = df_alerts.iloc[-1]
            col1, col2, col3 = st.columns(3)
            for col, node, loc in zip(
                [col1, col2, col3],
                ['node1_status', 'node2_status', 'node3_status'],
                ['Front Door', 'Window', 'Hallway']
            ):
                status = last.get(node, 'UNKNOWN')
                icon   = "🔴" if status == 'ANOMALY' else "🟢"
                col.metric(f"{icon} {loc}", status)

        # Recent alerts table
        st.subheader("Recent Alerts (last 20)")
        display_cols = ['timestamp', 'node', 'distance', 'score', 'label', 'coordinated']
        display_cols = [c for c in display_cols if c in df_alerts.columns]

        def highlight(row):
            if row.get('label') == 'ANOMALY':
                return ['background-color: #ffcccc'] * len(row)
            return ['background-color: #ccffcc'] * len(row)

        st.dataframe(
            df_alerts[display_cols].tail(20).style.apply(highlight, axis=1),
            use_container_width=True
        )

        # Per-node summary
        st.subheader("Per-Node Anomaly Summary")
        col1, col2, col3 = st.columns(3)
        for col, node in zip([col1, col2, col3], ['node1', 'node2', 'node3']):
            node_data     = df_alerts[df_alerts['node'] == node]
            total_node    = len(node_data)
            anomaly_count = len(node_data[node_data['label'] == 'ANOMALY'])
            pct = (anomaly_count / total_node * 100) if total_node > 0 else 0
            icon = "🔴" if anomaly_count > 0 else "🟢"
            col.metric(
                f"{icon} {node}",
                f"{anomaly_count} anomalies",
                f"{pct:.1f}% of {total_node} readings"
            )
    else:
        st.info("No live alert data yet.")
        st.markdown("""
        To see live data:
        - Connect ESP32 hardware and run `live_detect.py`
        - Or run `collect_data.py` with SIMULATE=True
        """)

    # Auto-refresh
    if auto_refresh:
        st.caption(f"⏱ Auto-refreshing every {refresh_sec} seconds...")
        time.sleep(refresh_sec)
        st.rerun()

# ── TAB 2: FL RESULTS ──────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Federated Learning — Client Results")

    files = ['models/client1.pkl', 'models/client2.pkl', 'models/client3.pkl']
    if all(os.path.exists(f) for f in files):
        rows = []
        for i, f in enumerate(files, 1):
            d = pickle.load(open(f, 'rb'))
            m = d.get('metrics', {})
            rows.append({
                'Client':   f'Client {i} (node{i})',
                'Samples':  d.get('sample_size', 0),
                'Accuracy': round(m.get('accuracy', 0), 4),
                'F1 Score': round(m.get('f1', 0), 4),
                'Recall':   round(m.get('recall', 0), 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("Run federated training first.")

# ── TAB 3: CONVERGENCE ─────────────────────────────────────────────────
with tabs[2]:
    st.subheader("FL Convergence — 15 Rounds")

    log_file = 'models/fl_convergence_log.csv'
    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
        st.line_chart(df_log.set_index('round')[['accuracy', 'f1', 'recall']])
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Accuracy", f"{df_log['accuracy'].iloc[-1]:.4f}",
                    f"+{df_log['accuracy'].iloc[-1]-df_log['accuracy'].iloc[0]:.4f}")
        col2.metric("Final F1",       f"{df_log['f1'].iloc[-1]:.4f}",
                    f"+{df_log['f1'].iloc[-1]-df_log['f1'].iloc[0]:.4f}")
        col3.metric("Final Recall",   f"{df_log['recall'].iloc[-1]:.4f}",
                    f"+{df_log['recall'].iloc[-1]-df_log['recall'].iloc[0]:.4f}")
        st.dataframe(df_log, use_container_width=True)
    else:
        st.warning("Run federated training first.")

# ── TAB 4: ABOUT ───────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Project Information")
    st.markdown("""
    **Privacy-Preserving Collaborative IDS using Federated Learning**

    | Property | Value |
    |---|---|
    | Hardware | 3x ESP32 + HC-SR04 + LoRa E32 |
    | Dataset | 4500 rows — 84.2% Normal, 15.8% Anomaly |
    | FL Algorithm | FedAvg — 15 rounds |
    | Normalization | Per-node (each node scaled independently) |
    | Cloud | Railway (Flask API) + Streamlit |

    **Node Deployment:**
    - node1 → Front Door (normal: 15–40 cm)
    - node2 → Window (normal: 2–20 cm)
    - node3 → Hallway (normal: 15–75 cm)

    **How Federated Learning Works:**
    1. Each node trains on its own local data
    2. Only model weights (not raw data) sent to server
    3. Server averages weights using FedAvg
    4. Global model sent back to all nodes
    5. Privacy preserved — raw data never leaves device
    """)