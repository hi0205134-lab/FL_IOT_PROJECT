import streamlit as st
import pandas as pd, numpy as np, pickle, os, sys, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title='FL IoT IDS', page_icon='shield', layout='wide')
st.title('Federated Learning — IoT Intrusion Detection System')
st.caption('Privacy-Preserving C-IDS | 3 ESP32 Nodes | LoRa E32 | HC-SR04')

tab1,tab2,tab3,tab4 = st.tabs(['Dataset','Model Metrics','FL Convergence','Live Detection'])

with tab1:
    st.header('Collected Sensor Dataset')
    f = 'dataset/clean_dataset.csv'
    if os.path.exists(f):
        df = pd.read_csv(f)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Total Samples', len(df))
        c2.metric('Node 1', len(df[df['node']=='node1']))
        c3.metric('Node 2', len(df[df['node']=='node2']))
        c4.metric('Node 3', len(df[df['node']=='node3']))
        col1,col2 = st.columns(2)
        col1.metric('Normal  (label=1)', int((df['label']==1).sum()))
        col2.metric('Anomaly (label=0)', int((df['label']==0).sum()))
        st.dataframe(df.tail(20), use_container_width=True)
        pivot = df.groupby('node')['distance'].mean().reset_index()
        st.bar_chart(pivot.set_index('node'))
    else:
        st.warning('clean_dataset.csv not found.')

with tab2:
    st.header('FL Client Model Metrics')
    rows=[]
    for i in range(1,4):
        p=f'models/client{i}.pkl'
        if os.path.exists(p):
            d=pickle.load(open(p,'rb')); m=d['metrics']
            rows.append({'Client':f'Node {i}','Samples':d['sample_size'],
                         'Accuracy':round(m['accuracy'],4),
                         'F1 Score':round(m['f1'],4),'Recall':round(m['recall'],4)})
    if rows:
        df_m=pd.DataFrame(rows)
        st.dataframe(df_m.set_index('Client'),use_container_width=True)
        st.bar_chart(df_m.set_index('Client')[['Accuracy','F1 Score','Recall']])
    else: st.info('Run training first.')

with tab3:
    st.header('Federated Learning Convergence')
    lf='models/fl_convergence_log.csv'
    if os.path.exists(lf):
        log=pd.read_csv(lf).set_index('round')
        c1,c2,c3=st.columns(3)
        c1.metric('Round 1 F1',  f"{log['f1'].iloc[0]:.4f}")
        c2.metric('Final F1',    f"{log['f1'].iloc[-1]:.4f}",
                  delta=f"+{log['f1'].iloc[-1]-log['f1'].iloc[0]:.4f}")
        c3.metric('Final Recall',f"{log['recall'].iloc[-1]:.4f}")
        st.line_chart(log[['f1','recall']])
        st.dataframe(log.round(4),use_container_width=True)
    else: st.info('Run training first.')

with tab4:
    st.header('Anomaly Detection')
    if os.path.exists('models/global_model.pkl') and os.path.exists('models/scaler.pkl'):
        from clients.model import SimpleNN
        gw=pickle.load(open('models/global_model.pkl','rb'))
        sc=pickle.load(open('models/scaler.pkl','rb'))
        model=SimpleNN(); model.set_weights(gw)
        st.success('Global FL model loaded and ready')
        node=st.selectbox('Select Node',['node1','node2','node3'])
        dist=st.slider('Distance (cm)',1.0,400.0,30.0,step=0.5)
        if st.button('Run Detection',type='primary'):
            scaled=np.clip((dist-sc['min'])/(sc['max']-sc['min']),0.0,1.0)
            X=np.array([[scaled]],dtype=np.float32)
            score=round(1.0-float(model.predict_proba(X)[0]),4)
            if score>0.5: st.error(f'ANOMALY DETECTED | Score:{score}')
            else:         st.success(f'Normal | Score:{score}')
            st.json({'ClientID':node,'Anomaly_Score':score,
                     'Is_Anomaly':score>0.5,'Distance_cm':dist,
                     'Timestamp':datetime.now().isoformat()})
    else: st.warning('Train the model first.')
