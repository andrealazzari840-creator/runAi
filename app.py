import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Apple Watch AI Analytics", layout="wide")

if 'aw_connected' not in st.session_state:
    st.session_state.aw_connected = False

@st.cache_data
def load_data():
    np.random.seed(42)
    days = 600
    dates = pd.date_range(start="2024-01-01", periods=days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Distance_km': np.random.normal(12, 4, days).clip(2, 35),
        'Avg_BPM': np.random.normal(148, 12, days).clip(110, 190),
        'Max_BPM': np.random.normal(170, 15, days).clip(130, 200),
        'Cadence_spm': np.random.normal(168, 8, days).clip(140, 200),
        'Elevation_m': np.random.normal(150, 80, days).clip(0, 1000),
        'Sleep_Hours': np.random.normal(7.2, 1.1, days).clip(4, 10),
        'HRV_ms': np.random.normal(60, 15, days).clip(20, 100),
        'RPE': np.random.randint(3, 10, days)
    })
    
    df['Daily_Load'] = df['RPE'] * df['Distance_km']
    df['Acute_Load_7D'] = df['Daily_Load'].rolling(window=7, min_periods=1).mean()
    df['Chronic_Load_28D'] = df['Daily_Load'].rolling(window=28, min_periods=1).mean()
    df['ACWR'] = (df['Acute_Load_7D'] / df['Chronic_Load_28D']).fillna(1.0)
    df['Pace_min_km'] = 9.0 - (df['Cadence_spm'] * 0.015) + (df['Avg_BPM'] * 0.005) + np.random.normal(0, 0.1, days)
    injury_prob = np.where((df['ACWR'] > 1.4) | (df['HRV_ms'] < 40) | (df['Sleep_Hours'] < 5.5), 0.75, 0.05)
    df['Injury_Event'] = np.random.binomial(1, injury_prob)
    
    return df.dropna()

df = load_data()

@st.cache_resource
def train_models(data):
    features = ['Distance_km', 'Avg_BPM', 'Max_BPM', 'Cadence_spm', 'Elevation_m', 'Sleep_Hours', 'HRV_ms', 'RPE', 'ACWR']
    X = data[features]
    
    xgb_perf = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    xgb_perf.fit(X, data['Pace_min_km'])
    
    xgb_inj = XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss', random_state=42)
    xgb_inj.fit(X, data['Injury_Event'])
    
    return xgb_perf, xgb_inj, features

xgb_perf, xgb_inj, feature_cols = train_models(df)

st.title("⌚ Apple Watch & AI Prediction")

st.sidebar.header("🔗 Sincronizzazione")
if st.sidebar.button("📡 Estrai Dati da Apple Watch"):
    st.session_state.aw_connected = True
    st.sidebar.success("✅ Sincronizzato!")

if st.session_state.aw_connected:
    st.sidebar.info("**Dati Caricati:**\n- 💤 Sonno: 5.2h\n- 💓 HRV: 41ms\n- 📈 ACWR: 1.48")

tab1, tab2, tab3 = st.tabs(["📊 Simulatore", "📈 Analisi", "🧠 Feature Importance"])

with tab1:
    st.header("Previsione Allenamento")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: in_dist = st.number_input("Distanza (km)", 5.0, 30.0, 10.0)
    with col2: in_elev = st.number_input("Dislivello (m)", 0.0, 1000.0, 100.0)
    with col3: in_rpe = st.slider("RPE", 1, 10, 6)
    with col4: in_cadence = st.number_input("Cadenza (spm)", 140.0, 200.0, 165.0)
    
    col5, col6, col7, col8, col9 = st.columns(5)
    with col5: in_avg_bpm = st.number_input("Avg BPM", 110.0, 190.0, 145.0)
    with col6: in_max_bpm = st.number_input("Max BPM", 130.0, 200.0, 170.0)
    with col7: in_sleep = st.number_input("Sonno (h)", 4.0, 10.0, 7.5)
    with col8: in_hrv = st.number_input("HRV (ms)", 20.0, 100.0, 55.0)
    with col9: in_acwr = st.number_input("ACWR", 0.5, 2.0, 1.2)
    
    if st.button("🚀 Previsione IA", use_container_width=True):
        input_df = pd.DataFrame({
            'Distance_km': [in_dist], 'Avg_BPM': [in_avg_bpm],
            'Max_BPM': [in_max_bpm], 'Cadence_spm': [in_cadence],
            'Elevation_m': [in_elev], 'Sleep_Hours': [in_sleep],
            'HRV_ms': [in_hrv], 'RPE': [in_rpe], 'ACWR': [in_acwr]
        })
        
        pred_pace = xgb_perf.predict(input_df)[0]
        pred_risk = xgb_inj.predict_proba(input_df)[0][1] * 100
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("⏱️ Ritmo Stimato", f"{pred_pace:.2f} min/km")
        col_res2.metric("⚠️ Rischio Infortuni", f"{pred_risk:.1f}%")
        
        if pred_risk > 60:
            st.error("RISCHIO ALTO - Riduci l'intensità!")

with tab2:
    st.header("Analisi Storica")
    fig = px.scatter(df, x='Sleep_Hours', y='HRV_ms', color='Injury_Event',
                     title="Sonno vs HRV e Infortuni", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Feature Importance")
    col_x1, col_x2 = st.columns(2)
    
    with col_x1:
        imp_inj = pd.DataFrame({'Param': feature_cols, 'Importanza': xgb_inj.feature_importances_}).sort_values('Importanza')
        fig_inj = px.bar(imp_inj, x='Importanza', y='Param', orientation='h', color='Importanza')
        st.plotly_chart(fig_inj, use_container_width=True)
    
    with col_x2:
        imp_perf = pd.DataFrame({'Param': feature_cols, 'Importanza': xgb_perf.feature_importances_}).sort_values('Importanza')
        fig_perf = px.bar(imp_perf, x='Importanza', y='Param', orientation='h', color='Importanza')
        st.plotly_chart(fig_perf, use_container_width=True)
