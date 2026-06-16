import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="⌚ Apple Watch AI Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: bold;
    }
    .status-green { color: #2ecc71; font-weight: bold; }
    .status-yellow { color: #f39c12; font-weight: bold; }
    .status-red { color: #e74c3c; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# GENERAZIONE DATASET CON KPI PROPRIETARI
# ==========================================
@st.cache_data
def generate_training_data():
    """Genera 50 sessioni di corsa con KPI proprietari per l'allenamento"""
    np.random.seed(42)
    
    distanza = np.round(np.random.uniform(5.0, 14.0, 50), 1)
    rpe = np.random.randint(4, 11, 50)
    ore_sonno = np.round(np.random.uniform(5.0, 8.5, 50), 1)
    qualita_sonno = np.random.randint(1, 6, 50)
    ore_lavoro = np.random.randint(4, 11, 50)
    stress_mentale = np.random.randint(2, 11, 50)
    gradi = np.random.randint(18, 36, 50)
    vento = np.random.randint(2, 26, 50)
    
    df_train = pd.DataFrame({
        'Sessione': [f'Corsa_{i}' for i in range(1, 51)],
        'Distanza_Km': distanza,
        'RPE': rpe,
        'Ore_Sonno': ore_sonno,
        'Qualita_Sonno': qualita_sonno,
        'Ore_Lavoro': ore_lavoro,
        'Stress_Mentale': stress_mentale,
        'Gradi_Celsius': gradi,
        'Vento_Km_h': vento
    })
    
    # KPI Proprietari (Inventati dall'Atleta)
    df_train['ISMA'] = np.round((df_train['Stress_Mentale'] * df_train['RPE']) / df_train['Ore_Sonno'], 2)
    df_train['ISLR'] = np.round((df_train['Ore_Lavoro'] * df_train['Stress_Mentale']) / df_train['Distanza_Km'], 2)
    df_train['IITR'] = np.round((df_train['Gradi_Celsius'] * df_train['Vento_Km_h']) / df_train['Distanza_Km'], 2)
    
    # KPI aggiuntivi
    df_train['Load_Index'] = np.round((df_train['RPE'] * df_train['Distanza_Km']) / df_train['Ore_Sonno'], 2)
    df_train['Recovery_Score'] = np.round((df_train['Ore_Sonno'] * df_train['Qualita_Sonno']) / df_train['Stress_Mentale'], 2)
    df_train['Environmental_Stress'] = np.round((df_train['Gradi_Celsius'] + df_train['Vento_Km_h']) / 2, 1)
    
    return df_train

df_train = generate_training_data()

# ==========================================
# CALCOLO SOGLIE SCIENTIFICHE
# ==========================================
@st.cache_data
def compute_thresholds(df):
    """Calcola le soglie per ogni KPI"""
    thresholds = {}
    kpi_list = ['ISMA', 'ISLR', 'IITR', 'Load_Index', 'Recovery_Score']
    
    for kpi in kpi_list:
        media = df[kpi].mean()
        dev_std = df[kpi].std()
        
        thresholds[kpi] = {
            'media': np.round(media, 2),
            'dev_std': np.round(dev_std, 2),
            'soglia_gialla': np.round(media + (1.0 * dev_std), 2),
            'soglia_rossa': np.round(media + (1.5 * dev_std), 2)
        }
    
    return thresholds

thresholds = compute_thresholds(df_train)

def assegna_semaforo(kpi_value, kpi_name):
    """Assegna stato in base al valore KPI"""
    soglia_gialla = thresholds[kpi_name]['soglia_gialla']
    soglia_rossa = thresholds[kpi_name]['soglia_rossa']
    
    if kpi_value >= soglia_rossa:
        return '🔴 PERICOLO', '#e74c3c'
    elif kpi_value >= soglia_gialla:
        return '🟡 ATTENZIONE', '#f39c12'
    else:
        return '🟢 SICURO', '#2ecc71'

# Stato Atleta Generale
df_train['Stato_Atleta'] = df_train.apply(lambda row: assegna_semaforo(row['ISMA'], 'ISMA')[0], axis=1)

# ==========================================
# STATO SESSIONE
# ==========================================
if 'aw_connected' not in st.session_state:
    st.session_state.aw_connected = False

default_metrics = {
    'live_dist': 10.0, 'live_hrv': 55.0, 'live_rpe': 6, 'live_acwr': 1.2,
    'live_avg_bpm': 145.0, 'live_max_bpm': 170.0, 'live_cadence': 165.0, 
    'live_elev': 100.0, 'live_sleep': 7.5, 'live_vo2': 52.0, 'live_temp': 37.1,
    'live_calories': 680, 'live_stress': 4, 'live_recovery': 85
}

for key, val in default_metrics.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================================
# GENERAZIONE DATI APPLE WATCH
# ==========================================
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
        'RPE': np.random.randint(3, 10, days),
        'VO2_Max': np.random.normal(52, 5, days).clip(35, 70),
        'Body_Temp_C': np.random.normal(37.0, 0.3, days).clip(36.2, 38.5),
        'Calories': np.random.normal(650, 150, days).clip(200, 1200),
        'Stress_Level': np.random.randint(1, 10, days),
        'Recovery_Index': np.random.normal(80, 12, days).clip(20, 100),
        'Ore_Lavoro': np.random.randint(4, 11, days),
        'Qualita_Sonno': np.random.randint(1, 6, days),
        'Gradi_Celsius': np.random.randint(18, 36, days),
        'Vento_Km_h': np.random.randint(2, 26, days)
    })
    
    df['Daily_Load'] = df['RPE'] * df['Distance_km']
    df['Acute_Load_7D'] = df['Daily_Load'].rolling(window=7, min_periods=1).mean()
    df['Chronic_Load_28D'] = df['Daily_Load'].rolling(window=28, min_periods=1).mean()
    df['ACWR'] = (df['Acute_Load_7D'] / df['Chronic_Load_28D']).fillna(1.0)
    df['Pace_min_km'] = 9.0 - (df['Cadence_spm'] * 0.015) + (df['Avg_BPM'] * 0.005) + np.random.normal(0, 0.1, days)
    
    # Calcola KPI proprietari
    df['ISMA'] = np.round((df['Stress_Level'] * df['RPE']) / (df['Sleep_Hours'] + 0.1), 2)
    df['ISLR'] = np.round((df['Ore_Lavoro'] * df['Stress_Level']) / (df['Distance_km'] + 0.1), 2)
    df['IITR'] = np.round((df['Gradi_Celsius'] * df['Vento_Km_h']) / (df['Distance_km'] + 0.1), 2)
    df['Load_Index'] = np.round((df['RPE'] * df['Distance_km']) / (df['Sleep_Hours'] + 0.1), 2)
    df['Recovery_Score'] = np.round((df['Sleep_Hours'] * df['Qualita_Sonno']) / (df['Stress_Level'] + 0.1), 2)
    
    injury_prob = np.where((df['ACWR'] > 1.4) | (df['HRV_ms'] < 40) | (df['Sleep_Hours'] < 5.5), 0.75, 0.05)
    df['Injury_Event'] = np.random.binomial(1, injury_prob)
    
    return df.dropna()

df = load_data()

@st.cache_resource
def train_models(data):
    features = ['Distance_km', 'Avg_BPM', 'Max_BPM', 'Cadence_spm', 'Elevation_m', 
                'Sleep_Hours', 'HRV_ms', 'RPE', 'ACWR', 'VO2_Max', 'Body_Temp_C', 
                'Stress_Level', 'ISMA', 'ISLR', 'IITR', 'Load_Index', 'Recovery_Score']
    X = data[features]
    
    xgb_perf = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    xgb_perf.fit(X, data['Pace_min_km'])
    
    xgb_inj = XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss', random_state=42)
    xgb_inj.fit(X, data['Injury_Event'])
    
    return xgb_perf, xgb_inj, features

xgb_perf, xgb_inj, feature_cols = train_models(df)

# ==========================================
# TITLE & HERO SECTION
# ==========================================
col_hero1, col_hero2 = st.columns([3, 1])
with col_hero1:
    st.title("⌚ Apple Watch AI Analytics + KPI Proprietari")
    st.markdown("### Monitoraggio Avanzato con KPI Inventati (ISMA, ISLR, IITR)")
    
with col_hero2:
    st.metric("Dataset Giorni", f"{len(df)}", "ultimi 600gg")

st.divider()

# ==========================================
# SIDEBAR - SINCRONIZZAZIONE
# ==========================================
st.sidebar.title("🔗 SINCRONIZZAZIONE DISPOSITIVO")
st.sidebar.markdown("---")

if not st.session_state.aw_connected:
    st.sidebar.warning("⏳ Apple Watch non sincronizzato")
    
    if st.sidebar.button("📡 SINCRONIZZA APPLE WATCH", use_container_width=True, key="sync_btn"):
        progress = st.sidebar.progress(0)
        status = st.sidebar.status("Sincronizzazione in corso...", expanded=True)
        
        steps = [
            ("🔍 Ricerca dispositivi BLE", 0.2),
            ("🔐 Autenticazione iCloud", 0.4),
            ("📥 Estrazione Health Data", 0.6),
            ("⚙️ Elaborazione KPI Proprietari", 0.8),
            ("✅ Sincronizzazione completata!", 1.0)
        ]
        
        import time
        for step, pct in steps:
            status.write(step)
            progress.progress(pct)
            time.sleep(0.5)
        
        st.session_state.aw_connected = True
        st.session_state.live_dist = np.random.normal(13, 2).clip(8, 20)
        st.session_state.live_avg_bpm = np.random.normal(152, 10).clip(130, 180)
        st.session_state.live_max_bpm = np.random.normal(175, 12).clip(150, 200)
        st.session_state.live_cadence = np.random.normal(170, 8).clip(155, 190)
        st.session_state.live_elev = np.random.normal(180, 50).clip(50, 400)
        st.session_state.live_sleep = np.random.normal(6.8, 1).clip(4.5, 9)
        st.session_state.live_hrv = np.random.normal(48, 12).clip(25, 80)
        st.session_state.live_acwr = np.random.normal(1.35, 0.25).clip(0.8, 2.0)
        st.session_state.live_rpe = np.random.randint(5, 9)
        st.session_state.live_vo2 = np.random.normal(50, 5).clip(40, 65)
        st.session_state.live_temp = np.random.normal(37.0, 0.2).clip(36.5, 37.5)
        st.session_state.live_calories = np.random.normal(720, 100).clip(500, 1000)
        st.session_state.live_stress = np.random.randint(2, 8)
        st.session_state.live_recovery = np.random.normal(82, 10).clip(50, 100)
        
        status.success("Sincronizzazione conclusa! ✨")
        st.rerun()

else:
    st.sidebar.success("✅ Apple Watch Connesso")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📊 Parametri Base")
    
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        st.metric("❤️ Freq. Cardiaca", f"{st.session_state.live_avg_bpm:.0f} bpm")
        st.metric("💤 Sonno", f"{st.session_state.live_sleep:.1f} h")
        st.metric("💓 HRV", f"{st.session_state.live_hrv:.0f} ms")
    
    with col_s2:
        st.metric("👟 Cadenza", f"{st.session_state.live_cadence:.0f} spm")
        st.metric("📈 ACWR", f"{st.session_state.live_acwr:.2f}")
        st.metric("🫁 VO2 Max", f"{st.session_state.live_vo2:.0f}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 KPI Proprietari")
    
    # Calcola KPI in tempo reale
    live_isma = np.round((st.session_state.live_stress * st.session_state.live_rpe) / st.session_state.live_sleep, 2)
    live_islr = np.round((5 * st.session_state.live_stress) / st.session_state.live_dist, 2)  # 5 ore lavoro simulate
    live_iitr = np.round((25 * 10) / st.session_state.live_dist, 2)  # temp/vento simulati
    
    col_kpi1, col_kpi2 = st.sidebar.columns(2)
    with col_kpi1:
        stato_isma, colore_isma = assegna_semaforo(live_isma, 'ISMA')
        st.markdown(f"**ISMA:** <span class='status-{colore_isma.replace('#', '')}'>{stato_isma}</span>", unsafe_allow_html=True)
        st.metric("Valore", live_isma, f"Soglia: {thresholds['ISMA']['soglia_rossa']}")
        
        stato_islr, colore_islr = assegna_semaforo(live_islr, 'ISLR')
        st.markdown(f"**ISLR:** <span class='status-{colore_islr.replace('#', '')}'>{stato_islr}</span>", unsafe_allow_html=True)
        st.metric("Valore", live_islr)
    
    with col_kpi2:
        stato_iitr, colore_iitr = assegna_semaforo(live_iitr, 'IITR')
        st.markdown(f"**IITR:** <span class='status-{colore_iitr.replace('#', '')}'>{stato_iitr}</span>", unsafe_allow_html=True)
        st.metric("Valore", live_iitr)
        
        stato_load, colore_load = assegna_semaforo(live_isma * 0.8, 'Load_Index')
        st.markdown(f"**Load Index:** <span class='status-{colore_load.replace('#', '')}'>{stato_load}</span>", unsafe_allow_html=True)
        st.metric("Valore", np.round(live_isma * 0.8, 2))
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🔌 SCOLLEGA DISPOSITIVO", use_container_width=True, key="disconnect_btn"):
        st.session_state.aw_connected = False
        st.rerun()

# ==========================================
# TABS PRINCIPALI
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 PREVISIONE", "📊 KPI PROPRIETARI", "📈 ANALISI", "🧬 FEATURES", "📋 DATABASE"])

# ==========================================
# TAB 1: PREVISIONE
# ==========================================
with tab1:
    st.header("🚀 Motore di Previsione IA")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: in_dist = st.number_input("🏃 Distanza (km)", value=float(st.session_state.live_dist), min_value=1.0, max_value=50.0)
    with col2: in_rpe = st.slider("💪 RPE", 1, 10, int(st.session_state.live_rpe))
    with col3: in_sleep = st.number_input("💤 Sonno (h)", value=float(st.session_state.live_sleep), min_value=2.0, max_value=12.0)
    with col4: in_stress = st.slider("😰 Stress", 1, 10, int(st.session_state.live_stress))
    
    col5, col6, col7, col8 = st.columns(4)
    with col5: in_avg_bpm = st.number_input("❤️ Avg BPM", value=float(st.session_state.live_avg_bpm), min_value=60.0, max_value=200.0)
    with col6: in_max_bpm = st.number_input("❤️ Max BPM", value=float(st.session_state.live_max_bpm), min_value=100.0, max_value=220.0)
    with col7: in_hrv = st.number_input("💓 HRV (ms)", value=float(st.session_state.live_hrv), min_value=10.0, max_value=150.0)
    with col8: in_acwr = st.number_input("📈 ACWR", value=float(st.session_state.live_acwr), min_value=0.5, max_value=3.0)
    
    col9, col10, col11, col12 = st.columns(4)
    with col9: in_cadence = st.number_input("👟 Cadenza (spm)", value=float(st.session_state.live_cadence), min_value=140.0, max_value=200.0)
    with col10: in_elev = st.number_input("⬆️ Dislivello (m)", value=float(st.session_state.live_elev), min_value=0.0, max_value=2000.0)
    with col11: in_vo2 = st.number_input("🫁 VO2 Max", value=float(st.session_state.live_vo2), min_value=20.0, max_value=80.0)
    with col12: in_temp = st.number_input("🌡️ Temp (°C)", value=float(st.session_state.live_temp), min_value=36.0, max_value=39.0)
    
    col13, col14 = st.columns(2)
    with col13:
        in_ore_lavoro = st.slider("💼 Ore Lavoro", 4, 10, 5)
        in_qualita_sonno = st.slider("😴 Qualità Sonno", 1, 5, 3)
    with col14:
        in_gradi = st.slider("🌡️ Gradi Celsius", 18, 36, 25)
        in_vento = st.slider("💨 Vento (km/h)", 2, 26, 10)
    
    if st.button("🚀 PREVISIONE IA CON KPI PROPRIETARI", use_container_width=True):
        input_df = pd.DataFrame({
            'Distance_km': [in_dist], 'Avg_BPM': [in_avg_bpm],
            'Max_BPM': [in_max_bpm], 'Cadence_spm': [in_cadence],
            'Elevation_m': [in_elev], 'Sleep_Hours': [in_sleep],
            'HRV_ms': [in_hrv], 'RPE': [in_rpe], 'ACWR': [in_acwr],
            'VO2_Max': [in_vo2], 'Body_Temp_C': [in_temp], 'Stress_Level': [in_stress],
            'ISMA': [np.round((in_stress * in_rpe) / in_sleep, 2)],
            'ISLR': [np.round((in_ore_lavoro * in_stress) / (in_dist + 0.1), 2)],
            'IITR': [np.round((in_gradi * in_vento) / (in_dist + 0.1), 2)],
            'Load_Index': [np.round((in_rpe * in_dist) / in_sleep, 2)],
            'Recovery_Score': [np.round((in_sleep * in_qualita_sonno) / in_stress, 2)]
        })
        
        pred_pace = xgb_perf.predict(input_df)[0]
        pred_risk = xgb_inj.predict_proba(input_df)[0][1] * 100
        
        st.divider()
        st.subheader("📊 RISULTATI")
        
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1: st.metric("⏱️ Pace", f"{pred_pace:.2f} min/km")
        with res_col2: st.metric("⚠️ Rischio Infortuni", f"{pred_risk:.1f}%")
        with res_col3: st.metric("🔥 Calorie Est.", f"{int(in_dist * 90 + in_rpe * 50)}")
        with res_col4: st.metric("⏰ Tempo Est.", f"{int((in_dist / (60/pred_pace)))}m")

# ==========================================
# TAB 2: KPI PROPRIETARI
# ==========================================
with tab2:
    st.header("🎯 KPI Proprietari - Soglie Scientifiche")
    
    col_kpi1, col_kpi2 = st.columns(2)
    
    with col_kpi1:
        st.subheader("📊 ISMA - Indice Stress/Sonno/RPE")
        st.markdown("**Formula:** (Stress × RPE) / Ore_Sonno")
        st.info("""
        **Interpretazione:**
        - 🟢 **SICURO** (Verde): < 14.50 - Atleta in condizioni ottimali
        - 🟡 **ATTENZIONE** (Giallo): 14.50-21.75 - Fatica elevata, monitora
        - 🔴 **PERICOLO** (Rosso): > 21.75 - RISCHIO INFORTUNIO ALTO
        """)
        
        fig_isma = px.box(df_train, y='ISMA', title="Distribuzione ISMA nel Dataset",
                         color_discrete_sequence=['#3498db'])
        fig_isma.add_hline(y=thresholds['ISMA']['soglia_gialla'], line_dash="dash", 
                          line_color="orange", annotation_text="Soglia Gialla")
        fig_isma.add_hline(y=thresholds['ISMA']['soglia_rossa'], line_dash="dash", 
                          line_color="red", annotation_text="Soglia Rossa")
        fig_isma.update_layout(height=400)
        st.plotly_chart(fig_isma, use_container_width=True)
    
    with col_kpi2:
        st.subheader("📊 ISLR - Indice Stress Lavoro/Distanza")
        st.markdown("**Formula:** (Ore_Lavoro × Stress) / Distanza")
        st.info("""
        **Interpretazione:**
        - 🟢 **SICURO** (Verde): < 3.44 - Equilibrio lavoro/corsa
        - 🟡 **ATTENZIONE** (Giallo): 3.44-5.16 - Carico elevato
        - 🔴 **PERICOLO** (Rosso): > 5.16 - CROLLO PERFORMANCE
        """)
        
        fig_islr = px.box(df_train, y='ISLR', title="Distribuzione ISLR nel Dataset",
                         color_discrete_sequence=['#9b59b6'])
        fig_islr.add_hline(y=thresholds['ISLR']['soglia_gialla'], line_dash="dash", 
                          line_color="orange", annotation_text="Soglia Gialla")
        fig_islr.add_hline(y=thresholds['ISLR']['soglia_rossa'], line_dash="dash", 
                          line_color="red", annotation_text="Soglia Rossa")
        fig_islr.update_layout(height=400)
        st.plotly_chart(fig_islr, use_container_width=True)
    
    col_kpi3, col_kpi4 = st.columns(2)
    
    with col_kpi3:
        st.subheader("📊 IITR - Indice Stress Ambientale/Distanza")
        st.markdown("**Formula:** (Gradi × Vento) / Distanza")
        st.info("""
        **Interpretazione:**
        - 🟢 **SICURO** (Verde): < 4.37 - Condizioni favorevoli
        - 🟡 **ATTENZIONE** (Giallo): 4.37-6.56 - Stress termico moderato
        - 🔴 **PERICOLO** (Rosso): > 6.56 - STRESS TERMICO ELEVATO
        """)
        
        fig_iitr = px.box(df_train, y='IITR', title="Distribuzione IITR nel Dataset",
                         color_discrete_sequence=['#e74c3c'])
        fig_iitr.add_hline(y=thresholds['IITR']['soglia_gialla'], line_dash="dash", 
                          line_color="orange", annotation_text="Soglia Gialla")
        fig_iitr.add_hline(y=thresholds['IITR']['soglia_rossa'], line_dash="dash", 
                          line_color="red", annotation_text="Soglia Rossa")
        fig_iitr.update_layout(height=400)
        st.plotly_chart(fig_iitr, use_container_width=True)
    
    with col_kpi4:
        st.subheader("📊 Load Index & Recovery Score")
        
        fig_load = px.histogram(df_train, x='Load_Index', nbins=20,
                               title="Load Index Distribution",
                               color_discrete_sequence=['#f39c12'])
        fig_load.update_layout(height=400)
        st.plotly_chart(fig_load, use_container_width=True)
    
    # Tabella Soglie Scientifiche
    st.divider()
    st.subheader("📋 Tabella Soglie Scientifiche")
    
    threshold_data = []
    for kpi, vals in thresholds.items():
        threshold_data.append({
            'KPI': kpi,
            'Media': vals['media'],
            'Dev. Std': vals['dev_std'],
            'Soglia Gialla': vals['soglia_gialla'],
            'Soglia Rossa': vals['soglia_rossa']
        })
    
    st.dataframe(pd.DataFrame(threshold_data), use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: ANALISI STORICA
# ==========================================
with tab3:
    st.header("📈 Analisi Storica con KPI")
    
    col_an1, col_an2 = st.columns(2)
    
    with col_an1:
        fig_trend = px.line(df.tail(100), x='Date', y=['ISMA', 'ISLR'],
                           title="Trend KPI Proprietari (ultimi 100gg)",
                           markers=True)
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_an2:
        fig_load_trend = px.line(df.tail(100), x='Date', y=['Load_Index', 'Recovery_Score'],
                                title="Load Index vs Recovery (ultimi 100gg)",
                                markers=True)
        fig_load_trend.update_layout(height=400)
        st.plotly_chart(fig_load_trend, use_container_width=True)
    
    col_an3, col_an4 = st.columns(2)
    
    with col_an3:
        fig_isma_dist = px.scatter(df, x='Distance_km', y='ISMA', color='Injury_Event',
                                  title="ISMA vs Distanza",
                                  color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        fig_isma_dist.update_layout(height=400)
        st.plotly_chart(fig_isma_dist, use_container_width=True)
    
    with col_an4:
        fig_kpi_injury = px.box(df, x='Injury_Event', y='ISMA',
                               title="ISMA per Infortuni",
                               color_discrete_sequence=['#3498db'])
        fig_kpi_injury.update_layout(height=400)
        st.plotly_chart(fig_kpi_injury, use_container_width=True)

# ==========================================
# TAB 4: FEATURE IMPORTANCE
# ==========================================
with tab4:
    st.header("🧬 Feature Importance (XGBoost)")
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        imp_inj = pd.DataFrame({'Feature': feature_cols, 'Importanza': xgb_inj.feature_importances_}).sort_values('Importanza', ascending=True)
        fig_inj = px.bar(imp_inj, x='Importanza', y='Feature', orientation='h',
                        color='Importanza', color_continuous_scale='Reds')
        fig_inj.update_layout(height=500)
        st.plotly_chart(fig_inj, use_container_width=True)
    
    with col_feat2:
        imp_perf = pd.DataFrame({'Feature': feature_cols, 'Importanza': xgb_perf.feature_importances_}).sort_values('Importanza', ascending=True)
        fig_perf = px.bar(imp_perf, x='Importanza', y='Feature', orientation='h',
                         color='Importanza', color_continuous_scale='Blues')
        fig_perf.update_layout(height=500)
        st.plotly_chart(fig_perf, use_container_width=True)

# ==========================================
# TAB 5: DATABASE COMPLETO
# ==========================================
with tab5:
    st.header("📋 Database Allenamenti - 50 Sessioni")
    
    st.subheader("Dati di Allenamento di Riferimento")
    display_cols = ['Sessione', 'Distanza_Km', 'RPE', 'Ore_Sonno', 'ISMA', 'ISLR', 'IITR', 'Stato_Atleta']
    st.dataframe(df_train[display_cols], use_container_width=True, height=400)
    
    st.divider()
    
    st.subheader("Download Dataset")
    csv = df_train.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV (50 sessioni)",
        data=csv,
        file_name="database_50_corse_kpi.csv",
        mime="text/csv"
    )
    
    st.divider()
    
    st.subheader("Dataset Apple Watch (600 giorni)")
    st.dataframe(df[['Date', 'Distance_km', 'ISMA', 'ISLR', 'IITR', 'Load_Index', 'Recovery_Score']].tail(30),
                use_container_width=True, height=400)

st.divider()
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; margin-top: 30px;'>
    <p><b>⌚ Apple Watch AI Analytics + KPI Proprietari</b> | Powered by XGBoost & Streamlit</p>
    <p>KPI Inventati: ISMA, ISLR, IITR - Formule Proprietarie per Analisi Atleta</p>
    </div>
    """, unsafe_allow_html=True)
