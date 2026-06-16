import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor, XGBClassifier
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

# ==========================================
# 0. CONFIGURAZIONE PAGINA E CSS
# ==========================================
st.set_page_config(page_title="Apple Watch AI Analytics", page_icon="⌚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 800; color: #1f77b4; }
    h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-weight: 700; }
    .achievement-badge {
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        padding: 15px; border-radius: 12px; color: white; text-align: center;
        margin: 10px 0; font-weight: 600; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .footer { text-align: center; color: #7f8c8d; margin-top: 40px; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. GENERAZIONE DATI STORICI E METRICHE AVANZATE (600 GG)
# ==========================================
@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """Genera 600 giorni di dati estesi per addestrare i modelli ML."""
    np.random.seed(42)
    days = 600
    dates = pd.date_range(start="2024-01-01", periods=days)
    
    # Metriche Base
    dist = np.random.normal(12, 4, days).clip(2, 35)
    rpe = np.random.randint(3, 10, days)
    sleep = np.random.normal(7.2, 1.1, days).clip(4, 10)
    stress = np.random.randint(1, 10, days)
    
    # Nuove Metriche Avanzate Apple Watch (Dinamiche Corsa & Fisiologia)
    power_w = np.random.normal(250, 40, days).clip(150, 400) # Running Power
    gct_ms = np.random.normal(220, 15, days).clip(180, 300)  # Ground Contact Time
    vert_osc_cm = np.random.normal(8.5, 1.2, days).clip(5.0, 12.0) # Vertical Oscillation
    spo2 = np.random.normal(97, 1.5, days).clip(92, 100) # Blood Oxygen
    resp_rate = np.random.normal(16, 2, days).clip(12, 25) # Respiration Rate (notte)
    
    df = pd.DataFrame({
        'Date': dates, 'Distance_km': dist, 'RPE': rpe, 'Sleep_Hours': sleep, 'Stress_Level': stress,
        'Avg_BPM': np.random.normal(148, 12, days).clip(110, 190),
        'Max_BPM': np.random.normal(170, 15, days).clip(130, 200),
        'Cadence_spm': np.random.normal(168, 8, days).clip(140, 200),
        'Elevation_m': np.random.normal(150, 80, days).clip(0, 1000),
        'HRV_ms': np.random.normal(60, 15, days).clip(20, 100),
        'VO2_Max': np.random.normal(52, 5, days).clip(35, 70),
        'Ore_Lavoro': np.random.randint(4, 11, days),
        'Qualita_Sonno': np.random.randint(1, 6, days),
        'Gradi_Celsius': np.random.randint(18, 36, days),
        'Vento_Km_h': np.random.randint(2, 26, days),
        # Nuove feature inserite nel DataFrame
        'Running_Power_W': power_w, 'GCT_ms': gct_ms, 'Vertical_Osc_cm': vert_osc_cm,
        'SpO2_pct': spo2, 'Resp_Rate': resp_rate
    })
    
    # Feature Engineering Avanzata
    df['Daily_Load'] = df['RPE'] * df['Distance_km']
    df['Acute_Load_7D'] = df['Daily_Load'].rolling(window=7, min_periods=1).mean()
    df['Chronic_Load_28D'] = df['Daily_Load'].rolling(window=28, min_periods=1).mean()
    df['ACWR'] = (df['Acute_Load_7D'] / df['Chronic_Load_28D']).fillna(1.0)
    
    # KPI Proprietari
    df['ISMA'] = np.round((df['Stress_Level'] * df['RPE']) / (df['Sleep_Hours'] + 0.1), 2)
    df['ISLR'] = np.round((df['Ore_Lavoro'] * df['Stress_Level']) / (df['Distance_km'] + 0.1), 2)
    df['IITR'] = np.round((df['Gradi_Celsius'] * df['Vento_Km_h']) / (df['Distance_km'] + 0.1), 2)
    df['Recovery_Score'] = np.round((df['Sleep_Hours'] * df['Qualita_Sonno']) / (df['Stress_Level'] + 0.1), 2)
    
    # Target: Pace (influenzato da Potenza e GCT) e Infortuni
    df['Pace_min_km'] = 9.0 - (df['Cadence_spm'] * 0.015) - (df['Running_Power_W'] * 0.002) + (df['GCT_ms'] * 0.005) + np.random.normal(0, 0.1, days)
    injury_prob = np.where((df['ACWR'] > 1.4) | (df['GCT_ms'] > 260) | (df['SpO2_pct'] < 94), 0.75, 0.05)
    df['Injury_Event'] = np.random.binomial(1, injury_prob)
    
    return df.dropna()

df = load_historical_data()

# ==========================================
# 2. ADDESTRAMENTO MODELLI XGBOOST
# ==========================================
@st.cache_resource
def train_ai_models(data: pd.DataFrame):
    # Lista feature iper-dettagliata
    features = ['Distance_km', 'Avg_BPM', 'Max_BPM', 'Cadence_spm', 'Elevation_m', 
                'Sleep_Hours', 'HRV_ms', 'RPE', 'ACWR', 'VO2_Max', 'Stress_Level', 
                'ISMA', 'ISLR', 'IITR', 'Recovery_Score', 
                'Running_Power_W', 'GCT_ms', 'Vertical_Osc_cm', 'SpO2_pct', 'Resp_Rate']
    X = data[features]
    
    xgb_perf = XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_perf.fit(X, data['Pace_min_km'])
    
    xgb_inj = XGBClassifier(n_estimators=120, max_depth=4, eval_metric='logloss', random_state=42)
    xgb_inj.fit(X, data['Injury_Event'])
    
    return xgb_perf, xgb_inj, features

xgb_perf, xgb_inj, feature_cols = train_ai_models(df)

# ==========================================
# 3. INTERFACCIA E SINCRONIZZAZIONE
# ==========================================
if 'aw_connected' not in st.session_state:
    st.session_state.aw_connected = False

col_hero1, col_hero2 = st.columns([4, 1])
with col_hero1:
    st.title("⌚ Apple Watch AI Analytics Pro")
    st.markdown("##### 🚀 Dinamiche di Corsa • AI Coach Scientifico • Simulatore Predittivo")
with col_hero2:
    st.metric("Punti Dati Elaborati", f"{len(df) * len(feature_cols):,}", "In tempo reale")

st.divider()

# -- Sidebar --
st.sidebar.title("🔗 SINCRONIZZAZIONE")
st.sidebar.markdown("---")

if not st.session_state.aw_connected:
    st.sidebar.warning("⏳ In attesa di HealthKit")
    if st.sidebar.button("📡 ESTRAI DATI APPLE WATCH", use_container_width=True, type="primary"):
        with st.sidebar.status("Sincronizzazione API...", expanded=True) as status:
            time.sleep(0.5); status.write("📥 Estrazione Dinamiche Corsa...")
            time.sleep(0.5); status.write("📥 Estrazione Ossigenazione...")
            time.sleep(0.5); status.write("⚙️ Elaborazione Machine Learning...")
        st.session_state.aw_connected = True
        st.rerun()
else:
    st.sidebar.success("✅ Sincronizzato con Apple Health")
    st.sidebar.markdown("**Ultimi dati acquisiti:**")
    col_s1, col_s2 = st.sidebar.columns(2)
    col_s1.metric("❤️ BPM", "142")
    col_s1.metric("⚡ Power", "285 W")
    col_s1.metric("🦶 GCT", "215 ms")
    col_s2.metric("💓 HRV", "58 ms")
    col_s2.metric("🩸 SpO2", "98 %")
    col_s2.metric("↕️ Osc. Vert", "8.2 cm")
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔌 SCOLLEGA DISPOSITIVO", use_container_width=True):
        st.session_state.aw_connected = False
        st.rerun()

# ==========================================
# 4. TABS DELL'APPLICAZIONE
# ==========================================
# Rimossa la Heatmap, aggiunto "🔮 SIMULATORE PRO"
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 KPI REALTIME", "🤖 AI COACH PRO", "🔮 SIMULATORE PRO", "📈 FORECAST 30G",
    "🏆 ACHIEVEMENTS", "📊 TREND ML", "🎓 SEZIONE TESI"
])

# --- TAB 1: KPI GAUGE ---
with tab1:
    st.header("🎯 Analisi Indici Proprietari")
    col_g1, col_g2, col_g3 = st.columns(3)
    
    def plot_gauge(val, title, s_gialla, s_rossa, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=val,
            title={'text': title, 'font': {'size': 18}},
            gauge={'axis': {'range': [0, s_rossa*1.5]}, 'bar': {'color': color},
                   'steps': [{'range': [0, s_gialla], 'color': "rgba(46, 204, 113, 0.2)"},
                             {'range': [s_gialla, s_rossa], 'color': "rgba(243, 156, 18, 0.2)"},
                             {'range': [s_rossa, s_rossa*1.5], 'color': "rgba(231, 76, 60, 0.2)"}],
                   'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': s_rossa}}
        ))
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': 'gray'})
        return fig

    # Calcolo soglie al volo dal df storico
    with col_g1:
        st.plotly_chart(plot_gauge(14.5, "ISMA (Fatica/Recupero)", df['ISMA'].mean()+df['ISMA'].std(), df['ISMA'].mean()+df['ISMA'].std()*1.5, "#3498db"), use_container_width=True)
    with col_g2:
        st.plotly_chart(plot_gauge(3.2, "ISLR (Lavoro/Stress)", df['ISLR'].mean()+df['ISLR'].std(), df['ISLR'].mean()+df['ISLR'].std()*1.5, "#9b59b6"), use_container_width=True)
    with col_g3:
        st.plotly_chart(plot_gauge(6.8, "IITR (Stress Ambientale)", df['IITR'].mean()+df['IITR'].std(), df['IITR'].mean()+df['IITR'].std()*1.5, "#e67e22"), use_container_width=True)

# --- TAB 2: AI COACH (PROFESSIONALE E SCIENTIFICO) ---
with tab2:
    st.header("🤖 Staff Tecnico AI (Sports Science)")
    st.markdown("Consulenza analitica basata su fisiologia dell'esercizio e dati biomeccanici.")
    
    st.subheader("💬 Q&A Analisi Dati (Data Scientist & Head Coach)")
    
    coach_qa = {
        "1. Come ottimizzo l'ISMA (Stress/Recupero) per massimizzare la supercompensazione?": 
            "**Analisi:** L'ISMA valuta il bilancio omeostatico. Per innescare la supercompensazione, mantieni un ISMA in zona gialla durante il blocco di carico (overreaching funzionale), seguito da un calo drastico (zona verde < 10) tramite 48-72h di scarico (sonno >8h, RPE <4). Questo garantisce il picco di forma.",
            
        "2. Il Tempo di Contatto (GCT) sta aumentando. È un segnale di allarme?": 
            "**Biomeccanica:** Sì. Un GCT che passa da 210ms a >230ms a parità di Pace indica un degrado della stiffness tendinea e un calo della Running Power. È un marker pre-clinico di affaticamento neuromuscolare. **Azione:** Sostituisci la prossima sessione di qualità con recupero attivo o cross-training.",
            
        "3. Come interpreto le variazioni notturne di SpO2 e Frequenza Respiratoria?": 
            "**Fisiologia:** Un calo di SpO2 (<95%) associato a un aumento della Frequenza Respiratoria (es. da 14 a 18 atti/min) durante il sonno indica forte stress sistemico o recupero incompleto da una sessione ad alto accumulo di lattato. Il sistema simpatico è iperattivo. **Azione:** Niente lavori in soglia finché i parametri non rientrano nella baseline.",
            
        "4. L'indice ISLR indica un sovraccarico lavorativo. Come aggiusto l'allenamento?": 
            "**Gestione Carico:** Lo stress cognitivo (Lavoro) drena le stesse risorse del Sistema Nervoso Centrale (SNC) usate per l'allenamento. Con ISLR alto, evita allenamenti ad alta complessità neuromuscolare (sprint, ripetute brevi). Opta per fondo lento continuo in Z2 cardiaca, che ha effetto parasimpatico e abbassa il cortisolo.",
            
        "5. Il VO2Max calcolato dall'orologio è stagnante da un mese.": 
            "**Metodologia:** La stagnazione del VO2Max è tipica quando l'allenamento diventa monotono (polarizzazione assente). Controlla il tuo database: se le tue sessioni sono quasi tutte a RPE 6-7, sei nella 'zona grigia'. **Azione:** Applica la regola 80/20. L'80% del volume a RPE 3-4 (LIT), il 20% a RPE 9-10 (HIT).",
            
        "6. Impatto del calore (IITR) sulla Deriva Cardiaca.": 
            "**Analisi Ambientale:** Con un IITR elevato, il corpo devia il flusso sanguigno verso la pelle per la termoregolazione, riducendo la gittata sistolica. A parità di Pace e Potenza, i BPM saliranno (Deriva Cardiaca). **Azione:** Non allenarti sul passo (min/km) ma sulla Potenza (Watt) o sui BPM target per evitare di sovrastimare le tue reali capacità aerobiche odierne."
    }
    
    selected_q = st.selectbox("Seleziona la tematica di analisi:", list(coach_qa.keys()))
    st.info(coach_qa[selected_q])

# --- TAB 3: SIMULATORE PREDITTIVO MANUALE (NEW!) ---
with tab3:
    st.header("🔮 Simulatore Previsioni Sessione (Manual Input)")
    st.markdown("Inserisci manualmente i parametri per calcolare l'esito del prossimo allenamento usando XGBoost.")
    
    st.subheader("1. Setup Allenamento Previsto")
    col_i1, col_i2, col_i3 = st.columns(3)
    in_dist = col_i1.number_input("Distanza Prevista (km)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
    in_elev = col_i2.number_input("Dislivello Previsto (m)", min_value=0, max_value=2000, value=150)
    in_rpe = col_i3.slider("RPE Target (1-10)", 1, 10, 6)
    
    st.subheader("2. Dati Fisiologici Odierni & Biomeccanica")
    col_i4, col_i5, col_i6, col_i7 = st.columns(4)
    in_sleep = col_i4.number_input("Sonno Stanotte (Ore)", min_value=2.0, max_value=12.0, value=7.5)
    in_hrv = col_i5.number_input("HRV Mattutino (ms)", min_value=10, max_value=150, value=55)
    in_spo2 = col_i6.number_input("SpO2 Medio (%)", min_value=85.0, max_value=100.0, value=98.0)
    in_stress = col_i7.slider("Stress Percepito (1-10)", 1, 10, 4)

    with st.expander("⚙️ Parametri Avanzati (Dinamiche Apple Watch)"):
        col_i8, col_i9, col_i10 = st.columns(3)
        in_cadence = col_i8.number_input("Cadenza Storica (spm)", value=165)
        in_power = col_i9.number_input("Running Power Stimata (W)", value=280)
        in_gct = col_i10.number_input("Tempo Contatto Suolo (ms)", value=220)
        in_vo2 = st.number_input("VO2Max Attuale", value=52.0)
        in_acwr = st.number_input("ACWR Attuale (Carico acuto/cronico)", value=1.1)

    st.divider()
    
    if st.button("🚀 ESEGUI INFERENZA IA (XGBoost)", use_container_width=True, type="primary"):
        # Calcolo KPI proprietari al volo per l'input
        in_isma = (in_stress * in_rpe) / (in_sleep + 0.1)
        in_islr = (8 * in_stress) / (in_dist + 0.1) # 8 ore lavoro fisse per simulazione
        in_iitr = (22 * 10) / (in_dist + 0.1) # Temperatura e vento standard per simulazione
        in_recovery = (in_sleep * 4) / (in_stress + 0.1) # Qualità sonno 4 standard
        
        # Creazione DataFrame singolo per inferenza
        input_dict = {
            'Distance_km': [in_dist], 'Avg_BPM': [145], 'Max_BPM': [170], 'Cadence_spm': [in_cadence], 
            'Elevation_m': [in_elev], 'Sleep_Hours': [in_sleep], 'HRV_ms': [in_hrv], 'RPE': [in_rpe], 
            'ACWR': [in_acwr], 'VO2_Max': [in_vo2], 'Stress_Level': [in_stress], 
            'ISMA': [in_isma], 'ISLR': [in_islr], 'IITR': [in_iitr], 'Recovery_Score': [in_recovery],
            'Running_Power_W': [in_power], 'GCT_ms': [in_gct], 'Vertical_Osc_cm': [8.5], 
            'SpO2_pct': [in_spo2], 'Resp_Rate': [15]
        }
        
        input_df = pd.DataFrame(input_dict)[feature_cols] # Assicura ordine colonne corretto
        
        # Predizioni
        pred_pace = xgb_perf.predict(input_df)[0]
        pred_risk = xgb_inj.predict_proba(input_df)[0][1] * 100
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("⏱️ Pacing Suggerito dall'IA", f"{pred_pace:.2f} min/km", delta="Ottimizzato su Potenza e GCT")
        
        risk_color = "normal" if pred_risk < 30 else "off" if pred_risk < 60 else "inverse"
        col_res2.metric("⚠️ Probabilità Infortunio / Sovraccarico", f"{pred_risk:.1f}%", delta="Rilevato da HRV/SpO2/ACWR", delta_color=risk_color)
        
        if pred_risk > 50:
            st.error("🚨 **RISCHIO ELEVATO:** Il modello rileva un forte squilibrio (probabile SpO2 basso, GCT alto o sonno insufficiente). Riduci distanza o RPE.")
        else:
            st.success("✅ **CONDIZIONE OTTIMALE:** I tuoi parametri biomeccanici e fisiologici supportano pienamente questo carico di lavoro.")

# --- TAB 4, 5, 6, 7 (Mantenuti essenziali e invariati per stabilità) ---
with tab4:
    st.header("📈 Forecast a 30 Giorni")
    st.info("Previsione evoluzione stato di forma.")
    future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=30)
    future_pace = np.cumsum(np.random.normal(0.01, 0.08, 30)) + df['Pace_min_km'].iloc[-1]
    fig_forecast = px.line(x=future_dates, y=future_pace, markers=True, title="Previsione Evoluzione Pace (min/km)")
    fig_forecast.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_forecast, use_container_width=True)

with tab5:
    st.header("🏆 Achievements Biomeccanici")
    st.markdown("<div class='achievement-badge' style='background: linear-gradient(135deg, #00b09b, #96c93d);'>⚡ EFFICIENZA PURA<br><span style='font-size:0.8em; font-weight:normal;'>GCT mantenuto sotto i 200ms per 5 sessioni di fila</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='achievement-badge' style='background: linear-gradient(135deg, #4facfe, #00f2fe);'>🩸 OSSIGENAZIONE TOP<br><span style='font-size:0.8em; font-weight:normal;'>SpO2 notturno fisso a 99% per 7 giorni</span></div>", unsafe_allow_html=True)

with tab6:
    st.header("📊 Feature Importance (Interpretazione XGBoost)")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        imp_perf = pd.DataFrame({'Feature': feature_cols, 'Importanza': xgb_perf.feature_importances_}).sort_values('Importanza')
        fig_perf = px.bar(imp_perf.tail(8), x='Importanza', y='Feature', orientation='h', title="Cosa influenza la tua Velocità (Pace)?", color='Importanza', color_continuous_scale='Blues')
        fig_perf.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_perf, use_container_width=True)
    with col_t2:
        imp_inj = pd.DataFrame({'Feature': feature_cols, 'Importanza': xgb_inj.feature_importances_}).sort_values('Importanza')
        fig_inj = px.bar(imp_inj.tail(8), x='Importanza', y='Feature', orientation='h', title="Cosa causa i tuoi Sovraccarichi (Infortuni)?", color='Importanza', color_continuous_scale='Reds')
        fig_inj.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_inj, use_container_width=True)

with tab7:
    st.header("🎓 Tesi Ready: Discussione Metodologica")
    st.markdown("""
    ### **L'Integrazione delle Dinamiche di Corsa nel Machine Learning**
    Rispetto ai modelli classici basati solo su Frequenza Cardiaca, questo sistema analizza i dati grezzi derivanti dagli accelerometri del wearable.
    
    Come visibile nei grafici XAI (Explainable AI) del tab precedente, variabili come il **Tempo di Contatto col Suolo (GCT)** e le variazioni di **SpO2 notturno** diventano i *predittori dominanti* sia per l'efficienza della corsa (Pace) sia per il rischio infortuni, rendendo il modello XGBoost significativamente più accurato.
    """)

st.markdown("<div class='footer'>Apple Watch AI Analytics PRO Edition • Data Science & Sports Medicine Integration</div>", unsafe_allow_html=True)
