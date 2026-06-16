%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor, XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. CONFIGURAZIONE E STILE (APPLE WATCH THEME)
# ==========================================
st.set_page_config(page_title="Apple Watch AI Pro", page_icon="⌚", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 800; color: #1f77b4; }
    .kpi-box { padding: 20px; border-radius: 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); margin-bottom: 15px;}
    .coach-alert-red { border-left: 6px solid #e74c3c; background: rgba(231, 76, 60, 0.1); padding: 15px; border-radius: 0 8px 8px 0; margin-bottom: 15px;}
    .coach-alert-yellow { border-left: 6px solid #f39c12; background: rgba(243, 156, 18, 0.1); padding: 15px; border-radius: 0 8px 8px 0; margin-bottom: 15px;}
    .coach-alert-green { border-left: 6px solid #2ecc71; background: rgba(46, 204, 113, 0.1); padding: 15px; border-radius: 0 8px 8px 0; margin-bottom: 15px;}
    h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. MOTORE DATI E MACHINE LEARNING
# ==========================================
@st.cache_data
def load_historical_data() -> pd.DataFrame:
    np.random.seed(42)
    days = 600
    dates = pd.date_range(start="2024-01-01", periods=days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Distance_km': np.random.normal(12, 4, days).clip(2, 35),
        'RPE': np.random.randint(3, 10, days),
        'Sleep_Hours': np.random.normal(7.2, 1.1, days).clip(4, 10),
        'Stress_Level': np.random.randint(1, 10, days),
        'Avg_BPM': np.random.normal(148, 12, days).clip(110, 190),
        'Cadence_spm': np.random.normal(168, 8, days).clip(140, 200),
        'HRV_ms': np.random.normal(60, 15, days).clip(20, 100),
        'SpO2_pct': np.random.normal(97, 1.5, days).clip(92, 100),
        'GCT_ms': np.random.normal(220, 15, days).clip(180, 300),
        'Running_Power_W': np.random.normal(250, 40, days).clip(150, 400),
        'Elevation_m': np.random.normal(150, 80, days).clip(0, 1000),
        'ACWR': np.random.normal(1.1, 0.3, days).clip(0.5, 2.5)
    })
    
    # Calcolo KPI
    df['ISMA'] = np.round((df['Stress_Level'] * df['RPE']) / (df['Sleep_Hours'] + 0.1), 2)
    df['ISLR'] = np.round((8 * df['Stress_Level']) / (df['Distance_km'] + 0.1), 2)
    df['IITR'] = np.round((22 * 10) / (df['Distance_km'] + 0.1), 2)
    
    # Target: Pace e Rischio Infortuni
    df['Pace_min_km'] = 9.0 - (df['Cadence_spm'] * 0.015) - (df['Running_Power_W'] * 0.002) + (df['GCT_ms'] * 0.005) + np.random.normal(0, 0.1, days)
    injury_prob = np.where((df['ACWR'] > 1.4) | (df['GCT_ms'] > 260) | (df['SpO2_pct'] < 94), 0.75, 0.05)
    df['Injury_Event'] = np.random.binomial(1, injury_prob)
    
    return df.dropna()

df = load_historical_data()

@st.cache_resource
def train_ai_models(data: pd.DataFrame):
    features = ['Distance_km', 'Avg_BPM', 'Cadence_spm', 'Elevation_m', 'Sleep_Hours', 'HRV_ms', 
                'RPE', 'ACWR', 'Stress_Level', 'ISMA', 'ISLR', 'IITR', 'Running_Power_W', 'GCT_ms', 'SpO2_pct']
    
    xgb_perf = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_perf.fit(data[features], data['Pace_min_km'])
    
    xgb_inj = XGBClassifier(n_estimators=150, max_depth=4, eval_metric='logloss', random_state=42)
    xgb_inj.fit(data[features], data['Injury_Event'])
    
    return xgb_perf, xgb_inj, features

xgb_perf, xgb_inj, feature_cols = train_ai_models(df)

# ==========================================
# 2. INIZIALIZZAZIONE REACTIVITY
# ==========================================
default_sim = {
    'sim_dist': 12.0, 'sim_rpe': 7, 'sim_sleep': 7.5, 'sim_stress': 4,
    'sim_hrv': 55.0, 'sim_spo2': 98.0, 'sim_gct': 210.0, 'sim_power': 285.0,
    'sim_cadence': 170.0, 'sim_acwr': 1.1, 'sim_elev': 150.0
}
for key, val in default_sim.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.title("⌚ Simulatore Apple Watch & Coach AI")
st.markdown("##### Modifica i parametri nel Simulatore e osserva le analisi adattarsi istantaneamente.")
st.divider()

# ==========================================
# 3. STRUTTURA A TAB
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎛️ 1. SIMULATORE PREVISIONI", 
    "🎯 2. KPI REALTIME", 
    "🤖 3. COACH AI (REFERTO)", 
    "📊 4. TREND & XAI",
    "🏆 5. ACHIEVEMENTS"
])

# ==========================================
# TAB 1: IL MOTORE (SIMULATORE)
# ==========================================
with tab1:
    st.header("🎛️ Configura la Sessione Odierna")
    
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    
    with col_sim1:
        st.subheader("🏃 Pianificazione")
        st.session_state.sim_dist = st.number_input("Distanza (km)", 1.0, 50.0, st.session_state.sim_dist, 0.5)
        st.session_state.sim_rpe = st.slider("RPE Target (1-10)", 1, 10, st.session_state.sim_rpe)
        st.session_state.sim_acwr = st.slider("ACWR (Carico Acuto/Cronico)", 0.5, 2.5, st.session_state.sim_acwr, 0.1)
        
    with col_sim2:
        st.subheader("🩸 Dati Notturni (Recupero)")
        st.session_state.sim_sleep = st.number_input("Ore di Sonno", 2.0, 12.0, st.session_state.sim_sleep, 0.5)
        st.session_state.sim_hrv = st.number_input("HRV Mattutino (ms)", 10.0, 150.0, st.session_state.sim_hrv)
        st.session_state.sim_spo2 = st.number_input("SpO2 Notturno (%)", 85.0, 100.0, st.session_state.sim_spo2)
        
    with col_sim3:
        st.subheader("⚡ Biomeccanica (Apple Watch)")
        st.session_state.sim_gct = st.number_input("Tempo Contatto Suolo (ms)", 150.0, 350.0, st.session_state.sim_gct)
        st.session_state.sim_power = st.number_input("Potenza Corsa (Watt)", 100.0, 500.0, st.session_state.sim_power)
        st.session_state.sim_stress = st.slider("Stress Lavorativo/Mentale", 1, 10, st.session_state.sim_stress)
        
    # Variabili fisse o derivate per questa simulazione
    st.session_state.sim_cadence = 170.0
    st.session_state.sim_elev = 150.0

    # 1. CALCOLO KPI IN TEMPO REALE
    live_isma = (st.session_state.sim_stress * st.session_state.sim_rpe) / (st.session_state.sim_sleep + 0.1)
    live_islr = (8 * st.session_state.sim_stress) / (st.session_state.sim_dist + 0.1)
    live_iitr = (22 * 10) / (st.session_state.sim_dist + 0.1)

    # 2. ESECUZIONE ML (XGBOOST) IN TEMPO REALE
    input_dict = {
        'Distance_km': [st.session_state.sim_dist], 'Avg_BPM': [145], 'Cadence_spm': [st.session_state.sim_cadence], 
        'Elevation_m': [st.session_state.sim_elev], 'Sleep_Hours': [st.session_state.sim_sleep], 
        'HRV_ms': [st.session_state.sim_hrv], 'RPE': [st.session_state.sim_rpe], 'ACWR': [st.session_state.sim_acwr], 
        'Stress_Level': [st.session_state.sim_stress], 'ISMA': [live_isma], 'ISLR': [live_islr], 'IITR': [live_iitr], 
        'Running_Power_W': [st.session_state.sim_power], 'GCT_ms': [st.session_state.sim_gct], 'SpO2_pct': [st.session_state.sim_spo2]
    }
    input_df = pd.DataFrame(input_dict)[feature_cols]
    
    pred_pace = xgb_perf.predict(input_df)[0]
    pred_risk = xgb_inj.predict_proba(input_df)[0][1] * 100

    st.divider()
    st.subheader("🎯 Output Predittivo Immediato (XGBoost)")
    col_out1, col_out2 = st.columns(2)
    col_out1.metric("⏱️ Pace Ottimale Stimato", f"{pred_pace:.2f} min/km", "Calcolato in base alla Running Power")
    
    risk_color = "normal" if pred_risk < 30 else "off" if pred_risk < 60 else "inverse"
    col_out2.metric("⚠️ Rischio Infortunio / Sovraccarico", f"{pred_risk:.1f}%", delta_color=risk_color)

# ==========================================
# TAB 2: KPI REALTIME
# ==========================================
with tab2:
    st.header("🎯 Analisi Indici Proprietari")
    st.markdown("I grafici riflettono la **condizione metabolica attuale** calcolata nel simulatore.")
    
    with st.expander("📚 Leggi la Teoria Medica dei KPI", expanded=False):
        st.markdown("""
        **Spiegazione Scientifica degli Indici:**
        * 🔴 **ISMA (Indice di Stress Metabolico e Adattamento):** Calcolato come `(Stress * RPE) / Sonno`. Misura il bilancio tra distruzione tissutale (allenamento) e anabolismo (recupero). Un ISMA alto indica che stai bloccando la supercompensazione.
        * 🟡 **ISLR (Indice Stress Lavoro-Recupero):** `(Ore Lavoro * Stress) / Distanza`. Il Sistema Nervoso Centrale non distingue tra fatica da ufficio e fatica da corsa. Questo indice unifica il *cognitive load* con lo scarico sportivo.
        * 🟠 **IITR (Indice Impatto Termoregolatorio):** Valuta la spesa cardiaca fittizia dovuta al meteo. Alto IITR = forte Deriva Cardiaca senza reali benefici allenanti.
        """)

    col_g1, col_g2, col_g3 = st.columns(3)
    
    def plot_gauge(val, title, s_gialla, s_rossa, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=val, title={'text': title, 'font': {'size': 18}},
            gauge={'axis': {'range': [0, s_rossa*1.5]}, 'bar': {'color': color},
                   'steps': [{'range': [0, s_gialla], 'color': "rgba(46, 204, 113, 0.2)"},
                             {'range': [s_gialla, s_rossa], 'color': "rgba(243, 156, 18, 0.2)"},
                             {'range': [s_rossa, s_rossa*1.5], 'color': "rgba(231, 76, 60, 0.2)"}],
                   'threshold': {'line': {'color': "red", 'width': 3}, 'value': s_rossa}}
        ))
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
        return fig

    # Medie storiche per le soglie
    m_isma = df['ISMA'].mean(); s_isma = df['ISMA'].std()
    m_islr = df['ISLR'].mean(); s_islr = df['ISLR'].std()
    
    with col_g1: st.plotly_chart(plot_gauge(live_isma, "ISMA", m_isma+s_isma, m_isma+s_isma*1.5, "#3498db"), use_container_width=True)
    with col_g2: st.plotly_chart(plot_gauge(live_islr, "ISLR", m_islr+s_islr, m_islr+s_islr*1.5, "#9b59b6"), use_container_width=True)
    with col_g3: st.plotly_chart(plot_gauge(live_iitr, "IITR", 10, 15, "#e67e22"), use_container_width=True)

# ==========================================
# TAB 3: COACH AI (REFERTO DINAMICO)
# ==========================================
with tab3:
    st.header("🤖 Referto Tecnico e Analisi (Sports Science)")
    st.markdown(f"*(Il Coach sta analizzando i dati per la tua sessione da **{st.session_state.sim_dist} km** a RPE **{st.session_state.sim_rpe}**)*")
    
    # 1. ANALISI SISTEMA NERVOSO E SONNO
    st.subheader("🧠 1. Sistema Nervoso e Readiness")
    if st.session_state.sim_sleep < 6.0 and st.session_state.sim_hrv < 40:
        st.markdown(f"<div class='coach-alert-red'><b>🚨 SOVRACCARICO PARASIMPATICO:</b> Il combinato disposto di sonno carente ({st.session_state.sim_sleep}h) e HRV depresso ({st.session_state.sim_hrv}ms) indica una forte inibizione del recupero. Il tuo Sistema Nervoso Centrale è esaurito. <b>Azione Coach:</b> Veto assoluto sui lavori di qualità. Si prescrive categoricamente riposo passivo o 20 min di recovery attivo (Z1 cardiaca).</div>", unsafe_allow_html=True)
    elif st.session_state.sim_sleep >= 7.5 and st.session_state.sim_hrv > 55:
        st.markdown(f"<div class='coach-alert-green'><b>✅ OMEOSTASI PERFETTA:</b> Il sonno ({st.session_state.sim_sleep}h) e la variabilità cardiaca mostrano un bilancio autonomico eccellente. Finestra anabolica aperta. <b>Azione Coach:</b> Condizione ideale per applicare carichi neurali (Sprint, HIT) o lavori in soglia anaerobica.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='coach-alert-yellow'><b>⚠️ RECUPERO INCOMPLETO (ZONA GRIGIA):</b> I parametri sono intermedi. <b>Azione Coach:</b> Puoi affrontare la sessione, ma implementa la regola del 'Cardiac Drift'. Se i battiti salgono del 10% a parità di Pace nella seconda metà dell'allenamento, interrompi l'intensità.</div>", unsafe_allow_html=True)

    # 2. ANALISI BIOMECCANICA APPLE WATCH
    st.subheader("🦶 2. Dinamiche di Corsa e Biomeccanica")
    if st.session_state.sim_gct > 250:
        st.markdown(f"<div class='coach-alert-red'><b>🚨 DEFICIT ELASTICO (GCT Alto):</b> Un Tempo di Contatto al Suolo di {st.session_state.sim_gct}ms è critico. La stiffness dei tendini d'Achille è crollata: stai 'affondando' nel terreno assorbendo energia eccentrica dannosa. <b>Azione Coach:</b> Altissimo rischio di infortunio al ginocchio o fascite plantare. Inserisci andature (plyometrics) nel riscaldamento e riduci i chilometri del 30%.</div>", unsafe_allow_html=True)
    elif st.session_state.sim_power > 300 and st.session_state.sim_cadence < 155:
        st.markdown(f"<div class='coach-alert-yellow'><b>⚠️ OVERSTRIDING (Frenata Eccesiva):</b> Esprimi una potenza massiccia ({st.session_state.sim_power}W) ma la cadenza è bassa. Stai atterrando di tallone troppo avanti rispetto al baricentro. <b>Azione Coach:</b> Focus tecnico. Aumenta la frequenza dei passi per proteggere le ginocchia, anche a scapito di una lieve riduzione della potenza.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='coach-alert-green'><b>✅ EFFICIENZA MECCANICA:</b> Il rapporto tra Potenza espressa ({st.session_state.sim_power}W) e tempo di contatto è bilanciato. La cinematica della corsa è in zona di totale sicurezza.</div>", unsafe_allow_html=True)

    # 3. VERDETTO FINALE METABOLICO
    st.subheader("⚖️ 3. Verdetto Prescrittivo")
    if pred_risk > 60 or live_isma > (m_isma + s_isma * 1.5):
        st.error(f"❌ VETO MEDICO: Rischio infortunio/sovraccarico al {pred_risk:.1f}%. I tuoi indici metabolici (ISMA) e biomeccanici non supportano l'RPE {st.session_state.sim_rpe} pianificato. Riposare oggi significa andare più veloci domani.")
    else:
        st.success(f"🏃 AUTORIZZAZIONE CONCESSA: Tutti i sistemi sono Go. Il Machine Learning ha calcolato il tuo ritmo ottimale odierno a {pred_pace:.2f} min/km. Buon allenamento!")

# ==========================================
# TAB 4: TREND E XAI (MACHINE LEARNING)
# ==========================================
with tab4:
    st.header("📊 Spiegabilità del Modello (XAI)")
    
    col_xai1, col_xai2 = st.columns(2)
    with col_xai1:
        st.subheader("Cosa influenza il Rischio Infortuni?")
        imp_inj = pd.DataFrame({'Feature': feature_cols, 'Importanza': xgb_inj.feature_importances_}).sort_values('Importanza')
        fig_inj = px.bar(imp_inj.tail(6), x='Importanza', y='Feature', orientation='h', color='Importanza', color_continuous_scale='Reds')
        fig_inj.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_inj, use_container_width=True)
        
    with col_xai2:
        st.subheader("Cosa determina la tua Velocità (Pace)?")
        imp_perf = pd.DataFrame({'Feature': feature_cols, 'Importanza': xgb_perf.feature_importances_}).sort_values('Importanza')
        fig_perf = px.bar(imp_perf.tail(6), x='Importanza', y='Feature', orientation='h', color='Importanza', color_continuous_scale='Blues')
        fig_perf.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_perf, use_container_width=True)

# ==========================================
# TAB 5: ACHIEVEMENTS
# ==========================================
with tab5:
    st.header("🏆 Gamification & Bio-Medaglie")
    st.markdown("I premi si sbloccano o bloccano **in base ai dati che inserisci nel simulatore.** (Prova a modificare le ore di sonno o il GCT!)")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        # Controllo sul Sonno
        if st.session_state.sim_sleep >= 8:
            st.markdown("<div class='kpi-box' style='background: linear-gradient(135deg, #4facfe, #00f2fe); color: white;'><h3 style='margin:0;'>😴 Maestro del Riposo</h3><p>Sonno ottimale rilevato stanotte (>8h). Il corpo ringrazia.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='kpi-box' style='opacity: 0.5;'>🔒 <i>Maestro del Riposo (Richiede 8h di sonno)</i></div>", unsafe_allow_html=True)
            
        # Controllo sul GCT
        if st.session_state.sim_gct <= 210:
            st.markdown("<div class='kpi-box' style='background: linear-gradient(135deg, #00b09b, #96c93d); color: white;'><h3 style='margin:0;'>⚡ Reattività Elastica</h3><p>GCT sotto i 210ms. Efficienza meccanica da élite.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='kpi-box' style='opacity: 0.5;'>🔒 <i>Reattività Elastica (Richiede GCT <= 210ms)</i></div>", unsafe_allow_html=True)
            
    with col_a2:
        # Controllo sull'ISMA
        if live_isma < 10:
            st.markdown("<div class='kpi-box' style='background: linear-gradient(135deg, #f093fb, #f5576c); color: white;'><h3 style='margin:0;'>🧘 Zen Master</h3><p>ISMA bassissimo. Assenza totale di stress metabolico.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='kpi-box' style='opacity: 0.5;'>🔒 <i>Zen Master (Richiede ISMA < 10)</i></div>", unsafe_allow_html=True)

        # Controllo sul Rischio
        if pred_risk < 15:
            st.markdown("<div class='kpi-box' style='background: linear-gradient(135deg, #FFD700, #FDB931); color: white;'><h3 style='margin:0;'>🛡️ Antifragile</h3><p>Rischio infortunio bassissimo calcolato dall'IA (<15%).</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='kpi-box' style='opacity: 0.5;'>🔒 <i>Antifragile (Richiede Rischio < 15%)</i></div>", unsafe_allow_html=True)
