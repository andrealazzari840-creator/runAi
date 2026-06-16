import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor, XGBClassifier
import warnings
import time


warnings.filterwarnings('ignore')


# ==========================================
# 0. SETUP E CSS PREMIUM
# ==========================================
st.set_page_config(
    page_title="Apple Watch Pro Coach",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 800; color: #1f77b4; }
    .kpi-card { background: linear-gradient(145deg, #1e1e1e, #2d2d2d); padding: 20px; border-radius: 15px; border: 1px solid #333; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 20px; color: white;}
    .coach-message { background: rgba(30, 144, 255, 0.1); border-left: 5px solid #1e90ff; padding: 20px; border-radius: 5px 15px 15px 5px; margin: 15px 0; font-size: 1.1em;}
    .alert-red { border-left: 5px solid #ff4757; background: rgba(255, 71, 87, 0.1); padding: 15px; border-radius: 5px;}
    .alert-green { border-left: 5px solid #2ed573; background: rgba(46, 213, 115, 0.1); padding: 15px; border-radius: 5px;}
    h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-weight: 800; letter-spacing: -0.5px;}
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 1. MOTORE MACHINE LEARNING
# ==========================================
@st.cache_data
def generate_complex_dataset() -> pd.DataFrame:
    """Genera un dataset storico avanzato basato sul dualismo Vita/Corsa"""
    np.random.seed(42)
    days = 600
    dates = pd.date_range(start="2024-01-01", periods=days)
    
    sleep = np.random.normal(7.2, 1.2, days).clip(3, 10)
    hrv = np.random.normal(55, 15, days).clip(15, 100)
    spo2 = np.random.normal(97, 1.5, days).clip(90, 100)
    stress = np.random.randint(1, 10, days)
    acwr = np.random.normal(1.1, 0.3, days).clip(0.6, 2.5)
    
    last_dist = np.random.normal(12, 5, days).clip(2, 42)
    last_gct = np.random.normal(220, 15, days).clip(170, 300)
    last_power = np.random.normal(260, 40, days).clip(150, 450)
    
    target_dist = np.random.normal(10, 4, days).clip(2, 30)
    target_rpe = np.random.randint(3, 10, days)
    
    df = pd.DataFrame({
        'Date': dates, 'Sleep': sleep, 'HRV': hrv, 'SpO2': spo2, 'Stress': stress, 'ACWR': acwr,
        'Last_Dist': last_dist, 'Last_GCT': last_gct, 'Last_Power': last_power,
        'Target_Dist': target_dist, 'Target_RPE': target_rpe
    })
    
    df['ISMA'] = (df['Stress'] * df['Target_RPE']) / (df['Sleep'] + 0.1)
    df['Pace_min_km'] = 8.5 + (df['Last_GCT'] * 0.01) - (df['Last_Power'] * 0.003) - (df['HRV'] * 0.01) + np.random.normal(0, 0.15, days)
    
    risk_prob = np.where((df['ACWR'] > 1.4) | ((df['Last_GCT'] > 250) & (df['HRV'] < 40)), 0.8, 0.05)
    df['Injury_Event'] = np.random.binomial(1, risk_prob)
    
    return df


df = generate_complex_dataset()


@st.cache_resource
def train_models(data: pd.DataFrame):
    features = ['Sleep', 'HRV', 'SpO2', 'Stress', 'ACWR', 'Last_Dist', 'Last_GCT', 'Last_Power', 'Target_Dist', 'Target_RPE', 'ISMA']
    
    model_pace = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model_pace.fit(data[features], data['Pace_min_km'])
    
    model_inj = XGBClassifier(n_estimators=200, max_depth=4, eval_metric='logloss', random_state=42)
    model_inj.fit(data[features], data['Injury_Event'])
    
    return model_pace, model_inj, features


model_pace, model_inj, feature_cols = train_models(df)


# ==========================================
# 2. INIZIALIZZAZIONE SESSION STATE
# ==========================================
if 'aw_synced' not in st.session_state:
    st.session_state.aw_synced = False
    st.session_state.coach_plan_generated = False
    st.session_state.form_submitted = False
    
    st.session_state.today_sleep = 7.0
    st.session_state.today_hrv = 50.0
    st.session_state.today_spo2 = 97.0
    st.session_state.today_stress = 5
    st.session_state.today_acwr = 1.1
    
    st.session_state.last_dist = 10.0
    st.session_state.last_gct = 220.0
    st.session_state.last_power = 250.0
    st.session_state.last_cadence = 165.0
    st.session_state.last_pace = "5:30"
    
    st.session_state.target_dist = 8.0
    st.session_state.target_rpe = 6


# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=40)
    st.title("Hub Dispositivo")
    
    if not st.session_state.aw_synced:
        st.warning("Nessun dato recente. Connetti l'orologio.")
        if st.button("📡 Sincronizza Dati HealthKit", type="primary", use_container_width=True):
            with st.spinner("Estrazione Salute Odierna..."): time.sleep(1)
            with st.spinner("Estrazione Dinamiche Corsa..."): time.sleep(1)
            
            st.session_state.today_sleep = 5.5
            st.session_state.today_hrv = 38.0
            st.session_state.today_spo2 = 96.5
            st.session_state.today_stress = 8
            st.session_state.today_acwr = 1.45
            
            st.session_state.last_dist = 18.5
            st.session_state.last_gct = 245.0
            st.session_state.last_power = 290.0
            st.session_state.last_cadence = 158.0
            st.session_state.last_pace = "4:55"
            
            st.session_state.aw_synced = True
            st.session_state.form_submitted = False
            st.rerun()
    else:
        st.success("Sincronizzazione completata: Oggi, 08:14 AM")
        st.markdown("### 🩸 Salute di Oggi")
        st.metric("Sonno", f"{st.session_state.today_sleep} h", "-1.5h vs media", delta_color="inverse")
        st.metric("HRV", f"{st.session_state.today_hrv} ms", "Stato: Affaticato", delta_color="inverse")
        
        st.markdown("### 🏃 Ultima Corsa (Ieri)")
        st.metric("Distanza", f"{st.session_state.last_dist} km")
        st.metric("Contatto Suolo (GCT)", f"{st.session_state.last_gct} ms", "+15ms vs base", delta_color="inverse")
        
        if st.button("🔌 Disconnetti", use_container_width=True):
            st.session_state.aw_synced = False
            st.session_state.coach_plan_generated = False
            st.session_state.form_submitted = False
            st.rerun()


# ==========================================
# 4. HEADER PRINCIPALE
# ==========================================
st.title("🍏 Apple Watch AI Pro - Sports Science Hub")
st.markdown("Sistema integrato di analisi dati vitali, biomeccanica e programmazione allenamenti tramite Machine Learning.")
st.divider()


# ==========================================
# 5. TABS INTERATTIVI
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗣️ 1. INTERVISTA COACH", 
    "🎛️ 2. SIMULATORE ML", 
    "📊 3. ANALITICA & RADAR", 
    "🎯 4. INDICI KPI", 
    "🏆 5. BIO-MEDAGLIE"
])


# =======================================================
# TAB 1: INTERVISTA AI COACH
# =======================================================
with tab1:
    st.header("🗣️ Intervista Pre-Allenamento")
    st.markdown("L'IA ha letto i tuoi dati biomedici. Ora ha bisogno del tuo **feedback soggettivo** per creare la scheda di oggi.")
    
    if not st.session_state.aw_synced:
        st.info("Sincronizza l'Apple Watch dalla barra laterale per iniziare.")
    else:
        col_c1, col_c2 = st.columns([1, 1])
        
        with col_c1:
            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            st.subheader("📝 Questionario Atleta")
            with st.form("coach_form", clear_on_submit=False):
                doms = st.slider("1. Livello di indolenzimento muscolare (DOMS)?", 1, 10, 5, help="1 = Fresco, 10 = Non riesco a camminare")
                time_avail = st.selectbox("2. Quanto tempo hai oggi?", ["< 30 min", "45 - 60 min", "> 90 min"])
                mentality = st.radio("3. Livello di energia mentale?", ["Scaricato/Stressato", "Normale", "Carico a molla!"])
                goal = st.selectbox("4. Obiettivo del prossimo mese?", ["Migliorare la Resistenza", "Aumentare la Velocità", "Recupero Infortuni", "Puro Benessere"])
                
                submitted = st.form_submit_button("Analizza e Genera Allenamento 🚀", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            if submitted and not st.session_state.form_submitted:
                st.session_state.form_submitted = True
                st.session_state.coach_plan_generated = True
                
                with st.spinner("Incrocio dati HealthKit con risposte soggettive..."):
                    time.sleep(1.5)
                
                st.subheader("📋 Referto e Prescrizione")
                
                if st.session_state.today_hrv < 45 and doms > 6:
                    diag = "🚨 **SOVRACCARICO SISTEMICO:** HRV basso e DOMS elevati. Residui dell'ultimo lungo nei muscoli."
                    presc_dist = 5.0
                    presc_rpe = 3
                    presc_type = "Fondo Lento Rigenerativo (Z1)"
                elif st.session_state.today_sleep > 7 and mentality == "Carico a molla!":
                    diag = "✅ **PICCO DI FORMA:** Metriche eccellenti e prontezza soggettiva. Sistema Nervoso pronto."
                    presc_dist = 12.0
                    presc_rpe = 8
                    presc_type = "Ripetute in Soglia (Interval Training)"
                else:
                    diag = "⚠️ **STATO INTERMEDIO:** Affaticamento lavorativo ma muscoli discreti. Manteniamo il motore acceso."
                    presc_dist = 8.0
                    presc_rpe = 5
                    presc_type = "Corsa Continua Media (Fartlek leggero)"

                st.session_state.target_dist = presc_dist
                st.session_state.target_rpe = presc_rpe

                st.markdown(f"<div class='coach-message'>{diag}</div>", unsafe_allow_html=True)
                
                st.markdown("### 🎯 La Tua Scheda di Oggi")
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Tipo Lavoro", presc_type)
                c_b.metric("Distanza Consigliata", f"{presc_dist} km")
                c_c.metric("Intensità (RPE)", f"{presc_rpe} / 10")
                
                st.success("👉 Dati pre-caricati nel **Simulatore ML**. Vai al Tab 2!")
        
        with col_c2:
            if not st.session_state.coach_plan_generated:
                st.info("Compila il form per ricevere la consulenza dell'IA.")


# =======================================================
# TAB 2: SIMULATORE ML
# =======================================================
with tab2:
    st.header("🎛️ Simulatore Sessione XGBoost")
    st.markdown("Testa l'allenamento suggerito o modifica per vedere come reagisce l'algoritmo.")
    
    col_s1, col_s2, col_s3 = st.columns([1,1,1])
    
    with col_s1:
        st.markdown("#### 🏃 Obiettivo Odierno")
        st.session_state.target_dist = st.number_input("Distanza Target (km)", 1.0, 42.0, float(st.session_state.target_dist), 0.5)
        st.session_state.target_rpe = st.slider("Sforzo (RPE)", 1, 10, int(st.session_state.target_rpe))
    
    with col_s2:
        st.markdown("#### 🩸 Fisiologia (Manipolabile)")
        sim_sleep = st.number_input("Sonno Nottata (h)", 2.0, 12.0, float(st.session_state.today_sleep), 0.5)
        sim_hrv = st.number_input("HRV (ms)", 10.0, 150.0, float(st.session_state.today_hrv))
        
    with col_s3:
        st.markdown("#### ⚙️ Biomeccanica Storica")
        st.info(f"Ultimo GCT: {st.session_state.last_gct} ms")
        st.info(f"Ultima Power: {st.session_state.last_power} W")

    live_isma = (st.session_state.today_stress * st.session_state.target_rpe) / (sim_sleep + 0.1)

    input_df = pd.DataFrame({
        'Sleep': [sim_sleep], 'HRV': [sim_hrv], 'SpO2': [st.session_state.today_spo2], 
        'Stress': [st.session_state.today_stress], 'ACWR': [st.session_state.today_acwr],
        'Last_Dist': [st.session_state.last_dist], 'Last_GCT': [st.session_state.last_gct], 
        'Last_Power': [st.session_state.last_power], 'Target_Dist': [st.session_state.target_dist], 
        'Target_RPE': [st.session_state.target_rpe], 'ISMA': [live_isma]
    })[feature_cols]

    pred_pace = model_pace.predict(input_df)[0]
    pred_risk = model_inj.predict_proba(input_df)[0][1] * 100

    st.divider()
    st.markdown("### 🧠 Risposta dell'Algoritmo (Live)")
    
    col_out1, col_out2 = st.columns(2)
    col_out1.metric("⏱️ Ritmo al km Atteso", f"{pred_pace:.2f} min/km", "Calcolato sulle tue dinamiche")
    
    if pred_risk < 30:
        col_out2.metric("🟢 Rischio Infortuni", f"{pred_risk:.1f}%", "- Sicuro")
    elif pred_risk < 60:
        col_out2.metric("🟡 Rischio Infortuni", f"{pred_risk:.1f}%", "- Monitorare")
    else:
        col_out2.metric("🔴 Rischio Infortuni", f"{pred_risk:.1f}%", "- PERICOLO")
        st.error("🚨 Rischio clinico inaccettabile. Diminuisci i chilometri.")


# =======================================================
# TAB 3: ANALITICA & RADAR
# =======================================================
with tab3:
    st.header("📊 Analitica Fisiologica e Profilazione")
    
    c_r1, c_r2 = st.columns(2)
    
    with c_r1:
        st.subheader("🕸️ Profilo Readiness (Radar Chart)")
        categories = ['Sonno (Qualità)', 'HRV (Recupero)', 'GCT (Elasticità)', 'SpO2 (Ossigeno)', 'Stress (Relax)']
        
        val_atleta = [
            min(10, max(0, (st.session_state.today_sleep / 9)*10)), 
            min(10, max(0, (st.session_state.today_hrv / 80)*10)), 
            min(10, max(0, 10 - ((st.session_state.last_gct - 180)/120)*10)),
            min(10, max(0, (st.session_state.today_spo2 - 90))), 
            min(10, max(0, 10 - st.session_state.today_stress))
        ]
        val_elite = [8.5, 9, 8.5, 9.5, 8]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=val_atleta, theta=categories, fill='toself', name='Tua Condizione Oggi', line_color='#ff4757'))
        fig_radar.add_trace(go.Scatterpolar(r=val_elite, theta=categories, fill='toself', name='Baseline Élite', line_color='#1e90ff'))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])), 
            showlegend=True, 
            paper_bgcolor='rgba(0,0,0,0)', 
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with c_r2:
        st.subheader("👟 Analisi Biomeccanica (Scatter)")
        st.markdown("Relazione tra **Potenza (Watt)** ed **Efficienza (GCT)**.")
        fig_scatter = px.scatter(
            df.tail(150), 
            x='Last_GCT', y='Last_Power', 
            color='Injury_Event', 
            labels={'Last_GCT': "Tempo Contatto (ms)", 'Last_Power': 'Potenza (W)'},
            color_continuous_scale='Reds', 
            trendline="ols"
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# =======================================================
# TAB 4: KPI MEDICI
# =======================================================
with tab4:
    st.header("🎯 Monitoraggio KPI Medici")
    
    live_isma = (st.session_state.today_stress * st.session_state.target_rpe) / (st.session_state.today_sleep + 0.1)
    live_islr = (8 * st.session_state.today_stress) / (st.session_state.target_dist + 0.1)

    c_k1, c_k2 = st.columns(2)
    
    with c_k1:
        st.markdown("### 🔴 ISMA (Indice Stress/Adattamento)")
        st.markdown("*(Stress Vita * Sforzo Allenamento) / Ore Sonno*")
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number", value=live_isma,
            gauge={'axis': {'range': [0, 30]}, 'bar': {'color': '#3498db'},
                   'steps': [{'range': [0, 10], 'color': "rgba(46, 213, 115, 0.3)"},
                             {'range': [10, 20], 'color': "rgba(255, 165, 2, 0.3)"},
                             {'range': [20, 30], 'color': "rgba(255, 71, 87, 0.3)"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'value': 20}}
        ))
        fig_g1.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g1, use_container_width=True)

    with c_k2:
        st.markdown("### 🟡 ISLR (Carico Lavoro/Scarico)")
        st.markdown("*(Ore Lavoro * Stress Mentale) / Distanza Corsa*")
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number", value=live_islr,
            gauge={'axis': {'range': [0, 20]}, 'bar': {'color': '#9b59b6'},
                   'steps': [{'range': [0, 6], 'color': "rgba(46, 213, 115, 0.3)"},
                             {'range': [6, 12], 'color': "rgba(255, 165, 2, 0.3)"},
                             {'range': [12, 20], 'color': "rgba(255, 71, 87, 0.3)"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'value': 12}}
        ))
        fig_g2.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g2, use_container_width=True)


# =======================================================
# TAB 5: BIO-MEDAGLIE
# =======================================================
with tab5:
    st.header("🏆 Bio-Medaglie e Gamification")
    st.markdown("I badge si aggiornano in tempo reale in base alla tua salute odierna.")
    
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        if st.session_state.today_sleep >= 8:
            st.markdown("<div class='kpi-card' style='border-top: 5px solid #00f2fe;'><h3>😴 Sonno da Re</h3><p>Hai dormito almeno 8 ore.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='kpi-card' style='opacity: 0.4;'>🔒 <i>Sonno da Re (Richiede 8h)</i></div>", unsafe_allow_html=True)
            
    with col_a2:
        if st.session_state.today_hrv >= 60:
            st.markdown("<div class='kpi-card' style='border-top: 5px solid #2ecc71;'><h3>💓 Cuore Elastico</h3><p>HRV > 60ms.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='kpi-card' style='opacity: 0.4;'>🔒 <i>Cuore Elastico (HRV > 60ms)</i></div>", unsafe_allow_html=True)

    with col_a3:
        if st.session_state.last_gct <= 215:
            st.markdown("<div class='kpi-card' style='border-top: 5px solid #f1c40f;'><h3>⚡ Gazzella</h3><p>GCT < 215ms.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='kpi-card' style='opacity: 0.4;'>🔒 <i>Gazzella (GCT < 215ms)</i></div>", unsafe_allow_html=True)
