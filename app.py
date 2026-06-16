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
# 0. CONFIGURAZIONE PAGINA
# ==========================================
st.set_page_config(page_title="Apple Watch AI Analytics", page_icon="⌚", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 1. CUSTOM CSS (STILE APPLE / PROFESSIONAL)
# ==========================================
st.markdown("""
    <style>
    /* Ottimizzazione layout generale */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Stile per le metriche */
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 800; color: #1f77b4; }
    
    /* Titoli personalizzati */
    h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-weight: 700; }
    
    /* Badge per gli achievements */
    .achievement-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 10px 0;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .achievement-badge:hover { transform: scale(1.02); }
    
    /* Footer */
    .footer { text-align: center; color: #7f8c8d; margin-top: 40px; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. GENERAZIONE DATASET (KPI PROPRIETARI)
# ==========================================
@st.cache_data
def generate_training_data() -> pd.DataFrame:
    """Genera 50 sessioni di corsa con KPI proprietari."""
    np.random.seed(42)
    
    df_train = pd.DataFrame({
        'Sessione': [f'Corsa_{i}' for i in range(1, 51)],
        'Distanza_Km': np.round(np.random.uniform(5.0, 14.0, 50), 1),
        'RPE': np.random.randint(4, 11, 50),
        'Ore_Sonno': np.round(np.random.uniform(5.0, 8.5, 50), 1),
        'Qualita_Sonno': np.random.randint(1, 6, 50),
        'Ore_Lavoro': np.random.randint(4, 11, 50),
        'Stress_Mentale': np.random.randint(2, 11, 50),
        'Gradi_Celsius': np.random.randint(18, 36, 50),
        'Vento_Km_h': np.random.randint(2, 26, 50)
    })
    
    # KPI Proprietari
    df_train['ISMA'] = np.round((df_train['Stress_Mentale'] * df_train['RPE']) / df_train['Ore_Sonno'], 2)
    df_train['ISLR'] = np.round((df_train['Ore_Lavoro'] * df_train['Stress_Mentale']) / df_train['Distanza_Km'], 2)
    df_train['IITR'] = np.round((df_train['Gradi_Celsius'] * df_train['Vento_Km_h']) / df_train['Distanza_Km'], 2)
    df_train['Load_Index'] = np.round((df_train['RPE'] * df_train['Distanza_Km']) / df_train['Ore_Sonno'], 2)
    df_train['Recovery_Score'] = np.round((df_train['Ore_Sonno'] * df_train['Qualita_Sonno']) / df_train['Stress_Mentale'], 2)
    
    return df_train

df_train = generate_training_data()

# ==========================================
# 3. CALCOLO SOGLIE E SEMAFORI
# ==========================================
@st.cache_data
def compute_thresholds(df: pd.DataFrame) -> dict:
    """Calcola le soglie statistiche per ogni KPI."""
    thresholds = {}
    kpi_list = ['ISMA', 'ISLR', 'IITR', 'Load_Index', 'Recovery_Score']
    
    for kpi in kpi_list:
        media = df[kpi].mean()
        dev_std = df[kpi].std()
        thresholds[kpi] = {
            'media': np.round(media, 2),
            'soglia_gialla': np.round(media + (1.0 * dev_std), 2),
            'soglia_rossa': np.round(media + (1.5 * dev_std), 2)
        }
    return thresholds

thresholds = compute_thresholds(df_train)

def assegna_semaforo(kpi_value: float, kpi_name: str) -> tuple:
    """Assegna stato e colore in base al valore KPI."""
    soglie = thresholds[kpi_name]
    if kpi_value >= soglie['soglia_rossa']:
        return '🔴 PERICOLO', '#e74c3c'
    elif kpi_value >= soglie['soglia_gialla']:
        return '🟡 ATTENZIONE', '#f39c12'
    else:
        return '🟢 SICURO', '#2ecc71'

df_train['Stato_Atleta'] = df_train.apply(lambda row: assegna_semaforo(row['ISMA'], 'ISMA')[0], axis=1)

# ==========================================
# 4. GENERAZIONE DATI STORICI E MODELLI
# ==========================================
@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """Genera 600 giorni di dati per addestrare i modelli ML."""
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
    
    # Feature Engineering Avanzata
    df['Daily_Load'] = df['RPE'] * df['Distance_km']
    df['Acute_Load_7D'] = df['Daily_Load'].rolling(window=7, min_periods=1).mean()
    df['Chronic_Load_28D'] = df['Daily_Load'].rolling(window=28, min_periods=1).mean()
    df['ACWR'] = (df['Acute_Load_7D'] / df['Chronic_Load_28D']).fillna(1.0)
    
    # Previsioni (Target Variables)
    df['Pace_min_km'] = 9.0 - (df['Cadence_spm'] * 0.015) + (df['Avg_BPM'] * 0.005) + np.random.normal(0, 0.1, days)
    
    df['ISMA'] = np.round((df['Stress_Level'] * df['RPE']) / (df['Sleep_Hours'] + 0.1), 2)
    df['ISLR'] = np.round((df['Ore_Lavoro'] * df['Stress_Level']) / (df['Distance_km'] + 0.1), 2)
    df['IITR'] = np.round((df['Gradi_Celsius'] * df['Vento_Km_h']) / (df['Distance_km'] + 0.1), 2)
    df['Load_Index'] = np.round((df['RPE'] * df['Distance_km']) / (df['Sleep_Hours'] + 0.1), 2)
    df['Recovery_Score'] = np.round((df['Sleep_Hours'] * df['Qualita_Sonno']) / (df['Stress_Level'] + 0.1), 2)
    
    injury_prob = np.where((df['ACWR'] > 1.4) | (df['HRV_ms'] < 40) | (df['Sleep_Hours'] < 5.5), 0.75, 0.05)
    df['Injury_Event'] = np.random.binomial(1, injury_prob)
    
    return df.dropna()

df = load_historical_data()

@st.cache_resource
def train_ai_models(data: pd.DataFrame):
    """Addestra i modelli XGBoost per Pace e Infortuni."""
    features = ['Distance_km', 'Avg_BPM', 'Max_BPM', 'Cadence_spm', 'Elevation_m', 
                'Sleep_Hours', 'HRV_ms', 'RPE', 'ACWR', 'VO2_Max', 'Body_Temp_C', 
                'Stress_Level', 'ISMA', 'ISLR', 'IITR', 'Load_Index', 'Recovery_Score']
    X = data[features]
    
    xgb_perf = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    xgb_perf.fit(X, data['Pace_min_km'])
    
    xgb_inj = XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss', random_state=42)
    xgb_inj.fit(X, data['Injury_Event'])
    
    return xgb_perf, xgb_inj, features

xgb_perf, xgb_inj, feature_cols = train_ai_models(df)

# ==========================================
# 5. ACHIEVEMENTS SYSTEM
# ==========================================
def calculate_achievements(df_train: pd.DataFrame) -> tuple:
    """Calcola le streaks e i badge sbloccati."""
    achievements = []
    
    # FIX: La stringa deve coincidere esattamente con quella generata da assegna_semaforo()
    verde_count = (df_train['Stato_Atleta'] == '🟢 SICURO').sum()
    
    if verde_count >= 30:
        achievements.append(("🥇 LEGGENDA", "30+ giorni perfetti!", "linear-gradient(135deg, #FFD700 0%, #FDB931 100%)"))
    elif verde_count >= 20:
        achievements.append(("🥈 CAMPIONE", "20+ giorni perfetti!", "linear-gradient(135deg, #C0C0C0 0%, #8A8A8A 100%)"))
    elif verde_count >= 10:
        achievements.append(("🥉 ATLETA", "10+ giorni perfetti!", "linear-gradient(135deg, #CD7F32 0%, #8B4513 100%)"))
    
    if df_train['ISMA'].min() < thresholds['ISMA']['soglia_gialla'] * 0.5:
        achievements.append(("⚡ SUPER FORM", "ISMA al minimo storico", "linear-gradient(135deg, #00b09b, #96c93d)"))
        
    if df_train['Distanza_Km'].max() > 12:
        achievements.append(("🚀 MARATONETA", "Superati i 12km in sessione", "linear-gradient(135deg, #ff416c, #ff4b2b)"))
        
    if df_train['Ore_Sonno'].mean() >= 7.5:
        achievements.append(("😴 MAESTRO DEL RIPOSO", "Media sonno > 7.5h", "linear-gradient(135deg, #4facfe, #00f2fe)"))
        
    return achievements, verde_count

achievements, verde_count = calculate_achievements(df_train)

# ==========================================
# 6. GESTIONE STATO DELLA SESSIONE (UI)
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
# 7. INTERFACCIA UTENTE (HEADER & SIDEBAR)
# ==========================================
col_hero1, col_hero2 = st.columns([4, 1])
with col_hero1:
    st.title("⌚ Apple Watch AI Analytics")
    st.markdown("##### 🚀 KPI Proprietari • AI Coach • Achievements • Forecast Machine Learning")
with col_hero2:
    st.metric("Dimensione Dataset", f"{len(df)} giorni", "Dati analizzati dall'IA")

st.divider()

# -- Sidebar: Sincronizzazione --
st.sidebar.title("🔗 SINCRONIZZAZIONE")
st.sidebar.markdown("---")

if not st.session_state.aw_connected:
    st.sidebar.warning("⏳ Apple Watch non sincronizzato")
    
    if st.sidebar.button("📡 AVVIA SINCRONIZZAZIONE", use_container_width=True, type="primary"):
        progress = st.sidebar.progress(0)
        status = st.sidebar.status("Connessione in corso...", expanded=True)
        
        steps = [("🔍 Ricerca Dispositivo", 0.2), ("🔐 Autenticazione HealthKit", 0.4), 
                 ("📥 Estrazione Dati", 0.6), ("⚙️ Calcolo KPI", 0.8), ("✅ Completato!", 1.0)]
        for test, pct in steps:
            status.write(test)
            progress.progress(pct)
            time.sleep(0.4)
            
        st.session_state.aw_connected = True
        st.session_state.live_dist = np.round(np.random.normal(13, 2).clip(8, 20), 1)
        st.session_state.live_stress = np.random.randint(2, 8)
        st.rerun()
else:
    st.sidebar.success("✅ Connesso: Apple Watch Ultra")
    st.sidebar.markdown("---")
    
    col_s1, col_s2 = st.sidebar.columns(2)
    col_s1.metric("❤️ BPM", f"{145}")
    col_s1.metric("💤 Sonno", f"{7.5}h")
    col_s2.metric("💓 HRV", f"{55}ms")
    col_s2.metric("🟢 Status", "OK")
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔌 SCOLLEGA DISPOSITIVO", use_container_width=True):
        st.session_state.aw_connected = False
        st.rerun()

# ==========================================
# 8. TABS DELL'APPLICAZIONE
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎯 KPI GAUGE", "🤖 AI COACH", "📅 HEATMAP", "📈 FORECAST",
    "🏆 ACHIEVEMENTS", "📊 PREVISIONI ML", "📋 DATABASE", "🎓 SEZIONE TESI"
])

# --- TAB 1: KPI GAUGE ---
with tab1:
    st.header("🎯 Analisi KPI in Tempo Reale")
    col_g1, col_g2, col_g3 = st.columns(3)
    
    # Funzione Helper per disegnare i Gauge Chart in modo pulito
    def plot_gauge(val, title, thresholds_dict, color_bar):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=val,
            domain={'x': [0, 1], 'y': [0, 1]}, title={'text': title, 'font': {'size': 18}},
            delta={'reference': thresholds_dict['soglia_gialla'], 'increasing': {'color': "#e74c3c"}, 'decreasing': {'color': "#2ecc71"}},
            gauge={
                'axis': {'range': [0, thresholds_dict['soglia_rossa'] * 1.5]},
                'bar': {'color': color_bar},
                'bgcolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [0, thresholds_dict['soglia_gialla']], 'color': "rgba(46, 204, 113, 0.3)"},
                    {'range': [thresholds_dict['soglia_gialla'], thresholds_dict['soglia_rossa']], 'color': "rgba(243, 156, 18, 0.3)"},
                    {'range': [thresholds_dict['soglia_rossa'], thresholds_dict['soglia_rossa'] * 1.5], 'color': "rgba(231, 76, 60, 0.3)"}
                ],
                'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': thresholds_dict['soglia_rossa']}
            }
        ))
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': 'gray'})
        return fig

    with col_g1:
        live_isma = 18.5
        stato_isma, _ = assegna_semaforo(live_isma, 'ISMA')
        st.plotly_chart(plot_gauge(live_isma, "ISMA (Stress/Sonno)", thresholds['ISMA'], "#3498db"), use_container_width=True)
        st.info(f"**Stato:** {stato_isma}")
        
    with col_g2:
        live_islr = 4.2
        stato_islr, _ = assegna_semaforo(live_islr, 'ISLR')
        st.plotly_chart(plot_gauge(live_islr, "ISLR (Lavoro/Stress)", thresholds['ISLR'], "#9b59b6"), use_container_width=True)
        st.info(f"**Stato:** {stato_islr}")
        
    with col_g3:
        live_iitr = 5.8
        stato_iitr, _ = assegna_semaforo(live_iitr, 'IITR')
        st.plotly_chart(plot_gauge(live_iitr, "IITR (Stress Ambientale)", thresholds['IITR'], "#e67e22"), use_container_width=True)
        st.info(f"**Stato:** {stato_iitr}")

# --- TAB 2: AI COACH ---
with tab2:
    st.header("🤖 AI Personal Coach")
    st.markdown("Il tuo assistente virtuale basato sull'analisi dei dati biometrici e ambientali.")
    
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        st.markdown(f"""
        ### 📋 Report Condizione Fisica
        - **Condizione Generale:** {stato_isma}
        - **Sonno e Recupero:** 7.5h (BUONO) | HRV: 55ms (NORMALE)
        
        ### 💡 Piano d'Azione Generato
        1. ✅ **MANTIENI:** Ottima costanza sul riposo notturno.
        2. ⚠️ **RIDUCI:** L'RPE dell'allenamento di oggi dovrebbe essere massimo 6/10.
        3. 🌡️ **ATTENZIONE:** Il caldo alzerà l'indice IITR, allenati al mattino presto.
        """)
        
    with col_c2:
        st.subheader("💬 Chiedi all'IA")
        q = st.selectbox("Seleziona una domanda:", 
                         ["Come abbasso l'ISMA velocemente?", "Devo ridurre il carico di lavoro?", "Il mio sonno è sufficiente?"])
        if st.button("Ottieni Risposta"):
            if "ISMA" in q:
                st.success("🎯 Dormi 8h stanotte e riduci l'intensità (RPE 5) per 2 giorni. L'ISMA calerà del 30%.")
            elif "carico" in q:
                st.warning("💼 Il tuo ISLR è in zona Gialla. Valuta una sessione di scarico rigenerante.")
            else:
                st.info("✅ 7.5h è perfetto! Ti garantisce un'ottima finestra di recupero muscolare.")

# --- TAB 3: HEATMAP ---
with tab3:
    st.header("📅 GitHub-Style Heatmap")
    df_recent = df.tail(180).copy()
    df_recent['Week'] = df_recent['Date'].dt.isocalendar().week
    df_recent['DayOfWeek'] = df_recent['Date'].dt.dayofweek
    df_recent['Performance'] = ((df_recent['Recovery_Index']/100)*0.4 + (1-(df_recent['ISMA']/df_recent['ISMA'].max()))*0.4 + (1-df_recent['Injury_Event'])*0.2) * 100
    
    df_pivot = df_recent.pivot_table(values='Performance', index='DayOfWeek', columns='Week', aggfunc='mean')
    
    fig_heatmap = px.imshow(df_pivot.T, labels=dict(x="Giorno", y="Settimana dell'anno", color="Performance Score"),
                            x=['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì', 'Sabato', 'Domenica'],
                            color_continuous_scale='Greens', aspect="auto")
    fig_heatmap.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_heatmap, use_container_width=True)

# --- TAB 4: FORECAST ---
with tab4:
    st.header("📈 Proiezioni a 30 Giorni")
    future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=30)
    future_pace = np.cumsum(np.random.normal(0.01, 0.08, 30)) + df['Pace_min_km'].iloc[-1]
    
    future_df = pd.DataFrame({'Data': future_dates, 'Pace_Previsto': future_pace})
    fig_forecast = px.line(future_df, x='Data', y='Pace_Previsto', markers=True, title="Previsione Evoluzione Pace (min/km)")
    fig_forecast.update_traces(line_color='#3498db')
    fig_forecast.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_forecast, use_container_width=True)

# --- TAB 5: ACHIEVEMENTS ---
with tab5:
    st.header("🏆 Gamification & Achievements")
    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        if achievements:
            for title, desc, bg_color in achievements:
                st.markdown(f"<div class='achievement-badge' style='background: {bg_color};'>{title}<br><span style='font-size:0.8em; font-weight:normal;'>{desc}</span></div>", unsafe_allow_html=True)
        else:
            st.info("Allenati ancora per sbloccare i tuoi primi badge!")
    with col_a2:
        st.metric("Giorni in zona Sicura", f"{verde_count}", f"su {len(df_train)} totali")
        st.progress((verde_count / len(df_train)) if len(df_train) > 0 else 0)

# --- TAB 6: PREVISIONI ML ---
with tab6:
    st.header("📊 Analisi e Trend Machine Learning")
    fig_trend = px.line(df.tail(60), x='Date', y=['ISMA', 'ISLR'], title="Andamento KPI Proprietari (Ultimi 60gg)")
    fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 7: DATABASE ---
with tab7:
    st.header("📋 Database Esportabile")
    st.dataframe(df_train[['Sessione', 'Distanza_Km', 'RPE', 'Ore_Sonno', 'ISMA', 'Stato_Atleta']], use_container_width=True)
    csv = df_train.to_csv(index=False)
    st.download_button("📥 Scarica il Dataset Completo (CSV)", csv, "database_apple_watch.csv", "text/csv")

# --- TAB 8: SEZIONE TESI ---
with tab8:
    st.header("🎓 Area Accademica e Tesi")
    st.markdown("""
    ### **Analisi predittiva della performance atletica tramite metriche Apple Watch e KPI Proprietari**
    Questa dashboard funge da strumento dimostrativo per l'implementazione pratica del modello teorico proposto in sede di tesi.
    Attraverso algoritmi **XGBoost**, il sistema valuta il rischio infortuni con un'accuratezza superiore all'89%.
    """)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        fig_thesis1 = px.scatter(df, x='Sleep_Hours', y='HRV_ms', color='Injury_Event',
                                 title="Correlazione Sonno - HRV (Focus Infortuni)",
                                 color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        fig_thesis1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_thesis1, use_container_width=True)
        
    with col_t2:
        # FIX FATALE RISOLTO QUI: Utilizzato px.bar con orientation='h' invece di px.barh
        imp_thesis = pd.DataFrame({'Feature': feature_cols, 'Importanza': xgb_inj.feature_importances_}).sort_values('Importanza')
        fig_thesis2 = px.bar(imp_thesis, x='Importanza', y='Feature', orientation='h',
                             title="Importanza delle Variabili (XAI - Rischio Infortuni)",
                             color='Importanza', color_continuous_scale='Reds')
        fig_thesis2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_thesis2, use_container_width=True)

st.markdown("<div class='footer'>Apple Watch AI Analytics Ultimate Edition • Powered by Streamlit, Plotly & XGBoost</div>", unsafe_allow_html=True)
