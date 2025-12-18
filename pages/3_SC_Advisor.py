# Interactive SC decision support: Pit vs Stay analysis
import streamlit as st
import pandas as pd
from ui.styling import apply_custom_style
from ui.components import glass_metric_card, info_card
from strategy_tools import StrategyTools

st.set_page_config(page_title="SC Advisor", layout="wide")
apply_custom_style()

st.title("Safety Car Advisor")
st.markdown("*Real-time PIT vs STAY decision support during Safety Car periods.*")

if 'race_data' not in st.session_state:
    st.warning("No Telemetry Data Loaded. Please go to Home and load a race.")
    st.stop()

laps = st.session_state['race_data']
session = st.session_state['session']
track_meta = pd.read_csv('track_metadata.csv')

event_name = session.event['EventName']

GP_ALIASES = {
    'british': 'uk', 'great britain': 'uk', 'britain': 'uk',
    'french': 'france', 'german': 'germany', 'russian': 'russia',
    'turkish': 'turkey', 'portuguese': 'portugal',
    'tuscan': 'italy', 'eifel': 'germany', 'sakhir': 'bahrain',
    'emilia': 'italy', 'styrian': 'austria', '70th': 'uk',
    'italian': 'italy', 'spanish': 'spain', 'hungarian': 'hungary',
    'belgian': 'belgium', 'dutch': 'netherlands', 'mexican': 'mexico',
    'brazilian': 'brazil', 'australian': 'australia', 'japanese': 'japan',
    'canadian': 'canada', 'austrian': 'austria', 'chinese': 'china',
    'american': 'usa', 'united states': 'usa',
}

current_track = track_meta[track_meta['track_name'].str.contains(event_name, case=False, regex=False)]

if current_track.empty:
    ignored_words = ['grand', 'prix', 'circuit', 'international', 'autodrome', 'autodromo', 'street', 'f1', 'formula', '1']
    event_words = [w.lower() for w in event_name.split() if w.lower() not in ignored_words]
    
    for idx, row in track_meta.iterrows():
        track_name_lower = row['track_name'].lower()
        if any(word in track_name_lower for word in event_words):
            current_track = track_meta.iloc[[idx]]
            break

if current_track.empty:
    for idx, row in track_meta.iterrows():
        if row['country'].lower() in event_name.lower():
            current_track = track_meta.iloc[[idx]]
            break

if current_track.empty:
    event_lower = event_name.lower()
    for alias, country in GP_ALIASES.items():
        if alias in event_lower:
            matches = track_meta[track_meta['country'].str.lower() == country]
            if len(matches) > 0:
                filtered = matches[matches['track_name'].str.lower().str.contains(event_lower.split()[0])]
                if not filtered.empty:
                    current_track = filtered.iloc[[0]]
                else:
                    current_track = matches.iloc[[0]]
                break

if current_track.empty or len(current_track) == 0:
    current_track = track_meta.iloc[0]
else:
    current_track = current_track.iloc[0]
total_laps = session.total_laps
if total_laps is None or total_laps == 0:
    if 'LapNumber' in laps.columns and len(laps) > 0:
        total_laps = int(laps['LapNumber'].max())
    else:
        total_laps = 57

pit_loss = float(current_track['estimated_green_flag_pit_loss_s'])

with st.sidebar:
    st.header("Driver State (YOU)")
    
    position = st.number_input("Position", min_value=1, max_value=20, value=1)
    tire_age = st.slider("Tire Age (Laps)", 0, 50, 20)
    compound = st.selectbox("Current Compound", ["SOFT", "MEDIUM", "HARD"], index=1)
    
    st.markdown("---")
    st.header("Race Context")
    
    sc_lap = st.slider("SC Deployed on Lap", 1, total_laps, total_laps // 2)
    laps_remaining = total_laps - sc_lap
    
    st.metric("Laps Remaining", laps_remaining)
    
    st.markdown("---")
    st.header("Gap Analysis")
    
    gap_ahead = st.number_input("Gap to Car Ahead (s)", value=2.5, step=0.1)
    gap_behind = st.number_input("Gap to Car Behind (s)", value=3.0, step=0.1)

tools = StrategyTools()
sc_result = tools.get_sc_advice(
    position=position,
    tire_age=tire_age,
    compound=compound,
    gap_ahead=gap_ahead,
    gap_behind=gap_behind,
    sc_lap=sc_lap,
    total_laps=total_laps,
    pit_loss=pit_loss
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Decision Analysis")
    
    if "PIT" in sc_result.recommendation.upper():
        st.success(f"{sc_result.recommendation}")
    else:
        st.error(f"{sc_result.recommendation}")
    
    st.markdown("### Confidence")
    st.progress(sc_result.confidence, text=f"{sc_result.confidence*100:.0f}%")
    
    st.markdown("### Reasoning")
    info_card("Model Logic", sc_result.reasoning)
    
    st.markdown("### Decision Factors")
    factors_df = pd.DataFrame([
        {"Factor": "Tire Age", "Value": f"{tire_age} laps", "Impact": "Older = Favor PIT"},
        {"Factor": "Position", "Value": f"P{position}", "Impact": "Leader = Risk losing position"},
        {"Factor": "Laps Remaining", "Value": f"{laps_remaining}", "Impact": "More laps = Favor PIT"},
        {"Factor": "Pit Loss (Free?)", "Value": f"{pit_loss:.1f}s", "Impact": "SC negates time loss"},
        {"Factor": "Gap Behind", "Value": f"{gap_behind}s", "Impact": "Small gap = Risk undercut"}
    ])
    st.dataframe(factors_df, use_container_width=True)

with col2:
    st.subheader("Quick Metrics")
    
    glass_metric_card("Pit Loss", f"{pit_loss:.1f}s")
    glass_metric_card("SC Risk Index", f"{current_track['historical_sc_risk_index']}")
    glass_metric_card("Expected SC Laps", "~3-5", help_text="Typical SC duration")
    
    st.markdown("---")

with st.expander("How the SC Advisor Works"):
    st.markdown("""
    **Inputs Considered:**
    - **Tire Age**: Older tires = stronger case for pitting.
    - **Position**: Leaders risk losing track position; followers have less to lose.
    - **Laps Remaining**: More laps = more time for fresh tires to pay off.
    - **Pit Loss**: Under SC, pit loss is effectively "free" (field bunches up).
    - **Gaps**: Small gap behind = risk of being undercut if you don't pit.
    
    **This is a Snapshot-Based Advisor:**
    - It evaluates the state *at the moment of SC deployment*.
    - It does NOT simulate human reactions or team radio dynamics.
    - It provides *model-optimal* advice, not a guarantee.
    """)