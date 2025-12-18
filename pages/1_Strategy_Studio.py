# Interactive strategy simulation and decision support
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ui.styling import apply_custom_style
from ui.components import glass_metric_card, info_card, section_header
from strategy_tools import StrategyTools
from visualization import plot_strategy_comparison, plot_confidence_envelope

st.set_page_config(page_title="Strategy Studio", layout="wide")
apply_custom_style()

st.title("Strategy Studio")

if 'race_data' not in st.session_state:
    st.warning("No Telemetry Data Loaded. Please go to Home and load a race.")
    st.stop()

laps = st.session_state['race_data']
session = st.session_state['session']
track_meta = pd.read_csv('track_metadata.csv')

event_name = session.event['EventName']

GP_ALIASES = {
    'british': 'uk',
    'great britain': 'uk',
    'britain': 'uk',
    'french': 'france',
    'german': 'germany',
    'russian': 'russia',
    'turkish': 'turkey',
    'portuguese': 'portugal',
    'tuscan': 'italy',         
    'eifel': 'germany',         
    'sakhir': 'bahrain',        
    'emilia': 'italy',        
    'styrian': 'austria',       
    '70th': 'uk',              
    'italian': 'italy',
    'spanish': 'spain',
    'hungarian': 'hungary',
    'belgian': 'belgium',
    'dutch': 'netherlands',
    'mexican': 'mexico',
    'brazilian': 'brazil',
    'australian': 'australia',
    'japanese': 'japan',
    'canadian': 'canada',
    'austrian': 'austria',
    'chinese': 'china',
    'american': 'usa',
    'united states': 'usa',
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
    st.warning(f"Track '{event_name}' not found in metadata. Using generic defaults.")
    current_track = track_meta.iloc[0] 
else:
    current_track = current_track.iloc[0]

total_laps = session.total_laps
if total_laps is None or total_laps == 0:
    if 'LapNumber' in laps.columns and len(laps) > 0:
        total_laps = int(laps['LapNumber'].max())
    else:
        total_laps = 57 
        
with st.sidebar:
    st.header("Simulation Config")
    
    n_sims = st.slider("Monte Carlo Iterations", 100, 2000, 500, step=100)
    
    st.markdown("### Decision Parameters")
    pit_loss = st.number_input(
        "Est. Pit Loss (s)", 
        value=float(current_track['estimated_green_flag_pit_loss_s']),
        step=0.1
    )
    
    run_sim = st.button("Run Simulation", type="primary")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Strategy Analysis: {current_track['track_name']}")
    
    if run_sim:
        with st.spinner("Running Monte Carlo Simulations..."):
            tools = StrategyTools(n_simulations=n_sims)
            
            strategy = tools.generate_neutral_strategy(
                current_track, 
                total_laps=total_laps
            )
            
            st.success("Simulation Complete.")
            
            if "Qatar Safety Rule" in strategy.reasoning:
                st.warning("**Qatar 2025 Rule Active**: Max 25 laps per tire set enforced. Strategies adjusted accordingly.")
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                glass_metric_card("Stops", f"{strategy.stop_count}")
            with m2:
                total_s = int(strategy.expected_time_s)
                h = total_s // 3600
                m = (total_s % 3600) // 60
                s = total_s % 60
                if h > 0:
                    time_str = f"{h}h {m}m {s}s"
                else:
                    time_str = f"{m}m {s}s"
                glass_metric_card("Time", time_str)
            with m3:
                abbrev = strategy.compound_sequence.replace('SOFT', 'S').replace('MEDIUM', 'M').replace('HARD', 'H')
                glass_metric_card("Tyres", abbrev)
            with m4:
                glass_metric_card("Conf.", f"{strategy.confidence*100:.0f}%")
            
            if strategy.alternatives:
                alt = strategy.alternatives[0]
                abbrev_alt = alt['compounds'].replace('SOFT', 'S').replace('MEDIUM', 'M').replace('HARD', 'H')
                delta = alt['delta_to_best']
                
                st.markdown(f"**Alternative Strategy:** {abbrev_alt} ({alt['stops']}-stop) · *+{delta:.1f}s slower*")
            
            st.markdown("---")
            
            st.subheader("Confidence Envelope")
            
            envelope_data = tools.get_confidence_envelope(strategy, current_track, total_laps)
            fig_env = plot_confidence_envelope(envelope_data, title=f"Strategy Confidence - {current_track['track_name']}")
            st.pyplot(fig_env, transparent=True)
            
    else:
        info_card(
            "Ready to Simulate",
            "Adjust parameters in the sidebar and click Run Simulation to generate neutral strategies."
        )
        st.info("Waiting for simulation...")

with col2:
    st.subheader("Track Intelligence")
    
    info = current_track
    overtaking_ease = 1.0 - float(info['relative_overtaking_index'])
    glass_metric_card("Overtaking", f"{overtaking_ease:.2f}", help_text="Ease Index: 0.0 (Hard) - 1.0 (Easy)")
    glass_metric_card("SC Risk", f"{info['historical_sc_risk_index']}", help_text="Historical probability factor")
    glass_metric_card("Degradation", info['expected_deg_index'])
    glass_metric_card("Aero Load", info['aero_load_category'])
    
    st.markdown("### Undercut Analyzer")
    gap = st.slider("Gap to Car Ahead (s)", 0.0, 5.0, 1.5, 0.1)
    
    if run_sim: 
        tools = StrategyTools()
        undercut = tools.analyze_undercut(
            gap=gap, 
            tire_age_self=20, 
            tire_age_rival=20, 
            pit_loss=pit_loss,
            current_lap=total_laps // 2,
            total_laps=total_laps
        )
        
        status_color = ":green" if "Undercut" in undercut.recommendation else ":red"
        st.markdown(f"**Recommendation:** {status_color}[{undercut.recommendation}]")
        st.progress(undercut.undercut_probability, text=f"Success Probability: {undercut.undercut_probability*100:.0f}%")
        
    st.markdown("---")
    
    st.subheader("The Hunter (Battle Mode)")
    
    with st.expander("Configure Battle Scenario", expanded=True):
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            st.markdown("**Hunter (Chaser)**")
            hunter_comp = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], key="h_comp")
            hunter_age = st.slider("Tire Age", 0, 40, 5, key="h_age")
        with h_col2:
            st.markdown("**Target (Leader)**")
            target_comp = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], index=1, key="t_comp")
            target_age = st.slider("Tire Age", 0, 40, 20, key="t_age")
            
        hunt_gap = st.slider("Gap to Target (s)", 0.0, 20.0, 5.0, 0.1, key="hunt_gap")
        laps_left = st.slider("Laps Remaining", 1, 30, 15, key="hunt_laps")
        
        if st.button("Project Battle", type="primary"):
            from strategy_tools import CatchUpEngine
            hunter_engine = CatchUpEngine()
            
            hunt_res = hunter_engine.analyze(
                current_gap=hunt_gap,
                laps_remaining=laps_left,
                hunter_compound=hunter_comp,
                hunter_tire_age=hunter_age,
                target_compound=target_comp,
                target_tire_age=target_age,
                track_meta=current_track
            )
            
            if hunt_res.catch_lap:
                st.success(f"**CATCH PREDICTED**: Lap {hunt_res.catch_lap} (in {hunt_res.catch_lap} laps)")
                if hunt_res.projected_pass_lap:
                    st.caption(f"Projected Overtake: Lap {hunt_res.projected_pass_lap}")
            else:
                st.error("Catch not projected within remaining laps.")
            
            st.info(f"**Analysis**: {hunt_res.reasoning}")
            
            chart_data = pd.DataFrame({
                'Lap': range(1, len(hunt_res.gap_trajectory) + 1),
                'Gap (s)': hunt_res.gap_trajectory
            })
            
            st.line_chart(chart_data, x='Lap', y='Gap (s)')