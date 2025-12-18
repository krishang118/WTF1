# Deep dive into tire degradation analysis
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ui.styling import apply_custom_style
from ui.components import glass_metric_card, info_card
from analytical_engines import AnalyticalEngines
from ml_enhancements import XGBoostTireEngine

st.set_page_config(page_title="Performance Lab", layout="wide")
apply_custom_style()

st.title("Performance Lab")
st.markdown("*Tire Degradation Analysis: Linear vs XGBoost*")

if 'race_data' not in st.session_state:
    st.warning("No Telemetry Data Loaded.")
    st.stop()

laps = st.session_state['race_data']
session = st.session_state['session']
analytics = AnalyticalEngines()

with st.sidebar:
    st.markdown("### Analysis Config")
    drivers = sorted(laps['Driver'].unique())
    selected_driver = st.selectbox("Select Driver", drivers, index=0)
    
    compounds = sorted(laps['Compound'].unique())
    selected_compound = st.selectbox("Select Compound", compounds, index=0)

driver_laps = laps[laps['Driver'] == selected_driver]
compound_laps = driver_laps[driver_laps['Compound'] == selected_compound]

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Degradation Analysis: {selected_driver} on {selected_compound}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if len(compound_laps) > 0:
        ax.scatter(compound_laps['TireAge'], compound_laps['LapTimeSeconds'], 
                   color='white', alpha=0.5, label='Actual Laps')
    
    deg_per_lap = 0.0
    if len(compound_laps) > 5:
        z = np.polyfit(compound_laps['TireAge'], compound_laps['LapTimeSeconds'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(compound_laps['TireAge'].min(), compound_laps['TireAge'].max() + 10, 50)
        ax.plot(x_range, p(x_range), color='red', linestyle='--', label='Linear Model')
        deg_per_lap = z[0]
        
    try:
        xgb_engine = XGBoostTireEngine()
        xgb_engine.train_model(laps) 
        start_fuel = compound_laps['FuelEstimate'].max() if 'FuelEstimate' in compound_laps.columns else 30.0
        curve = xgb_engine.predict_degradation_curve(selected_compound, max_laps=40, start_fuel=start_fuel)
        
        if curve:
            ax.plot(curve['laps'], curve['predicted_times'], color='#00ff00', linewidth=2, label='XGBoost (Non-Linear)')
    except Exception as e:
        st.warning(f"ML Model unavailable: {e}")

    ax.set_xlabel('Tire Age (Laps)', color='white')
    ax.set_ylabel('Lap Time (seconds)', color='white')
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white')
    
    st.pyplot(fig, transparent=True)

with col2:
    st.subheader("Model Stats")
    glass_metric_card("Linear Deg", f"{deg_per_lap:.3f} s/lap")
    glass_metric_card("Data Points", f"{len(compound_laps)}")
    
    info_card(
        "Model Comparison",
        "Linear: Assumes constant wear rate.\n\nXGBoost: Captures non-linear tire/fuel interactions."
    )
    with st.expander("How to Interpret"):
        st.markdown("""
        - Positive degradation: Tire slowing down (normal)
        - Flat curve: Low deg compound/track
        - Steep curve: High deg (pit early)
        - XGBoost curve: Shows fuel-burn masking effect
        """)