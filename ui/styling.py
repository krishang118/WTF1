# Handles CSS injection
import streamlit as st

def apply_custom_style():
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #000000;
            background-image: none;
        }
        
        div[data-testid="stMetric"], div.stDataFrame {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
            border-color: rgba(255, 255, 255, 0.8); 
        }
        
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.5px;
        }
        
        span[data-testid="stMetricLabel"] {
            color: rgba(255, 255, 255, 0.6) !important;
            font-size: 0.9rem !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-family: 'Outfit', sans-serif; 
            font-weight: 600;
        }
        
        [data-testid="stSidebar"] {
            background-color: #000000;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        button[kind="secondary"] {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            transition: all 0.2s !important;
        }
        
        button[kind="secondary"]:hover {
            border-color: #ffffff !important;
            color: #ffffff !important;
        }
        
        .js-plotly-plot .plotly .main-svg {
            background: transparent !important;
        }
        
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:hover,
        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="base-input"] > div:hover,
        div[data-baseweb="base-input"] > div:focus-within {
            border-color: #ffffff !important;
        }        
        </style>
    """, unsafe_allow_html=True)

def load_fonts():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@400;600;700&display=swap');
        </style>
    """, unsafe_allow_html=True)