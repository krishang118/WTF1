# Reusable UI widgets for the interface
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

def glass_metric_card(label: str, value: str, delta: str = None, help_text: str = None):
    delta_html = ""
    if delta:
        delta_html = f'<div style="color: #4CAF50; font-size: 0.8rem; margin-top: 4px; font-weight: 500;">{delta}</div>'

    st.markdown(f"""
    <div style="
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    ">
        <div style="
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'Inter', sans-serif;
            margin-bottom: 5px;
        ">{label}</div>
        <div style="
            color: #ffffff;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1.8rem;
            line-height: 1.2;
        ">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def info_card(title: str, content: str):
    st.markdown(f"""
    <div style="
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #FF1E1E;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    ">
        <h4 style="margin-top: 0; color: #FF1E1E;">{title}</h4>
        <p style="margin-bottom: 0; font-size: 0.95rem; color: #ddd;">{content}</p>
    </div>
    """, unsafe_allow_html=True)

def section_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(subtitle)
    st.markdown("---")