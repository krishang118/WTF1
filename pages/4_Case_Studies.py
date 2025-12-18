# Historical race analysis and strategy replay
import streamlit as st
import pandas as pd
from ui.styling import apply_custom_style
from ui.components import glass_metric_card, info_card
from strategy_tools import CaseStudyEngine

st.set_page_config(page_title="Case Studies", layout="wide")
apply_custom_style()

st.title("Case Studies")
st.markdown("*Retrospective Analysis of Famous Strategic Battles*")

@st.cache_resource
def get_engine():
    return CaseStudyEngine()

engine = get_engine()

with st.sidebar:
    st.header("Case Selection")
    
    cases = {
        '2021_Abu_Dhabi': 'Abu Dhabi 2021 (The Decider)',
        '2019_Singapore': 'Singapore 2019 (The Undercut)',
        '2020_Sakhir': 'Sakhir 2020 (Mercedes Chaos)'
    }
    
    selected_key = st.selectbox(
        "Choose Race",
        options=list(cases.keys()),
        format_func=lambda x: cases[x]
    )
    
    run_btn = st.button("Load Case Study", type="primary")

    st.info("""
    **Note:** This analysis uses historical data and WTF1's neutral strategy model to evaluate decisions made in the heat of the moment.
    """)
if run_btn:
    with st.spinner(f"Loading data for {cases[selected_key]}..."):
        try:
            track_meta_df = pd.read_csv("track_metadata.csv")    
            track_map = {
                '2021_Abu_Dhabi': 'Yas Marina Circuit',
                '2019_Singapore': 'Marina Bay Street Circuit',
                '2020_Sakhir': 'Bahrain International Circuit (Outer)'
            }
            
            target_track = track_map.get(selected_key)
            if target_track:
                track_row = track_meta_df[track_meta_df['track_name'] == target_track].iloc[0]
            else:
                track_row = None
            
            result = engine.analyze_case(selected_key, track_meta=track_row)
            
        except Exception as e:
            st.error(f"Error loading case: {e}")
            st.stop()

    race_info = result['race_info']
    st.header(f"{race_info['year']} {race_info['gp']} Grand Prix")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Key Moment")
        st.warning(f"**Event:** {race_info['key_moment']}")
        if race_info.get('controversy') and race_info['controversy'] != 'N/A':
            st.markdown(f"**Controversy:** {race_info['controversy']}")
        
        st.markdown("### Key Decisions Timeline")
        for lap, title, desc in result['key_decisions']:
            st.markdown(f"""
            <div style="
                background-color: rgba(255, 255, 255, 0.05);
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                border-left: 3px solid #ff4b4b;">
                <strong>{lap}</strong>: {title}<br>
                <span style="color: #888; font-size: 0.9em;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.subheader("WTF1 Model View")
        if 'model_recommendation' in result:
            rec = result['model_recommendation']
            glass_metric_card("Optimal Stops", str(rec['stop_count']))
            total_seconds = int(rec['expected_time'])
            m, s = divmod(total_seconds, 60)
            h, m = divmod(m, 60)
            if h > 0:
                time_str = f"{h}h {m}m {s}s"
            else:
                time_str = f"{m}m {s}s"
            glass_metric_card("Expected Time", time_str)
            
            st.markdown("**Reasoning:**")
            st.caption(rec['reasoning'])
        else:
            st.info("Model analysis not available for this case.")
            
    st.markdown("---")    
    st.subheader("Strategic Lessons")
    for lesson in result['lessons']:
        st.success(lesson)

    st.markdown("---")
    st.caption(result.get('disclaimer', ''))

else:
    info_card(
        "Select a Case Study",
        "Choose a famous historical race from the sidebar to analyze the strategic decisions made."
    )