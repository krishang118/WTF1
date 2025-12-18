# WTF1 Home Page
import streamlit as st
import pandas as pd
import fastf1
from ui.styling import apply_custom_style, load_fonts
from ui.components import glass_metric_card, info_card, section_header
from data_engine import DataEngine
from wtf1_core import WTF1Config
st.set_page_config(
    page_title="WTF1 Strategy Analytics",

    layout="wide",
    initial_sidebar_state="expanded"
)
load_fonts()
apply_custom_style()
@st.cache_data(show_spinner=False, ttl=3600)
def get_event_schedule(year):
    try:
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule['EventFormat'] != 'testing']
        return schedule
    except Exception as e:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def get_available_sessions(year, gp_name):
    try:
        event = fastf1.get_event(year, gp_name)
        sessions = []        
        possible = ['FP1', 'FP2', 'FP3', 'Q', 'S', 'SQ', 'R']
        session_names = {
            'FP1': 'Practice 1',
            'FP2': 'Practice 2', 
            'FP3': 'Practice 3',
            'Q': 'Qualifying',
            'S': 'Sprint',
            'SQ': 'Sprint Qualifying',
            'R': 'Race'
        }
        
        for sess_id in possible:
            try:
                sess = event.get_session(sess_id)
                if sess is not None:
                    sessions.append((session_names.get(sess_id, sess_id), sess_id))
            except:
                pass
        
        return sessions if sessions else [('Race', 'R')]
    except Exception as e:
        return [
            ('Race', 'R'),
            ('Qualifying', 'Q'),
            ('Practice 1', 'FP1'),
            ('Practice 2', 'FP2'),
            ('Practice 3', 'FP3')
        ]

@st.cache_data(show_spinner=False)
def load_race_data(year, gp, session_type='R'):
    engine = DataEngine()
    try:
        session = engine.load_session(year, gp, session_type)
        laps = engine.clean_laps(session.laps)
        featured = engine.engineer_features(laps, session)
        return session, featured
    except Exception as e:
        return None, str(e)

with st.sidebar:
    st.title("What's in The F1")
    st.markdown("Strategy Analytics System")
    
    st.markdown("---")
    st.markdown("### Race Control")
    
    year = st.selectbox("Season", list(range(2025, 2017, -1)), index=0)    
    schedule = get_event_schedule(year)
    if schedule is not None:
        gp_list = schedule['EventName'].tolist()
    else:
        gp_list = [
            "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
            "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
            "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix",
            "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
            "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
            "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
            "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
            "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"
        ]
    
    gp = st.selectbox("Grand Prix", gp_list, index=0)    
    available_sessions = get_available_sessions(year, gp)
    session_names = [s[0] for s in available_sessions]
    session_codes = [s[1] for s in available_sessions]
    
    session_choice = st.selectbox("Session", session_names, index=0)
    session_type = session_codes[session_names.index(session_choice)]
    
    st.markdown("---")
    
    if st.button("Load Telemetry", type="primary", use_container_width=True):
        with st.spinner(f"Loading: {gp} {year} - {session_choice}..."):
            session, result = load_race_data(year, gp, session_type)
            
            if session:
                st.session_state['session'] = session
                st.session_state['race_data'] = result
                st.session_state['session_type'] = session_choice
                st.session_state['q_segment'] = 'All' 
                st.success("Telemetry Loaded.")
            else:
                st.error(f"Failed: {result}")
    
    if 'session_type' in st.session_state and 'Qualifying' in st.session_state.get('session_type', ''):
        st.markdown("---")
        st.markdown("### Qualifying Segment")
        q_segment = st.selectbox("Filter by", ["All (Q1+Q2+Q3)", "Q1 Only", "Q2 Only", "Q3 Only"], index=0)
        st.session_state['q_segment'] = q_segment
        
        if 'race_data' in st.session_state and q_segment != "All (Q1+Q2+Q3)":
            data = st.session_state['race_data']
            if 'Driver' in data.columns:
                unique_drivers = data['Driver'].nunique()
                if q_segment == "Q3 Only":
                    q3_drivers = data.groupby('Driver')['LapNumber'].max().nlargest(10).index.tolist()
                    st.session_state['filtered_data'] = data[data['Driver'].isin(q3_drivers)]
                elif q_segment == "Q2 Only":
                    all_drivers = data['Driver'].unique()
                    q2_drivers = [d for d in all_drivers if d not in data.groupby('Driver')['LapNumber'].max().nlargest(10).index][:5]
                    st.session_state['filtered_data'] = data[data['Driver'].isin(q2_drivers)]
                elif q_segment == "Q1 Only":
                    q1_drivers = data.groupby('Driver')['LapNumber'].max().nsmallest(5).index.tolist()
                    st.session_state['filtered_data'] = data[data['Driver'].isin(q1_drivers)]
        else:
            st.session_state['filtered_data'] = st.session_state.get('race_data')    
    st.markdown("---")
    with st.expander("How to Use System"):
        st.markdown("""
        **Step 1: Select Race**  
        Choose a Year, then a GP from the dynamically populated list.
        
        **Step 2: Select Session**  
        Sessions are auto-detected from FastF1. Only available sessions are shown.
        
        **Step 3: Load Telemetry**  
        Click "Load Telemetry" to fetch data.
        
        **Step 4: Navigate Pages**  
        - **Strategy Studio**: Monte Carlo pit strategy simulations.
        - **Performance Lab**: Tire degradation & XGBoost analysis.
        - **SC Advisor**: PIT vs STAY decisions during Safety Car.
        """)

st.title(f"What's in The F1 / {gp.replace(' Grand Prix', '')} {year}")
st.markdown("### Strategy & Performance Dashboard")

if 'race_data' not in st.session_state:
    info_card(
        "Awaiting Data", 
        "Select a race from the sidebar and click 'Load Telemetry' to initialize the strategy engine."
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        glass_metric_card("System Status", "Standby", "Ready")
    with col2:
        glass_metric_card("Data Source", "FastF1 API", "Online")
    with col3:
        glass_metric_card("ML Models", "XGBoost", "Active")
        
else:
    data = st.session_state['race_data']
    session = st.session_state['session']
    session_type_loaded = st.session_state.get('session_type', 'Race')
    
    total_laps = session.total_laps
    if total_laps is None or total_laps == 0:
        if 'LapNumber' in data.columns and len(data) > 0:
            total_laps = int(data['LapNumber'].max())
        else:
            total_laps = "N/A"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            avg_track = session.weather_data['TrackTemp'].mean()
            glass_metric_card("Track (Avg)", f"{avg_track:.1f}°C", help_text="Average track surface temperature")
        except:
            glass_metric_card("Track Temp", "N/A")
    
    with col2:
        try:
            avg_air = session.weather_data['AirTemp'].mean()
            glass_metric_card("Air (Avg)", f"{avg_air:.1f}°C", help_text="Average air temperature")
        except:
            glass_metric_card("Air Temp", "N/A")
    
    with col3:
        if "Race" in session_type_loaded:
            try:
                winner = session.results.iloc[0]['Abbreviation']
            except:
                winner = "N/A"
            glass_metric_card("Winner", winner)
        else:
            try:
                polesitter = session.results.iloc[0]['Abbreviation']
            except:
                polesitter = "N/A"
            glass_metric_card("P1", polesitter)
    
    with col4:
        glass_metric_card("Total Laps", str(total_laps))
    st.markdown("---")    
    st.markdown("### Classification")
    try:
        results = session.results[['Position', 'Abbreviation', 'TeamName', 'Time', 'Points']].copy()
        results['Time'] = results['Time'].astype(str).str.replace('0 days ', '', regex=False)
        st.dataframe(results, use_container_width=True)
    except Exception as e:
        st.warning(f"Results not available for this session: {e}")

    info_card(
        "Navigation", 
        "Use the sidebar to access Strategy Studio, Performance Lab, or SC Advisor."
    )