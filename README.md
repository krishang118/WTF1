# What's in The F1 (WTF1) - Advanced F1 Strategy & Analytics Platform

WTF1 is a comprehensive data-driven Formula 1 analysis system designed to bridge the gap between public data and professional strategy software. Built on the FastF1 API, it leverages advanced statistical modeling (Monte Carlo simulations, XGBoost degradation analysis) to provide explainable insights for real-time strategy optimization. 

## Core Modules

### 1. Strategy Studio 

The command center for race strategy planning and real-time battle management:
- Neutral Strategy Generator: Runs 1,000+ Monte Carlo simulations to determine the optimal tire strategy (e.g., Medium-Hard-Hard) with localized uncertainty envelopes.
- The Hunter (Battle Mode): A novel catch-up projector that calculates if a chasing car can overhaul a target within the remaining laps, accounting for compound deltas and tire age.
- Undercut Analyzer: Real-time gap analysis to determine if a pit stop will grant track position, calculating the required out-lap delta.
- Track Intelligence: Displays key metrics like Overtaking Difficulty (calibrated heuristics), SC Risk, and Aero Load.

### 2. Performance Lab

A deep-dive telemetry explorer for understanding vehicle and driver performance:
- Tire Degradation Models:
    - Linear: Standard decay models.
    - XGBoost: Non-linear models capturing complex interactions between fuel load, track evolution, and tire wear.
- Fuel-Corrected Pace: Isolates pure driver pace by removing the masking effect of fuel burn (estimated at ~0.06s/lap).
- Telemetry Comparison: Lap-by-lap trace analysis.

### 3. Safety Car Advisor

Real-time decision support for high-pressure SC/VSC moments:
- Pit vs Stay: Provides clear, probabilistic recommendations based on track position, tire age, and reduced pit loss under SC conditions.
- Risk vs Reward: Weighs the value of fresh tires against the cost of lost track position (traffic).

### 4. Case Studies

Retrospective analysis of famous strategic battles to validate the model's logic:
- Library: Includes Abu Dhabi 2021, Singapore 2019 (Undercut), Sakhir 2020.
- Educational: Validates what the model *would* have done versus what actually happened.

## Technical Architecture

The system is built as a modular Python application with a Streamlit frontend; these are the main python files at play:

| Component | File | Description |
| :--- | :--- | :--- |
| Entry Point | `Home.py` | Main application runner and session configuration. |
| Core Orchestrator | `wtf1_core.py` | Manages configuration and inter-module communication. |
| Data Engine | `data_engine.py` | Handles FastF1 data ingestion, caching, and cleaning. |
| Analytical Engines | `analytical_engines.py` | The physics/math layer (Fuel, Drag, Degradation engines). |
| ML Module | `ml_enhancements.py` | XGBoost implementations for advanced degradation modeling. |
| Strategy Tools | `strategy_tools.py` | High-level decision logic (Monte Carlo, SC Advisor, The Hunter). |
| UI Components | `ui/components.py` | Reusable glassmorphic widgets (Metric cards, Info cards). |
| UI Styling | `ui/styling.py` | Custom CSS injection for the "Dark Glassmorphism" theme. |
| Visualization | `visualization.py` | Specialized plotting library for F1 data. |
| Pages | `pages/*.py` | Individual views: Strategy Studio, Performance Lab, SC Advisor, Case Studies. |
| Static Data | `track_metadata.csv` | database of track parameters (Pit loss, SC risk, Overtaking index). |
| Validation | `validation.py` | Automated sanity checks and backtesting suites. |

## How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/WTF1.git
    cd WTF1
    ```
    pip install fastf1 pandas numpy scipy scikit-learn matplotlib seaborn xgboost streamlit streamlit-extras

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    streamlit run Home.py
    ```

*"WTF1 impresses not because it claims to know everything, but because it knows exactly what it doesn't know."*

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License. 
