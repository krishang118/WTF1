# What's in The F1 (WTF1) - Advanced F1 Strategy & Analytics Platform

WTF1 is a comprehensive data-driven Formula 1 analysis system designed to bridge the gap between public data and professional strategy software. Built on the FastF1 API, it leverages advanced statistical modeling (Monte Carlo simulations, XGBoost degradation analysis) to provide explainable insights for real-time strategy optimization. 

###

<p align="center">
  <img src="demos/Demo1.gif" width="800" />
</p>

<p align="center">
  <img src="demos/Demo3.gif" width="400" />
  <img src="demos/Demo5.gif" width="400" />
</p>

## Core Modules

### 1. Strategy Studio 

The command center for race strategy planning and real-time battle management:
- Neutral Strategy Generator: Runs 1,000+ Monte Carlo simulations to determine the optimal tire strategy (for example, Medium-Hard-Hard) with localized uncertainty envelopes (even has support for the 2025 GP Qatar 2-stop rule).
- The Hunter (Battle Mode): A novel catch-up projector that calculates if a chasing car can overhaul a target within the remaining laps, accounting for compound deltas and tire age.
- Undercut Analyzer: Real-time gap analysis to determine if a pit stop will grant track position, calculating the required out-lap delta.
- Track Intelligence: Displays key metrics like Overtaking Difficulty (calibrated heuristics), SC Risk, and Aero Load.

###

<p align="center">
  <img src="demos/Demo6.gif" width="800" />
</p>

<p align="center">
  <img src="demos/Demo7.gif" width="680" />
  <img src="demos/Demo8.gif" width="680" />
</p>

### 2. Performance Lab

A deep-dive telemetry explorer for understanding vehicle and driver performance:
- Tire Degradation Models:
    - Linear: Standard decay models.
    - XGBoost: Non-linear models capturing complex interactions between fuel load, track evolution, and tire wear.
- Fuel-Corrected Pace: Isolates pure driver pace by removing the masking effect of fuel burn (estimated at ~0.06s/lap).
- Telemetry Comparison: Lap-by-lap trace analysis.

###

<p align="center">
  <img src="demos/Demo9.gif" width="800" />
</p>

### 3. Safety Car Advisor

Real-time decision support for high-pressure SC/VSC moments:
- Pit vs Stay: Provides clear, probabilistic recommendations based on track position, tire age, and reduced pit loss under SC conditions.
- Risk vs Reward: Weighs the value of fresh tires against the cost of lost track position (traffic).

###

<p align="center">
  <img src="demos/Demo10.gif" width="800" />
  <img src="demos/Demo11.gif" width="800" />
</p>

### 4. Case Studies

Retrospective analysis of famous strategic battles to validate the model's logic:
- Library: Includes Abu Dhabi 2021, Singapore 2019 (Undercut), Sakhir 2020.
- Educational: Validates what the model *would* have done versus what actually happened.

###

<p align="center">
  <img src="demos/Demo12.gif" width="800" />
</p>

## Technical Architecture

The system is built as a modular Python application with a Streamlit frontend; these are the main files at play:

| Component | File | Description |
| :--- | :--- | :--- |
| Entry Point | `Home.py` | Main application runner and session configuration |
| Core Orchestrator | `wtf1_core.py` | Manages configuration and inter-module communication |
| Data Engine | `data_engine.py` | Handles FastF1 data ingestion, caching, and cleaning |
| Analytical Engines | `analytical_engines.py` | The physics/math layer (Fuel, Drag, Degradation engines) |
| ML Module | `ml_enhancements.py` | XGBoost implementations for advanced degradation modeling |
| Strategy Tools | `strategy_tools.py` | High-level decision logic (Monte Carlo, SC Advisor, The Hunter) |
| UI Components | `ui/components.py` | Reusable widgets (Metric cards, Info cards) |
| UI Styling | `ui/styling.py` | Custom stylistic CSS injection |
| Visualization | `visualization.py` | Specialized plotting library for F1 data |
| Pages | `pages/*.py` | Individual views: Strategy Studio, Performance Lab, SC Advisor, Case Studies |
| Static Data | `track_metadata.csv` | Database of track parameters (Pit loss, SC risk, Overtaking index) |
| Validation | `validation.py` | Automated sanity checks and backtesting suites |

## How to Run

1. Make sure Python 3.8+ is installed.
2. Clone this repository on your local machine.
3. Install the required dependencies:
    ```bash
    pip install fastf1 pandas numpy scipy scikit-learn matplotlib seaborn xgboost streamlit streamlit-extras
    ```
4.  Run the application:
    ```bash
    streamlit run Home.py
    ```
    
## Contributing

Contributions are welcome!

## License

Distributed under the MIT License. 
