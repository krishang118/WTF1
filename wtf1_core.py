# WTF1 Main Module
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class FuelConfig:
    MAX_FUEL_KG: float = 110.0
    BURN_RATE_KG_PER_LAP: float = 1.8
    TIME_PENALTY_S_PER_10KG: float = 0.35
    
    def get_fuel_at_lap(self, lap_number: int, total_laps: int) -> float:
        consumed = lap_number * self.BURN_RATE_KG_PER_LAP
        return max(0.0, self.MAX_FUEL_KG - consumed)
    
    def get_fuel_penalty(self, fuel_kg: float) -> float:
        return fuel_kg * (self.TIME_PENALTY_S_PER_10KG / 10.0)

@dataclass(frozen=True)
class TireConfig:
    COMPOUNDS: Tuple[str, ...] = ('SOFT', 'MEDIUM', 'HARD')
    COMPOUND_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'SOFT': '#FF0000',     
        'MEDIUM': '#FFD700',    
        'HARD': '#FFFFFF',     
        'INTERMEDIATE': '#00FF00',  
        'WET': '#0000FF'       
    })
    BASE_DEG_RATES: Dict[str, float] = field(default_factory=lambda: {
        'SOFT': 0.08,
        'MEDIUM': 0.05,
        'HARD': 0.03
    })
    CLIFF_MULTIPLIERS: Dict[str, float] = field(default_factory=lambda: {
        'SOFT': 18,
        'MEDIUM': 28,
        'HARD': 40
    })

def _default_compound_colors():
    return {
        'SOFT': '#FF0000',
        'MEDIUM': '#FFD700',
        'HARD': '#FFFFFF',
        'INTERMEDIATE': '#00FF00',
        'WET': '#0000FF'
    }

def _default_deg_rates():
    return {
        'SOFT': 0.08,
        'MEDIUM': 0.05,
        'HARD': 0.03
    }

def _default_cliff_multipliers():
    return {
        'SOFT': 18,
        'MEDIUM': 28,
        'HARD': 40
    }

@dataclass
class TireConfigMutable:
    COMPOUNDS: Tuple[str, ...] = ('SOFT', 'MEDIUM', 'HARD')
    COMPOUND_COLORS: Dict[str, str] = field(default_factory=_default_compound_colors)
    BASE_DEG_RATES: Dict[str, float] = field(default_factory=_default_deg_rates)
    CLIFF_MULTIPLIERS: Dict[str, float] = field(default_factory=_default_cliff_multipliers)

@dataclass(frozen=True)
class TrafficConfig:
    DIRTY_AIR_THRESHOLD_S: float = 1.2 
    DIRTY_AIR_PENALTY_S: float = 0.3 
    DRS_THRESHOLD_S: float = 1.0 
    DRS_BENEFIT_S: float = 0.4 

@dataclass(frozen=True)
class SimulationConfig:
    N_SIMULATIONS: int = 1000 
    DEG_VARIANCE: float = 0.02 
    PIT_WINDOW_VARIANCE: int = 2 
    RANDOM_SEED: Optional[int] = 42 

class DataLabel(Enum):
    FACTUAL = "FACTUAL" 
    MODEL_DERIVED = "MODEL_DERIVED" 
    HEURISTIC = "HEURISTIC" 

class WTF1Config:
    PROJECT_ROOT = Path(__file__).parent
    CACHE_DIR = PROJECT_ROOT / "cache"
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    TRACK_METADATA_FILE = PROJECT_ROOT / "track_metadata.csv"    
    SEASON_START = 2018
    SEASON_END = 2024
    SUPPORTED_SESSIONS = ('FP1', 'FP2', 'FP3', 'Q', 'SQ', 'R', 'SS')    
    fuel = FuelConfig()
    tire = TireConfigMutable()
    traffic = TrafficConfig()
    simulation = SimulationConfig()
    @classmethod
    def setup(cls) -> None:
        cls.CACHE_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)        
        try:
            import fastf1
            fastf1.Cache.enable_cache(str(cls.CACHE_DIR))
        except ImportError:
            warnings.warn("FastF1 not installed. Run: pip install fastf1")

    @classmethod
    def load_track_metadata(cls) -> pd.DataFrame:
        if not cls.TRACK_METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Track metadata not found at {cls.TRACK_METADATA_FILE}. "
                "Ensure track_metadata.csv is in the project root."
            )

        df = pd.read_csv(cls.TRACK_METADATA_FILE)
        
        df.attrs['column_labels'] = {
            'track_name': DataLabel.FACTUAL,
            'country': DataLabel.FACTUAL,
            'length_km': DataLabel.FACTUAL,
            'num_corners': DataLabel.FACTUAL,
            'estimated_green_flag_pit_loss_s': DataLabel.MODEL_DERIVED,
            'drs_zones_typical': DataLabel.MODEL_DERIVED,
            'relative_overtaking_index': DataLabel.HEURISTIC,
            'aero_load_category': DataLabel.HEURISTIC,
            'expected_deg_index': DataLabel.HEURISTIC,
            'historical_sc_risk_index': DataLabel.HEURISTIC
        }
        
        return df

SYSTEM_ASSUMPTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        WTF1 SYSTEM ASSUMPTIONS                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FUEL MODEL:                                                                 ║
║  • Burn rate: ~1.8 kg/lap [ESTIMATED, varies by track/driving style]        ║
║  • Weight penalty: ~0.035s per kg [ESTIMATED from historical data]          ║
║  • Max fuel: 110 kg (FIA regulation) [FACTUAL]                              ║
║                                                                              ║
║  TIRE MODEL:                                                                 ║
║  • Linear degradation assumed (reality: often non-linear)                   ║
║  • Cliff onset estimated per compound [HEURISTIC]                           ║
║  • No car-specific tire behavior modeled                                    ║
║  • Track temperature effects are averaged, not precise                      ║
║                                                                              ║
║  TRAFFIC MODEL:                                                              ║
║  • Dirty air threshold: 1.2s gap [ESTIMATED]                                ║
║  • Dirty air penalty: ~0.3s/lap base [ESTIMATED, track-dependent]           ║
║  • DRS benefit: ~0.4s average [ESTIMATED, highly variable]                  ║
║                                                                              ║
║  STRATEGY MODEL:                                                             ║
║  • "Neutral strategy" = average driver, no team orders                      ║
║  • Pit loss estimates have ±1-2s variance                                   ║
║  • Safety car timing is probabilistic, not predictive                       ║
║  • No car performance differences modeled                                   ║
║                                                                              ║
║  DATA SOURCE:                                                                ║
║  • FastF1 only (public timing data with ~5 day delay)                       ║
║  • No proprietary telemetry or team data                                    ║
║  • Track metadata is curated, not scraped                                   ║
║                                                                              ║
║  LIMITATIONS:                                                                ║
║  • Accuracy: ±8s over full race (teams need ±2s)                            ║
║  • No real-time capability (post-race analysis only)                        ║
║  • No driver skill or psychology modeling                                   ║
║  • No weather prediction (uses historical conditions)                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
ETHICAL_BOUNDARIES = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ETHICAL BOUNDARIES                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ✓ READ-ONLY: This system analyzes, never controls                          ║
║  ✓ PUBLIC DATA: All sources are reproducible from FastF1                    ║
║  ✓ HONEST: Never claims access to team secrets or FIA internals            ║
║  ✓ EXPLAINABLE: Every recommendation includes reasoning                     ║
║  ✓ HUMBLE: States "model-optimal", never "team-optimal"                     ║
║                                                                              ║
║  ✗ NO LIVE CONTROL: Cannot make race-day decisions                          ║
║  ✗ NO SUPERIORITY CLAIMS: Does not compete with team engineers             ║
║  ✗ NO MANIPULATION: Cannot influence race outcomes                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
def print_system_info() -> None:
    print(SYSTEM_ASSUMPTIONS)
    print(ETHICAL_BOUNDARIES)

def run_quick_analysis(
    track: str,
    year: int,
    session: str = 'R',
    verbose: bool = True
) -> Dict[str, Any]:

    from data_engine import DataEngine
    from analytical_engines import AnalyticalEngines
    from strategy_tools import StrategyTools
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"WTF1 ANALYSIS: {track} {year} - Session: {session}")
        print(f"{'='*70}\n")
    
    WTF1Config.setup()
    
    engine = DataEngine()
    session_data = engine.load_session(year, track, session)
    clean_laps = engine.clean_laps(session_data.laps)
    featured_laps = engine.engineer_features(clean_laps, session_data)
    
    analytics = AnalyticalEngines()
    tire_analysis = analytics.analyze_tire_degradation(featured_laps)
    pace_analysis = analytics.analyze_fuel_corrected_pace(featured_laps)
    traffic_analysis = analytics.analyze_traffic_impact(featured_laps)
    
    strategy = StrategyTools()
    track_meta = WTF1Config.load_track_metadata()
    track_info = track_meta[track_meta['track_name'].str.contains(track, case=False)]
    
    if len(track_info) == 0:
        raise ValueError(f"Track '{track}' not found in metadata")
    
    optimal_strategy = strategy.generate_neutral_strategy(
        track_info=track_info.iloc[0],
        total_laps=len(clean_laps['LapNumber'].unique()),
        tire_analysis=tire_analysis
    )
    
    results = {
        'track': track,
        'year': year,
        'session': session,
        'tire_analysis': tire_analysis,
        'pace_analysis': pace_analysis,
        'traffic_analysis': traffic_analysis,
        'optimal_strategy': optimal_strategy
    }
    
    if verbose:
        print_strategy_output(optimal_strategy, track, year)
    
    return results

def print_strategy_output(strategy: Dict[str, Any], track: str, year: int) -> None:
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{track} {year} - Neutral Strategy (Model-Optimal)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Optimal Stop Count: {strategy.get('stop_count', 'N/A')}-stop
Pit Windows:        {strategy.get('pit_windows', 'N/A')}
Compound Sequence:  {strategy.get('compound_sequence', 'N/A')}
Expected Race Time: {strategy.get('expected_time', 'N/A')} ±{strategy.get('uncertainty', 'N/A')}

Confidence Envelope:
  ├── Best case:  {strategy.get('best_case', 'N/A')}
  ├── Mean:       {strategy.get('mean_case', 'N/A')}
  └── Worst case: {strategy.get('worst_case', 'N/A')}

Reasoning:
{strategy.get('reasoning', 'No reasoning available')}

[NOTE: This is MODEL-OPTIMAL based on neutral conditions. 
 Actual team strategies may differ due to car-specific factors,
 driver preferences, and tactical considerations.]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='WTF1 - Strategy-Grade F1 Analytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wtf1_core.py --track Bahrain --year 2024 --session R
  python wtf1_core.py --track Monaco --year 2023 --session Q
  python wtf1_core.py --info  # Show system assumptions
        """
    )
    parser.add_argument('--track', type=str, help='Track name (e.g., Bahrain, Monaco)')
    parser.add_argument('--year', type=int, help='Season year (2018-2024)')
    parser.add_argument('--session', type=str, default='R',
                        help='Session type: R (race), Q (quali), FP1/2/3')
    parser.add_argument('--info', action='store_true',
                        help='Show system assumptions and ethics')
    parser.add_argument('--quick-analysis', action='store_true',
                        help='Run quick analysis for specified track/year')
    args = parser.parse_args()
    if args.info:
        print_system_info()
        return
    if args.track and args.year:
        run_quick_analysis(args.track, args.year, args.session)
    else:
        print("WTF1 - What's in The F1")
        print("Use --help for usage information")
        print_system_info()
if __name__ == "__main__":
    main()