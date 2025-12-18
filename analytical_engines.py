# WTF1 Analytical Engines
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

@dataclass
class DegradationResult:
    compound: str
    deg_rate_s_per_lap: float  
    r_squared: float    
    cliff_lap: Optional[int] 
    confidence_interval: Tuple[float, float] 
    sample_size: int
    reasoning: str
    def to_dict(self) -> Dict[str, Any]:
        return {
            'compound': self.compound,
            'deg_rate': self.deg_rate_s_per_lap,
            'r_squared': self.r_squared,
            'cliff_lap': self.cliff_lap,
            'confidence_interval': self.confidence_interval,
            'sample_size': self.sample_size,
            'reasoning': self.reasoning
        }

@dataclass
class PaceAnalysisResult:
    driver: str
    raw_avg_pace: float
    fuel_corrected_pace: float
    pace_std: float
    fastest_lap: float
    consistency_score: float  
    reasoning: str

@dataclass
class TrafficAnalysisResult:
    total_laps_in_traffic: int
    total_time_lost: float
    avg_penalty_per_lap: float
    worst_stint_penalty: float
    reasoning: str

@dataclass 
class SCRiskResult:
    track_name: str
    historical_sc_probability: float
    expected_sc_laps: Tuple[int, int] 
    risk_category: str 
    strategy_impact: str 
    reasoning: str

class TireDegradationEngine:
    
    def __init__(self):
        self.typical_cliff_laps = {
            'SOFT': 18,
            'MEDIUM': 28,
            'HARD': 40
        }
        
        self.expected_deg_ranges = {
            'SOFT': (0.04, 0.15),
            'MEDIUM': (0.02, 0.10),
            'HARD': (0.01, 0.06)
        }
    
    def analyze(
        self,
        laps: pd.DataFrame,
        compound: Optional[str] = None
    ) -> Dict[str, DegradationResult]:
        results = {}
        compound_col = 'CompoundNorm' if 'CompoundNorm' in laps.columns else 'Compound'
        
        compounds_to_analyze = [compound] if compound else laps[compound_col].unique()
        
        for comp in compounds_to_analyze:
            if comp == 'UNKNOWN':
                continue
            
            compound_laps = laps[laps[compound_col] == comp]
            if len(compound_laps) < 5:  
                continue
            
            result = self._analyze_compound(compound_laps, comp)
            if result:
                results[comp] = result
        
        return results
    
    def _analyze_compound(
        self,
        laps: pd.DataFrame,
        compound: str
    ) -> Optional[DegradationResult]:
        time_col = 'LapTimeSeconds'
        age_col = 'TireAge' if 'TireAge' in laps.columns else 'StintLap'
        
        if time_col not in laps.columns or age_col not in laps.columns:
            return None
        
        data = laps[[age_col, time_col]].dropna()
        if len(data) < 5:
            return None
        
        x = data[age_col].values
        y = data[time_col].values
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        except:
            return None
        
        n = len(x)
        t_crit = stats.t.ppf(0.975, n - 2)
        ci_low = slope - t_crit * std_err
        ci_high = slope + t_crit * std_err
        
        deg_rate = max(0, slope)
        
        cliff_lap = self._estimate_cliff(compound, deg_rate, y)
        
        exp_range = self.expected_deg_ranges.get(compound, (0.01, 0.15))
        in_range = exp_range[0] <= deg_rate <= exp_range[1]
        
        reasoning = self._generate_reasoning(
            compound, deg_rate, r_value**2, cliff_lap, n, in_range
        )
        
        return DegradationResult(
            compound=compound,
            deg_rate_s_per_lap=round(deg_rate, 4),
            r_squared=round(r_value**2, 3),
            cliff_lap=cliff_lap,
            confidence_interval=(round(ci_low, 4), round(ci_high, 4)),
            sample_size=n,
            reasoning=reasoning
        )
    
    def _estimate_cliff(
        self,
        compound: str,
        deg_rate: float,
        lap_times: np.ndarray
    ) -> Optional[int]:
        base_cliff = self.typical_cliff_laps.get(compound, 25)
        
        if deg_rate > 0.10:  
            cliff_lap = int(base_cliff * 0.7)
        elif deg_rate < 0.03: 
            cliff_lap = int(base_cliff * 1.3)
        else:
            cliff_lap = base_cliff
        
        return cliff_lap
    
    def _generate_reasoning(
        self,
        compound: str,
        deg_rate: float,
        r_squared: float,
        cliff_lap: Optional[int],
        n: int,
        in_expected_range: bool
    ) -> str:
        parts = []
        
        if deg_rate < 0.03:
            parts.append(f"Low degradation on {compound}s ({deg_rate:.3f}s/lap)")
        elif deg_rate < 0.08:
            parts.append(f"Moderate degradation on {compound}s ({deg_rate:.3f}s/lap)")
        else:
            parts.append(f"High degradation on {compound}s ({deg_rate:.3f}s/lap)")
        
        if r_squared > 0.7:
            parts.append(f"Strong linear fit (R²={r_squared:.2f})")
        elif r_squared > 0.4:
            parts.append(f"Moderate linear fit (R²={r_squared:.2f})")
        else:
            parts.append(f"Weak linear fit (R²={r_squared:.2f}) - consider non-linear effects")
        
        if cliff_lap:
            parts.append(f"Estimated cliff around lap {cliff_lap}")
        
        if n < 20:
            parts.append(f"[CAUTION: Limited data ({n} laps)]")
        
        if not in_expected_range:
            parts.append("[WARNING: Degradation outside typical range - verify data quality]")
        
        return ". ".join(parts) + "."


class FuelCorrectedPaceEngine:

    def __init__(self):
        self.max_fuel = 110.0
        self.fuel_burn_rate = 1.8
        self.fuel_penalty_per_kg = 0.035
    
    def analyze_driver_pace(
        self,
        laps: pd.DataFrame,
        driver: str
    ) -> Optional[PaceAnalysisResult]:
        driver_laps = laps[laps['Driver'] == driver].copy()
        
        if len(driver_laps) < 5:
            return None
        
        time_col = 'FuelCorrectedTime' if 'FuelCorrectedTime' in driver_laps.columns else 'LapTimeSeconds'
        raw_col = 'LapTimeSeconds'
        
        if time_col not in driver_laps.columns:
            return None
        
        clean_laps = driver_laps[driver_laps['LapType'] == 'green_flag'] if 'LapType' in driver_laps.columns else driver_laps
        times = clean_laps[time_col].dropna()
        raw_times = clean_laps[raw_col].dropna() if raw_col in clean_laps.columns else times
        
        if len(times) < 3:
            return None
        
        fuel_corrected_pace = times.mean()
        raw_avg_pace = raw_times.mean()
        pace_std = times.std()
        fastest_lap = times.min()
        
        consistency = max(0, min(1, 1 - (pace_std - 0.3) / 2.0))
        
        reasoning = self._generate_reasoning(
            driver, raw_avg_pace, fuel_corrected_pace, pace_std, fastest_lap, consistency
        )
        
        return PaceAnalysisResult(
            driver=driver,
            raw_avg_pace=round(raw_avg_pace, 3),
            fuel_corrected_pace=round(fuel_corrected_pace, 3),
            pace_std=round(pace_std, 3),
            fastest_lap=round(fastest_lap, 3),
            consistency_score=round(consistency, 2),
            reasoning=reasoning
        )
    
    def compare_drivers(
        self,
        laps: pd.DataFrame,
        drivers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if drivers is None:
            drivers = laps['Driver'].unique()
        
        results = []
        for driver in drivers:
            result = self.analyze_driver_pace(laps, driver)
            if result:
                results.append({
                    'Driver': result.driver,
                    'RawPace': result.raw_avg_pace,
                    'CorrectedPace': result.fuel_corrected_pace,
                    'FastestLap': result.fastest_lap,
                    'Consistency': result.consistency_score,
                    'PaceStd': result.pace_std
                })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        return df.sort_values('CorrectedPace').reset_index(drop=True)
    
    def _generate_reasoning(
        self,
        driver: str,
        raw_pace: float,
        corrected_pace: float,
        std: float,
        fastest: float,
        consistency: float
    ) -> str:
        fuel_effect = raw_pace - corrected_pace
        
        parts = [f"{driver}: Fuel-corrected pace {corrected_pace:.3f}s"]
        
        if abs(fuel_effect) > 0.5:
            parts.append(f"Fuel effect: {fuel_effect:+.3f}s")
        
        if consistency > 0.7:
            parts.append("Highly consistent stint")
        elif consistency < 0.3:
            parts.append("Variable pace (traffic/errors?)")
        
        gap_to_fastest = corrected_pace - fastest
        if gap_to_fastest > 0.5:
            parts.append(f"Gap to personal best: {gap_to_fastest:.3f}s")
        
        return ". ".join(parts) + "."

class TrafficDirtyAirEngine:

    TRACK_MULTIPLIERS = {
        'Monaco': 1.5,  
        'Hungary': 1.3,
        'Singapore': 1.4,
        'Zandvoort': 1.2,
        'Monza': 0.6,  
        'Spa': 0.8,
        'default': 1.0
    }
    
    def __init__(self, dirty_air_threshold: float = 1.2):
        self.threshold = dirty_air_threshold
        self.base_penalty = 0.3  
    
    def analyze(
        self,
        laps: pd.DataFrame,
        track_name: Optional[str] = None
    ) -> TrafficAnalysisResult:
        if 'CleanAirFlag' not in laps.columns:
            return TrafficAnalysisResult(
                total_laps_in_traffic=0,
                total_time_lost=0,
                avg_penalty_per_lap=0,
                worst_stint_penalty=0,
                reasoning="Traffic data not available (no gap calculations)"
            )
        
        multiplier = 1.0
        if track_name:
            for key, mult in self.TRACK_MULTIPLIERS.items():
                if key.lower() in track_name.lower():
                    multiplier = mult
                    break
        
        traffic_laps = laps[laps['CleanAirFlag'] == False]
        total_traffic_laps = len(traffic_laps)
        
        if 'TrafficPenalty' in laps.columns:
            total_time_lost = (traffic_laps['TrafficPenalty'] * multiplier).sum()
            avg_penalty = total_time_lost / total_traffic_laps if total_traffic_laps > 0 else 0
        else:
            total_time_lost = total_traffic_laps * self.base_penalty * multiplier
            avg_penalty = self.base_penalty * multiplier
        
        worst_stint_penalty = 0
        if 'Stint' in traffic_laps.columns and len(traffic_laps) > 0:
            stint_penalties = traffic_laps.groupby('Stint').size() * avg_penalty
            worst_stint_penalty = stint_penalties.max() if len(stint_penalties) > 0 else 0
        
        reasoning = self._generate_reasoning(
            total_traffic_laps, total_time_lost, avg_penalty, track_name, multiplier
        )
        
        return TrafficAnalysisResult(
            total_laps_in_traffic=total_traffic_laps,
            total_time_lost=round(total_time_lost, 2),
            avg_penalty_per_lap=round(avg_penalty, 3),
            worst_stint_penalty=round(worst_stint_penalty, 2),
            reasoning=reasoning
        )
    
    def identify_drs_trains(
        self,
        laps: pd.DataFrame,
        min_cars: int = 3,
        min_laps: int = 3
    ) -> List[Dict[str, Any]]:
        trains = []
        
        if 'GapAhead' not in laps.columns or 'Position' not in laps.columns:
            return trains
        
        for lap_num in laps['LapNumber'].unique():
            lap_data = laps[laps['LapNumber'] == lap_num].sort_values('Position')
            
            current_train = []
            for i, (_, row) in enumerate(lap_data.iterrows()):
                gap = row.get('GapAhead', None)
                if pd.notna(gap) and gap < 1.0:
                    if not current_train:
                        current_train = [row['Driver']]
                    current_train.append(row['Driver'])
                else:
                    if len(current_train) >= min_cars:
                        trains.append({
                            'lap': lap_num,
                            'cars': current_train.copy(),
                            'size': len(current_train)
                        })
                    current_train = []
        
        sustained_trains = []
        train_tracker = {}
        
        for train in trains:
            key = tuple(sorted(train['cars']))
            if key not in train_tracker:
                train_tracker[key] = []
            train_tracker[key].append(train['lap'])
        
        for cars, laps_list in train_tracker.items():
            consecutive = 1
            max_consecutive = 1
            for i in range(1, len(laps_list)):
                if laps_list[i] == laps_list[i-1] + 1:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 1
            
            if max_consecutive >= min_laps:
                sustained_trains.append({
                    'cars': list(cars),
                    'duration_laps': max_consecutive,
                    'lap_range': (min(laps_list), max(laps_list)),
                    'time_lost_estimate': max_consecutive * 0.5 
                })
        
        return sustained_trains
    
    def _generate_reasoning(
        self,
        traffic_laps: int,
        time_lost: float,
        avg_penalty: float,
        track_name: Optional[str],
        multiplier: float
    ) -> str:
        parts = []
        
        if traffic_laps > 0:
            parts.append(f"Spent {traffic_laps} laps in dirty air")
            parts.append(f"Total time lost: ~{time_lost:.1f}s")
        else:
            parts.append("Clean air for entire stint")
        
        if track_name and multiplier != 1.0:
            if multiplier > 1.0:
                parts.append(f"{track_name}: High aero sensitivity (×{multiplier})")
            else:
                parts.append(f"{track_name}: Low aero sensitivity (×{multiplier})")
        
        if avg_penalty > 0.4:
            parts.append("[Significant traffic impact on strategy]")
        
        return ". ".join(parts) + "."

class DRSImpactEngine:    
    DRS_BENEFIT_PER_ZONE = 0.15 
    
    def __init__(self):
        pass
    
    def estimate_drs_benefit(
        self,
        track_meta: pd.Series,
        within_drs_range: bool = True
    ) -> Dict[str, Any]:
        drs_zones = track_meta.get('drs_zones_typical', 2)
        overtaking_index = track_meta.get('relative_overtaking_index', 0.5)
        
        if not within_drs_range:
            return {
                'benefit_s': 0,
                'zones': drs_zones,
                'overtake_probability': 0,
                'reasoning': 'Not within DRS range (gap > 1s)'
            }
        
        benefit = drs_zones * self.DRS_BENEFIT_PER_ZONE
        
        base_overtake_prob = 0.3 
        overtake_prob = base_overtake_prob * (1 - overtaking_index) * (drs_zones / 2)
        overtake_prob = max(0.05, min(0.7, overtake_prob))
        
        reasoning = self._generate_reasoning(
            drs_zones, benefit, overtake_prob, overtaking_index
        )
        
        return {
            'benefit_s': round(benefit, 3),
            'zones': drs_zones,
            'overtake_probability': round(overtake_prob, 2),
            'reasoning': reasoning
        }
    
    def _generate_reasoning(
        self,
        zones: int,
        benefit: float,
        overtake_prob: float,
        overtaking_index: float
    ) -> str:
        parts = [
            f"{zones} DRS zone(s) providing ~{benefit:.2f}s/lap advantage"
        ]
        
        if overtaking_index > 0.7:
            parts.append("Difficult overtaking track (consider alternative strategies)")
        elif overtaking_index < 0.3:
            parts.append("Easier overtaking track (DRS very effective)")
        
        parts.append(f"Estimated overtake probability: {overtake_prob*100:.0f}% per lap")
        
        return ". ".join(parts) + "."

class SafetyCarRiskEngine:    
    def __init__(self):
        self.typical_sc_windows = {
            'high_risk': [(1, 5), (20, 35), (45, 55)],
            'medium_risk': [(1, 5), (30, 45)],
            'low_risk': [(1, 5)]
        }
    
    def assess_risk(
        self,
        track_meta: pd.Series,
        total_laps: int
    ) -> SCRiskResult:
        track_name = track_meta.get('track_name', 'Unknown')
        sc_risk = track_meta.get('historical_sc_risk_index', 0.35)
        
        if sc_risk >= 0.6:
            risk_category = 'High'
            window = self.typical_sc_windows['high_risk']
        elif sc_risk >= 0.35:
            risk_category = 'Medium'
            window = self.typical_sc_windows['medium_risk']
        else:
            risk_category = 'Low'
            window = self.typical_sc_windows['low_risk']
        
        scaled_window = self._scale_window(window, total_laps)
        
        strategy_impact = self._assess_strategy_impact(sc_risk, risk_category)
        
        reasoning = self._generate_reasoning(
            track_name, sc_risk, risk_category, scaled_window
        )
        
        return SCRiskResult(
            track_name=track_name,
            historical_sc_probability=round(sc_risk, 2),
            expected_sc_laps=scaled_window[0] if scaled_window else (0, 0),
            risk_category=risk_category,
            strategy_impact=strategy_impact,
            reasoning=reasoning
        )
    
    def _scale_window(
        self,
        windows: List[Tuple[int, int]],
        total_laps: int
    ) -> List[Tuple[int, int]]:
        scaled = []
        for start, end in windows:
            scale = total_laps / 57
            scaled.append((
                int(start * scale),
                min(int(end * scale), total_laps)
            ))
        return scaled
    
    def _assess_strategy_impact(
        self,
        sc_risk: float,
        category: str
    ) -> str:
        if category == 'High':
            return (
                "Consider more aggressive early strategy (1-stop attempt). "
                "SC likely to provide free pit opportunity. "
                "Extend strategy confidence envelope by ±15s."
            )
        elif category == 'Medium':
            return (
                "Standard strategy appropriate. "
                "Prepare for SC contingency at pit windows. "
                "Extend strategy confidence envelope by ±10s."
            )
        else:
            return (
                "Low SC probability - optimize for green flag. "
                "Minimal envelope extension needed. "
                "Focus on tire degradation and traffic management."
            )
    
    def _generate_reasoning(
        self,
        track_name: str,
        sc_risk: float,
        category: str,
        windows: List[Tuple[int, int]]
    ) -> str:
        parts = [
            f"{track_name}: {category} SC risk ({sc_risk*100:.0f}% historical rate)"
        ]
        
        if windows:
            window_str = ", ".join([f"L{s}-L{e}" for s, e in windows])
            parts.append(f"Typical SC windows: {window_str}")
        
        if category == 'High':
            parts.append("[Factor heavily into strategy planning]")
        
        return ". ".join(parts) + "."

class AnalyticalEngines:

    def __init__(self):
        self.tire_engine = TireDegradationEngine()
        self.pace_engine = FuelCorrectedPaceEngine()
        self.traffic_engine = TrafficDirtyAirEngine()
        self.drs_engine = DRSImpactEngine()
        self.sc_engine = SafetyCarRiskEngine()
    
    def analyze_tire_degradation(
        self,
        laps: pd.DataFrame
    ) -> Dict[str, DegradationResult]:
        return self.tire_engine.analyze(laps)
    
    def analyze_fuel_corrected_pace(
        self,
        laps: pd.DataFrame
    ) -> pd.DataFrame:
        return self.pace_engine.compare_drivers(laps)
    
    def analyze_traffic_impact(
        self,
        laps: pd.DataFrame,
        track_name: Optional[str] = None
    ) -> TrafficAnalysisResult:
        return self.traffic_engine.analyze(laps, track_name)
    
    def identify_drs_trains(
        self,
        laps: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        return self.traffic_engine.identify_drs_trains(laps)
    
    def estimate_drs_impact(
        self,
        track_meta: pd.Series,
        within_range: bool = True
    ) -> Dict[str, Any]:
        return self.drs_engine.estimate_drs_benefit(track_meta, within_range)
    
    def assess_sc_risk(
        self,
        track_meta: pd.Series,
        total_laps: int
    ) -> SCRiskResult:
        return self.sc_engine.assess_risk(track_meta, total_laps)
    
    def run_full_analysis(
        self,
        laps: pd.DataFrame,
        track_meta: pd.Series,
        total_laps: int
    ) -> Dict[str, Any]:
        track_name = track_meta.get('track_name', None)
        
        return {
            'tire_degradation': self.analyze_tire_degradation(laps),
            'pace_comparison': self.analyze_fuel_corrected_pace(laps),
            'traffic_analysis': self.analyze_traffic_impact(laps, track_name),
            'drs_trains': self.identify_drs_trains(laps),
            'drs_impact': self.estimate_drs_impact(track_meta),
            'sc_risk': self.assess_sc_risk(track_meta, total_laps)
        }

if __name__ == "__main__":
    print("WTF1 Analytical Engines - Testing")
    print("=" * 60)
    
    from data_engine import DataEngine
    
    try:
        engine = DataEngine()
        session = engine.load_session(2024, 'Bahrain', 'R')
        clean_laps = engine.clean_laps(session.laps)
        featured = engine.engineer_features(clean_laps, session)
        
        analytics = AnalyticalEngines()
        
        print("\n1. Testing Tire Degradation Engine...")
        tire_results = analytics.analyze_tire_degradation(featured)
        for compound, result in tire_results.items():
            print(f"   {compound}: {result.deg_rate_s_per_lap:.3f} s/lap (R²={result.r_squared})")
        
        print("\n2. Testing Fuel-Corrected Pace Engine...")
        pace_results = analytics.analyze_fuel_corrected_pace(featured)
        print(f"   Top 3 drivers by corrected pace:")
        print(pace_results.head(3).to_string())
        
        print("\n3. Testing Traffic Analysis...")
        traffic = analytics.analyze_traffic_impact(featured, 'Bahrain')
        print(f"   {traffic.reasoning}")
        
        print("\n4. Testing DRS Train Detection...")
        trains = analytics.identify_drs_trains(featured)
        print(f"   Found {len(trains)} sustained DRS trains")
        
        print("\n" + "=" * 60)
        print("All engine tests complete!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")