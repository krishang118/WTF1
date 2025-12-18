# WTF1 Strategy Tools
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import warnings
from itertools import product

import pandas as pd
import numpy as np

class StrategyType(Enum):
    ONE_STOP = 1
    TWO_STOP = 2
    THREE_STOP = 3

@dataclass
class StrategyResult:
    stop_count: int
    pit_windows: List[Tuple[int, int]]
    compound_sequence: str
    expected_time_s: float
    uncertainty_s: float
    best_case_s: float
    worst_case_s: float
    reasoning: str
    alternatives: List[Dict[str, Any]]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'stop_count': self.stop_count,
            'pit_windows': self.pit_windows,
            'compound_sequence': self.compound_sequence,
            'expected_time': self.expected_time_s,
            'uncertainty': self.uncertainty_s,
            'best_case': self.best_case_s,
            'worst_case': self.worst_case_s,
            'reasoning': self.reasoning,
            'alternatives': self.alternatives,
            'confidence': self.confidence
        }

@dataclass
class UndercutResult:
    undercut_probability: float
    required_outlap_delta: float
    overcut_window: Tuple[int, int]
    recommendation: str
    reasoning: str
    gap_evolution: List[float]

@dataclass
class SCDecisionResult:
    recommendation: str
    confidence: float
    reasoning: str
    alternative_scenario: str
    expected_positions: Dict[str, int]
    
@dataclass
class DRSEscapeResult:
    strategies: List[Dict[str, Any]]
    recommended: str
    time_saved: float
    reasoning: str

class NeutralStrategyGenerator:
    COMPOUND_PACE_OFFSET = {
        'SOFT': -0.8,   
        'MEDIUM': 0.0,   
        'HARD': 0.4      
    }
    
    COMPOUND_DEG_RATE = {
        'SOFT': 0.08,    
        'MEDIUM': 0.05,
        'HARD': 0.03
    }
    
    COMPOUND_LIFE = {
        'SOFT': 20,
        'MEDIUM': 30,
        'HARD': 45
    }
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        np.random.seed(42) 
    
    def generate(
        self,
        track_meta: pd.Series,
        total_laps: int,
        tire_analysis: Optional[Dict] = None,
        sc_risk: float = 0.35
    ) -> StrategyResult:

        pit_loss = track_meta.get('estimated_green_flag_pit_loss_s', 23.0)
        deg_index = track_meta.get('expected_deg_index', 'Medium')
        
        is_qatar = 'Qatar' in str(track_meta.get('country', '')) or 'Losail' in str(track_meta.get('track_name', ''))
        qatar_limit = 25
        rule_active = is_qatar and total_laps > qatar_limit
        
        deg_multiplier = self._get_deg_multiplier(deg_index)
        
        if tire_analysis:
            deg_rates = {
                comp: result.deg_rate_s_per_lap
                for comp, result in tire_analysis.items()
            }
        else:
            deg_rates = {
                comp: rate * deg_multiplier
                for comp, rate in self.COMPOUND_DEG_RATE.items()
            }
        
        strategies = []
        
        for start, end in product(['SOFT', 'MEDIUM'], ['MEDIUM', 'HARD']):
            if start != end:  
                result = self._evaluate_strategy(
                    [start, end],
                    total_laps,
                    pit_loss,
                    deg_rates,
                    sc_risk,
                    1
                )
                
                avg_stint = total_laps / 2
                if not (rule_active and avg_stint > qatar_limit):
                    strategies.append(result)
        
        for compounds in [
            ['SOFT', 'MEDIUM', 'HARD'],
            ['SOFT', 'HARD', 'MEDIUM'],
            ['MEDIUM', 'HARD', 'SOFT'],
            ['SOFT', 'MEDIUM', 'MEDIUM'],
            ['MEDIUM', 'MEDIUM', 'HARD'],
        ]:
            result = self._evaluate_strategy(
                compounds,
                total_laps,
                pit_loss,
                deg_rates,
                sc_risk,
                2
            )
            
            avg_stint = total_laps / 3
            if not (rule_active and avg_stint > qatar_limit):
                strategies.append(result)
        
        if deg_multiplier > 1.0:
            for compounds in [
                ['SOFT', 'MEDIUM', 'MEDIUM', 'HARD'],
                ['SOFT', 'SOFT', 'MEDIUM', 'HARD'],
            ]:
                result = self._evaluate_strategy(
                    compounds,
                    total_laps,
                    pit_loss,
                    deg_rates,
                    sc_risk,
                    3
                )
                strategies.append(result)
        
        strategies.sort(key=lambda x: x['expected_time'])
        
        best = strategies[0]
        alternatives = strategies[1:4]
        
        reasoning = self._generate_reasoning(best, alternatives, pit_loss, deg_rates, rule_active)
        
        return StrategyResult(
            stop_count=best['stops'],
            pit_windows=best['pit_windows'],
            compound_sequence=best['compounds'],
            expected_time_s=round(best['expected_time'], 1),
            uncertainty_s=round(best['uncertainty'], 1),
            best_case_s=round(best['best_case'], 1),
            worst_case_s=round(best['worst_case'], 1),
            reasoning=reasoning,
            alternatives=[
                {
                    'compounds': alt['compounds'],
                    'stops': alt['stops'],
                    'expected_time': round(alt['expected_time'], 1),
                    'delta_to_best': round(alt['expected_time'] - best['expected_time'], 1)
                }
                for alt in alternatives
            ],
            confidence=self._calculate_confidence(best, alternatives, sc_risk)
        )
    
    def _get_deg_multiplier(self, deg_index: str) -> float:
        multipliers = {
            'Low': 0.7,
            'Low-Medium': 0.85,
            'Medium': 1.0,
            'Medium-High': 1.15,
            'High': 1.3
        }
        return multipliers.get(deg_index, 1.0)
    
    def _evaluate_strategy(
        self,
        compounds: List[str],
        total_laps: int,
        pit_loss: float,
        deg_rates: Dict[str, float],
        sc_risk: float,
        num_stops: int
    ) -> Dict[str, Any]:
        n_stints = len(compounds)
        stint_length = total_laps // n_stints
        
        pit_windows = []
        current_lap = 0
        for i in range(n_stints - 1):
            compound = compounds[i]
            max_life = self.COMPOUND_LIFE.get(compound, 25)
            
            optimal_pit = min(current_lap + max_life, current_lap + stint_length)
            window_start = max(1, optimal_pit - 3)
            window_end = min(total_laps - 5, optimal_pit + 3)
            pit_windows.append((window_start, window_end))
            current_lap = optimal_pit
        
        race_times = []
        
        for _ in range(self.n_simulations):
            total_time = 0
            lap_in_stint = 1
            
            for i, compound in enumerate(compounds):
                if i < len(pit_windows):
                    stint_laps = pit_windows[i][0] - (pit_windows[i-1][0] if i > 0 else 0)
                    stint_laps = max(1, stint_laps)
                else:
                    stint_laps = total_laps - (pit_windows[-1][0] if pit_windows else 0)
                
                base_pace = 90.0  
                pace_offset = self.COMPOUND_PACE_OFFSET.get(compound, 0)
                
                base_deg = deg_rates.get(compound, 0.05)
                deg_variance = np.random.normal(0, 0.02)  
                actual_deg = max(0, base_deg + deg_variance)
                stint_time = 0
                for lap in range(1, stint_laps + 1):
                    lap_time = base_pace + pace_offset + (actual_deg * lap)
                    lap_time += np.random.normal(0, 0.3)
                    stint_time += lap_time
                
                total_time += stint_time
            
            for _ in range(num_stops):
                pit_time = pit_loss + np.random.normal(0, 0.5)
                total_time += pit_time
            
            if np.random.random() < sc_risk:
                total_time -= pit_loss * 0.7  
            
            race_times.append(total_time)
        
        race_times = np.array(race_times)
        
        return {
            'compounds': '-'.join(compounds),
            'stops': num_stops,
            'pit_windows': pit_windows,
            'expected_time': np.mean(race_times),
            'uncertainty': np.std(race_times),
            'best_case': np.percentile(race_times, 5),
            'worst_case': np.percentile(race_times, 95)
        }
    
    def _generate_reasoning(
        self,
        best: Dict,
        alternatives: List[Dict],
        pit_loss: float,
        deg_rates: Dict,
        qatar_rule: bool = False
    ) -> str:
        parts = []
        
        if qatar_rule:
            parts.append("  • Qatar Safety Rule enforced: Max 25 laps/set")
        
        total_pit_cost = best['stops'] * pit_loss
        parts.append(f"  • {best['stops']}-stop optimal: pit loss = {total_pit_cost:.1f}s")
        
        compounds = best['compounds'].split('-')
        for comp in compounds:
            deg = deg_rates.get(comp, 0.05)
            parts.append(f"  • {comp}: {deg:.3f}s/lap degradation")
        
        if alternatives:
            alt = alternatives[0]
            delta = alt['expected_time'] - best['expected_time']
            parts.append(f"  • Next best ({alt['compounds']}): +{delta:.1f}s slower")
        
        if best['pit_windows']:
            windows = ', '.join([f"L{w[0]}-L{w[1]}" for w in best['pit_windows']])
            parts.append(f"  • Optimal pit windows: {windows}")
        
        return '\n'.join(parts)
    
    def _calculate_confidence(
        self,
        best: Dict,
        alternatives: List[Dict],
        sc_risk: float
    ) -> float:
        confidence = 0.80
        
        if alternatives:
            gap = alternatives[0]['expected_time'] - best['expected_time']
            if gap > 15:
                confidence += 0.10
            elif gap > 5:
                confidence += 0.05
            elif gap > 2:
                confidence += 0.02
        
        if sc_risk > 0.6:
            confidence -= 0.15
        elif sc_risk > 0.4:
            confidence -= 0.05
        
        if best['uncertainty'] > 25:
            confidence -= 0.1
        elif best['uncertainty'] < 8:
            confidence += 0.05
        
        return max(0.4, min(0.99, confidence))
        
class UndercutAnalyzer:

    def __init__(self):
        self.fresh_tire_advantage = 1.2  
        self.cold_tire_penalty = 0.8     
    
    def analyze(
        self,
        gap_ahead: float,
        tire_age_self: int,
        tire_age_rival: int,
        pit_loss: float,
        current_lap: int,
        total_laps: int,
        traffic_behind: bool = False
    ) -> UndercutResult:
        fresh_tire_gain = self.fresh_tire_advantage * 1.5  
        
        if traffic_behind:
            traffic_penalty = 1.5
        else:
            traffic_penalty = 0
            
        required_delta = fresh_tire_gain - gap_ahead - traffic_penalty
        
        if required_delta > 0:
            if required_delta > 2.0:
                probability = 0.9
            elif required_delta > 1.0:
                probability = 0.75
            else:
                probability = 0.5
        else:
            probability = max(0.1, 0.5 + required_delta / 2)
        
        overcut_gain = self.cold_tire_penalty * 2  
        tire_age_diff = tire_age_self - tire_age_rival
        
        if tire_age_diff < 5 and gap_ahead < (pit_loss - overcut_gain):
            overcut_window = (current_lap + 2, current_lap + 5)
            overcut_viable = True
        else:
            overcut_window = (0, 0)
            overcut_viable = False
        
        gap_evolution = self._simulate_gap(
            gap_ahead, tire_age_self, tire_age_rival, 10
        )
        
        if probability > 0.7 and required_delta > 0:
            recommendation = "UNDERCUT RECOMMENDED"
        elif overcut_viable and tire_age_self < 15:
            recommendation = "OVERCUT VIABLE"
        else:
            recommendation = "HOLD POSITION (No Advantage)"
        
        reasoning = self._generate_reasoning(
            gap_ahead, required_delta, probability, 
            overcut_viable, traffic_behind
        )
        
        return UndercutResult(
            undercut_probability=round(probability, 2),
            required_outlap_delta=round(required_delta, 2),
            overcut_window=overcut_window,
            recommendation=recommendation,
            reasoning=reasoning,
            gap_evolution=gap_evolution
        )
    
    def _simulate_gap(
        self,
        initial_gap: float,
        tire_age_self: int,
        tire_age_rival: int,
        n_laps: int
    ) -> List[float]:
        gaps = [initial_gap]
        gap = initial_gap
        
        deg_diff = (tire_age_self - tire_age_rival) * 0.02
        
        for lap in range(n_laps):
            gap += deg_diff
            gaps.append(round(gap, 2))
        
        return gaps
    
    def _generate_reasoning(
        self,
        gap: float,
        required_delta: float,
        probability: float,
        overcut_viable: bool,
        traffic: bool
    ) -> str:
        parts = []
        
        parts.append(f"Current gap: {gap:.1f}s")
        
        if required_delta > 0:
            parts.append(f"Undercut needs {required_delta:.1f}s faster out-lap than rival's in-lap")
            parts.append(f"Success probability: {probability*100:.0f}%")
        else:
            parts.append(f"Undercut unlikely: would need {-required_delta:.1f}s impossible gain")
        
        if overcut_viable:
            parts.append("Overcut window available: stay out 2-5 more laps")
        
        if traffic:
            parts.append("[WARNING: Traffic behind will cost ~1.5s on rejoin]")
        
        return ". ".join(parts) + "."

class SafetyCarAdvisor:
    def __init__(self):
        self.avg_sc_duration = 4  
        self.gap_elimination_factor = 0.9  
    
    def advise(
        self,
        position: int,
        tire_age: int,
        tire_compound: str,
        gap_ahead: float,
        gap_behind: float,
        sc_lap: int,
        total_laps: int,
        pit_loss: float,
        already_pitted_recently: bool = False
    ) -> SCDecisionResult:

        remaining_laps = total_laps - sc_lap
        
        free_stop_savings = pit_loss * 0.75  
        fresh_tire_advantage = min(tire_age * 0.04, 2.0) * remaining_laps
        
        positions_lost = self._estimate_positions_lost(gap_behind, position)
        position_value = positions_lost * 3 
        
        stay_score = 0
        pit_score = 0
        
        if tire_age < 10:
            stay_score += 3
        elif tire_age > 25:
            pit_score += 3
        
        if position <= 3:
            stay_score += 2  
        elif position > 10:
            pit_score += 1  
        
        if tire_compound == 'HARD' and remaining_laps < 20:
            stay_score += 2  
        elif tire_compound == 'SOFT' and remaining_laps > 15:
            pit_score += 2  
        
        if already_pitted_recently:
            stay_score += 4
        
        if free_stop_savings > 15:
            pit_score += 3
        
        if remaining_laps < 10:
            stay_score += 1
        
        pit_score += fresh_tire_advantage / 5
        net_score = pit_score - stay_score
        
        if net_score > 2:
            recommendation = "PIT NOW"
            confidence = min(0.95, 0.6 + net_score * 0.1)
        elif net_score < -2:
            recommendation = "STAY OUT"
            confidence = min(0.95, 0.6 + abs(net_score) * 0.1)
        else:
            recommendation = "MARGINAL - PIT IF COMPOUND CHANGE HELPS"
            confidence = 0.5
        
        reasoning = self._generate_reasoning(
            position, tire_age, tire_compound, remaining_laps,
            free_stop_savings, positions_lost, pit_score, stay_score
        )
        
        alternative = self._generate_alternative(
            recommendation, remaining_laps, tire_age
        )
        
        return SCDecisionResult(
            recommendation=recommendation,
            confidence=round(confidence, 2),
            reasoning=reasoning,
            alternative_scenario=alternative,
            expected_positions={
                'after_pit': position + positions_lost,
                'if_stay': position
            }
        )
    
    def _estimate_positions_lost(self, gap_behind: float, position: int) -> int:
        if gap_behind > 3.0:
            return 0
        elif gap_behind > 1.5:
            return 1
        elif gap_behind > 0.8:
            return 2
        else:
            return min(3, 20 - position)
    
    def _generate_reasoning(
        self,
        position: int,
        tire_age: int,
        compound: str,
        remaining: int,
        free_savings: float,
        pos_lost: int,
        pit_score: float,
        stay_score: float
    ) -> str:
        parts = []
        
        parts.append(f"P{position}, {compound}s ({tire_age} laps old), {remaining} laps remaining")
        parts.append(f"Free stop saves ~{free_savings:.1f}s vs green flag pit")
        
        if pos_lost > 0:
            parts.append(f"Would likely lose {pos_lost} position(s)")
        else:
            parts.append("Clear gap behind - no positions at risk")
        
        if tire_age > 20:
            parts.append(f"Tire age ({tire_age}) suggests fresh tires beneficial")
        elif tire_age < 10:
            parts.append(f"Recently pitted ({tire_age} laps) - stay out")
        
        parts.append(f"[Analysis: PIT={pit_score:.1f}, STAY={stay_score:.1f}]")
        
        return ". ".join(parts) + "."
    
    def _generate_alternative(
        self,
        rec: str,
        remaining: int,
        tire_age: int
    ) -> str:
        if "PIT" in rec:
            return f"If SC ends quickly (<2 laps), STAY may preserve position better"
        else:
            return f"If SC extends (>4 laps) or tire age exceeds {tire_age + 5}, reconsider PIT"

class DRSTrainEscape:
    
    def __init__(self):
        self.train_time_loss = 0.6  
    
    def analyze(
        self,
        current_lap: int,
        total_laps: int,
        tire_age: int,
        pit_loss: float,
        train_size: int,
        gap_to_leaders: float,
        track_meta: pd.Series
    ) -> DRSEscapeResult:
        remaining = total_laps - current_lap
        overtaking_index = track_meta.get('relative_overtaking_index', 0.5)
        
        strategies = []
        
        pit_early = self._evaluate_early_pit(
            current_lap, remaining, tire_age, pit_loss, train_size
        )
        strategies.append(pit_early)
        
        overcut = self._evaluate_overcut(
            current_lap, remaining, tire_age, pit_loss, train_size, gap_to_leaders
        )
        strategies.append(overcut)
        
        if overtaking_index < 0.5:
            overtake = self._evaluate_overtake(
                current_lap, remaining, train_size, gap_to_leaders, overtaking_index
            )
            strategies.append(overtake)
        
        strategies.sort(key=lambda x: x.get('score', 0), reverse=True)
        best = strategies[0]
        
        time_saved = (remaining * self.train_time_loss) - best.get('cost', 0)
        
        reasoning = self._generate_reasoning(
            best, train_size, remaining, overtaking_index
        )
        
        return DRSEscapeResult(
            strategies=strategies,
            recommended=best['name'],
            time_saved=round(max(0, time_saved), 1),
            reasoning=reasoning
        )
    
    def _evaluate_early_pit(
        self,
        current_lap: int,
        remaining: int,
        tire_age: int,
        pit_loss: float,
        train_size: int
    ) -> Dict[str, Any]:
        time_lost_in_train = remaining * self.train_time_loss
        clean_air_remaining = remaining - 2  
        
        if tire_age < 10:
            penalty = 3  
        else:
            penalty = 0
        
        net_gain = time_lost_in_train - pit_loss - penalty
        
        return {
            'name': 'PIT EARLY',
            'description': f'Pit now (L{current_lap}), exit in clean air',
            'score': net_gain,
            'cost': pit_loss,
            'risk': 'Low' if tire_age > 15 else 'Medium'
        }
    
    def _evaluate_overcut(
        self,
        current_lap: int,
        remaining: int,
        tire_age: int,
        pit_loss: float,
        train_size: int,
        gap_to_leaders: float
    ) -> Dict[str, Any]:
        laps_until_others_pit = max(5, 20 - tire_age)
        
        if laps_until_others_pit < remaining:
            positions_gained = min(train_size - 1, 3)
            score = positions_gained * 2 - (laps_until_others_pit * 0.3)
        else:
            score = -5  
        
        return {
            'name': 'OVERCUT - STAY OUT',
            'description': f'Stay out {laps_until_others_pit} laps, pit after train',
            'score': score,
            'cost': 0,
            'risk': 'High' if tire_age > 25 else 'Low'
        }
    
    def _evaluate_overtake(
        self,
        current_lap: int,
        remaining: int,
        train_size: int,
        gap_to_leaders: float,
        overtaking_index: float
    ) -> Dict[str, Any]:
        laps_to_overtake = (1 / (1 - overtaking_index)) * train_size
        
        if laps_to_overtake < remaining:
            score = 3 - overtaking_index * 5
        else:
            score = -3
        
        return {
            'name': 'ATTACK - OVERTAKE',
            'description': f'Push to overtake ({laps_to_overtake:.0f} laps estimated)',
            'score': score,
            'cost': 0, 
            'risk': 'Medium'
        }
    
    def _generate_reasoning(
        self,
        best: Dict,
        train_size: int,
        remaining: int,
        overtaking_index: float
    ) -> str:
        parts = []
        
        parts.append(f"Stuck in {train_size}-car DRS train")
        parts.append(f"Losing ~{self.train_time_loss}s/lap vs clean air")
        parts.append(f"{remaining} laps remaining")
        
        if overtaking_index > 0.6:
            parts.append("[Difficult overtaking track - pit strategy preferred]")
        
        parts.append(f"Recommended: {best['name']}")
        parts.append(f"Risk level: {best.get('risk', 'Unknown')}")
        
        return ". ".join(parts) + "."

class StrategyConfidenceEnvelope:

    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        np.random.seed(42)
    
    def calculate_envelope(
        self,
        base_strategy: StrategyResult,
        track_meta: pd.Series,
        total_laps: int
    ) -> Dict[str, Any]:
        sc_risk = track_meta.get('historical_sc_risk_index', 0.35)
        
        race_times = []
        scenarios = []
        
        for i in range(self.n_simulations):
            
            deg_perturbation = np.random.normal(0, 0.02)  
            pit_perturbation = np.random.normal(0, 1.0)   
            
            sc_occurs = np.random.random() < sc_risk
            sc_lap = np.random.randint(10, total_laps - 10) if sc_occurs else None
            
            traffic_laps = np.random.randint(0, 15)
            traffic_cost = traffic_laps * 0.3
            
            base_time = base_strategy.expected_time_s
            adjusted = base_time
            
            adjusted += deg_perturbation * total_laps
            
            adjusted += pit_perturbation * base_strategy.stop_count
            
            if sc_occurs:
                adjusted -= 15  
            
            adjusted += traffic_cost
            
            race_times.append(adjusted)
            scenarios.append({
                'time': adjusted,
                'sc': sc_occurs,
                'traffic_laps': traffic_laps
            })
        
        race_times = np.array(race_times)
        
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = {
            f'p{p}': round(np.percentile(race_times, p), 1)
            for p in percentiles
        }
        
        envelope_width = percentile_values['p95'] - percentile_values['p5']
        if envelope_width < 10:
            robustness = 'High'
        elif envelope_width < 20:
            robustness = 'Medium'
        else:
            robustness = 'Low'
        
        return {
            'mean': round(np.mean(race_times), 1),
            'std': round(np.std(race_times), 1),
            'percentiles': percentile_values,
            'best_case': percentile_values['p5'],
            'worst_case': percentile_values['p95'],
            'envelope_width': round(envelope_width, 1),
            'robustness': robustness,
            'sc_impact': round(np.mean([s['time'] for s in scenarios if s['sc']]) - 
                              np.mean([s['time'] for s in scenarios if not s['sc']]), 1),
            'histogram_data': np.histogram(race_times, bins=20)
        }

class CaseStudyEngine:

    FAMOUS_RACES = {
        '2021_Abu_Dhabi': {
            'year': 2021,
            'gp': 'Abu Dhabi',
            'session': 'R',
            'key_moment': 'SC lap 54 (Latifi crash)',
            'controversy': 'SC restart procedure debate',
            'key_decisions': [
                ('LAP 14', 'HAM pits for Hard', 'Standard 1-stop'),
                ('LAP 37', 'VER pits for Hard', 'Covering Hamilton'),
                ('LAP 54', 'SC deployed', 'Latifi crash'),
                ('LAP 54', 'VER pits for Soft', 'Free stop under SC'),
                ('LAP 58', 'Race ends', 'VER wins')
            ]
        },
        '2019_Singapore': {
            'year': 2019,
            'gp': 'Singapore',
            'session': 'R',
            'key_moment': 'Ferrari Undercut (Vettel jumps Leclerc)',
            'controversy': 'Vettel jumps Leclerc after undercut',
            'key_decisions': [
                ('LAP 20', 'VET pits (undercut attempt)', 'Pit from P3'),
                ('LAP 21', 'LEC pits (response)', 'Pit from P1'),
                ('LAP 22', 'VET exits ahead of LEC', 'Undercut worked')
            ]
        },
        '2020_Sakhir': {
            'year': 2020,
            'gp': 'Sakhir',
            'session': 'R',
            'key_moment': 'Mercedes pit chaos',
            'controversy': 'Wrong tires fitted + missed stop',
            'key_decisions': [
                ('LAP 62', 'RUS pit for fresh tires', 'Leading race'),
                ('LAP 62', 'Mixed tires fitted', 'Wrong compound'),
                ('LAP 63', 'RUS pits AGAIN', 'To fix tires'),
                ('LAP 87', 'RUS puncture', 'Drops from P2')
            ]
        }
    }
    
    def __init__(self):
        from data_engine import DataEngine
        self.data_engine = DataEngine()
        self.strategy_gen = NeutralStrategyGenerator()
    
    def analyze_case(
        self,
        case_name: str,
        track_meta: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
   
        if case_name not in self.FAMOUS_RACES:
            available = list(self.FAMOUS_RACES.keys())
            raise ValueError(f"Case '{case_name}' not found. Available: {available}")
        
        case = self.FAMOUS_RACES[case_name]
        
        try:
            session = self.data_engine.load_session(
                case['year'], case['gp'], case['session']
            )
            clean_laps = self.data_engine.clean_laps(session.laps)
            featured = self.data_engine.engineer_features(clean_laps, session)
        except Exception as e:
            return {
                'case': case_name,
                'error': f"Could not load data: {str(e)}",
                'fallback': self._get_case_summary(case)
            }
        
        analysis = {
            'case': case_name,
            'race_info': {
                'year': case['year'],
                'gp': case['gp'],
                'key_moment': case['key_moment']
            },
            'key_decisions': case['key_decisions'],
            'model_analysis': {},
            'lessons': []
        }
        
        if track_meta is not None:
            total_laps = int(clean_laps['LapNumber'].max()) if len(clean_laps) > 0 else 57
            strategy = self.strategy_gen.generate(
                track_meta, 
                total_laps,
                sc_risk=0.5 
            )
            analysis['model_recommendation'] = strategy.to_dict()
        
        analysis['lessons'] = self._extract_lessons(case_name)
        
        analysis['disclaimer'] = (
            "This is retrospective analysis. WTF1 cannot predict future events. "
            "This case study shows how the model would have evaluated the situation at each decision point."
        )
        
        return analysis
    
    def _get_case_summary(self, case: Dict) -> Dict:
        return {
            'year': case['year'],
            'gp': case['gp'],
            'key_moment': case['key_moment'],
            'decisions': case['key_decisions'],
            'note': 'Full analysis requires FastF1 data access'
        }
    
    def _extract_lessons(self, case_name: str) -> List[str]:
        lessons = {
            '2021_Abu_Dhabi': [
                "SC can completely override tire strategy",
                "Track position vs tire freshness trade-off is context-dependent",
                "Late-race SC favors flexible strategies",

            ],
            '2019_Singapore': [
                "Undercut power depends on out-lap advantage",
                "Street circuits amplify undercut effectiveness",
                "Team order conflicts arise from pure pace optimization",
                "Traffic-free out-lap is worth 2-3s"
            ],
            '2020_Sakhir': [
                "Pit stop execution is as important as strategy",
                "Human factors cannot be modeled",
                "Leading race increases conservative tendency",
                "Model limitation: Cannot model pit crew errors"
            ]
        }
        return lessons.get(case_name, ["No specific lessons extracted"])

class StrategyTools:

    def __init__(self, n_simulations: int = 1000):
        self.strategy_gen = NeutralStrategyGenerator(n_simulations)
        self.undercut = UndercutAnalyzer()
        self.sc_advisor = SafetyCarAdvisor()
        self.drs_escape = DRSTrainEscape()
        self.envelope = StrategyConfidenceEnvelope(n_simulations)
        self.case_study = CaseStudyEngine()
    
    def generate_neutral_strategy(
        self,
        track_info: pd.Series,
        total_laps: int,
        tire_analysis: Optional[Dict] = None
    ) -> StrategyResult:
        sc_risk = track_info.get('historical_sc_risk_index', 0.35)
        return self.strategy_gen.generate(
            track_info, total_laps, tire_analysis, sc_risk
        )
    
    def analyze_undercut(
        self,
        gap: float,
        tire_age_self: int,
        tire_age_rival: int,
        pit_loss: float,
        current_lap: int,
        total_laps: int
    ) -> UndercutResult:
        return self.undercut.analyze(
            gap, tire_age_self, tire_age_rival,
            pit_loss, current_lap, total_laps
        )
    
    def get_sc_advice(
        self,
        position: int,
        tire_age: int,
        compound: str,
        gap_ahead: float,
        gap_behind: float,
        sc_lap: int,
        total_laps: int,
        pit_loss: float
    ) -> SCDecisionResult:
        return self.sc_advisor.advise(
            position, tire_age, compound,
            gap_ahead, gap_behind, sc_lap,
            total_laps, pit_loss
        )
    
    def analyze_drs_escape(
        self,
        current_lap: int,
        total_laps: int,
        tire_age: int,
        pit_loss: float,
        train_size: int,
        gap_to_leaders: float,
        track_meta: pd.Series
    ) -> DRSEscapeResult:
        return self.drs_escape.analyze(
            current_lap, total_laps, tire_age,
            pit_loss, train_size, gap_to_leaders, track_meta
        )
    
    def get_confidence_envelope(
        self,
        strategy: StrategyResult,
        track_meta: pd.Series,
        total_laps: int
    ) -> Dict[str, Any]:
        return self.envelope.calculate_envelope(strategy, track_meta, total_laps)
    
    def run_case_study(self, case_name: str) -> Dict[str, Any]:
        return self.case_study.analyze_case(case_name)

@dataclass
class CatchUpResult:
    catch_lap: Optional[int]
    laps_to_catch: Optional[int]
    difficulty_index: float  
    gap_trajectory: List[float]
    projected_pass_lap: Optional[int]
    reasoning: str

class CatchUpEngine:
    def __init__(self):
        self.overtake_threshold = 1.0  
        
    def analyze(
        self,
        current_gap: float,
        laps_remaining: int,
        hunter_compound: str,
        hunter_tire_age: int,
        target_compound: str,
        target_tire_age: int,
        track_meta: Optional[pd.Series] = None
    ) -> CatchUpResult:
        pace_offsets = {'SOFT': -0.8, 'MEDIUM': 0.0, 'HARD': 0.4}
        deg_rates = {'SOFT': 0.08, 'MEDIUM': 0.05, 'HARD': 0.03}
        
        if track_meta is not None:
             overtake_diff = 1.0 - track_meta.get('relative_overtaking_index', 0.5)
        else:
             overtake_diff = 0.5
        gap = current_gap
        trajectory = [gap]
        catch_lap = None
        pass_lap = None
        
        hunter_age = hunter_tire_age
        target_age = target_tire_age
        
        for lap in range(1, laps_remaining + 1):
            hunter_pace = pace_offsets.get(hunter_compound, 0) + (hunter_age * deg_rates.get(hunter_compound, 0.05))
            target_pace = pace_offsets.get(target_compound, 0) + (target_age * deg_rates.get(target_compound, 0.05))
            
            gain = target_pace - hunter_pace
            
            if gap < 1.0:
                gain += 0.4 
            
            gap -= gain
            gap = max(0, gap) 
            trajectory.append(round(gap, 2))
            
            hunter_age += 1
            target_age += 1
            
            if catch_lap is None and gap < 0.8:
                catch_lap = lap
            
            if pass_lap is None and gap < 0.1:
                pass_lap = lap
                break
        
        if catch_lap:
            difficulty = overtake_diff 
            if catch_lap > laps_remaining * 0.8:
                difficulty += 0.2 
            if gap > 0.3:
                difficulty += 0.1 
        else:
            difficulty = 1.0 
            
        difficulty = min(1.0, difficulty)
        
        reasoning = self._generate_reasoning(
            current_gap, catch_lap, hunter_compound, target_compound, difficulty
        )
        
        return CatchUpResult(
            catch_lap=catch_lap,
            laps_to_catch=catch_lap,
            difficulty_index=round(difficulty, 2),
            gap_trajectory=trajectory,
            projected_pass_lap=pass_lap,
            reasoning=reasoning
        )
        
    def _generate_reasoning(
        self,
        gap: float,
        catch_lap: Optional[int],
        h_comp: str,
        t_comp: str,
        difficulty: float

    ) -> str:
        parts = []
        if catch_lap:
            parts.append(f"Projected catch in {catch_lap} laps")
            parts.append(f"Hunter on {h_comp} vs Target {t_comp}")
            
            if difficulty > 0.7:
                 parts.append("Overtake Difficulty: High (Critical tire delta)")
            elif difficulty < 0.4:
                 parts.append("Overtake Difficulty: Low (Easy pass)")
            else:
                 parts.append("Overtake Difficulty: Medium")
        else:
            parts.append("Catch not projected within remaining laps")
            
        return ". ".join(parts)

if __name__ == "__main__":
    print("WTF1 Strategy Tools - Testing")
    print("=" * 60)
    
    import pandas as pd
    
    track_meta = pd.Series({
        'track_name': 'Bahrain International Circuit',
        'country': 'Bahrain',
        'length_km': 5.412,
        'estimated_green_flag_pit_loss_s': 23.6,
        'drs_zones_typical': 3,
        'relative_overtaking_index': 0.35,
        'aero_load_category': 'Medium',
        'expected_deg_index': 'High',
        'historical_sc_risk_index': 0.43
    })
    
    tools = StrategyTools(n_simulations=500)  
    
    print("\n1. Testing Neutral Strategy Generator...")
    strategy = tools.generate_neutral_strategy(track_meta, 57)
    print(f"   Optimal: {strategy.stop_count}-stop ({strategy.compound_sequence})")
    print(f"   Expected: {strategy.expected_time_s}s ±{strategy.uncertainty_s}s")
    print(f"   Confidence: {strategy.confidence}")
    
    print("\n2. Testing Undercut Analyzer...")
    undercut = tools.analyze_undercut(
        gap=2.5, tire_age_self=15, tire_age_rival=12,
        pit_loss=23.6, current_lap=25, total_laps=57
    )
    print(f"   {undercut.recommendation}")
    print(f"   Probability: {undercut.undercut_probability}")
    
    print("\n3. Testing SC Advisor...")
    sc_advice = tools.get_sc_advice(
        position=5, tire_age=18, compound='MEDIUM',
        gap_ahead=3.2, gap_behind=1.8, sc_lap=34,
        total_laps=57, pit_loss=23.6
    )
    print(f"   {sc_advice.recommendation} (confidence: {sc_advice.confidence})")
    
    print("\n4. Testing Confidence Envelope...")
    envelope = tools.get_confidence_envelope(strategy, track_meta, 57)
    print(f"   Robustness: {envelope['robustness']}")
    print(f"   Envelope: {envelope['best_case']}s to {envelope['worst_case']}s")
    
    print("\n5. Testing Case Study Engine...")
    case = tools.run_case_study('2021_Abu_Dhabi')
    print(f"   Case: {case['case']}")
    print(f"   Key moment: {case['race_info']['key_moment']}")
    
    print("\n" + "=" * 60)
    print("All strategy tool tests complete!")