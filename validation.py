# WTF1 Validation Module
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import warnings
from datetime import datetime

import pandas as pd
import numpy as np

@dataclass
class SanityCheckResult:
    name: str
    passed: bool
    expected: str
    actual: str
    severity: str
    explanation: str

@dataclass
class ValidationResult:
    metric: str
    value: float
    benchmark: float
    passed: bool
    context: str

class SanityChecker:
    
    def __init__(self):
        self.checks_run = []
        self.checks_passed = 0
        self.checks_failed = 0
    
    def run_all_checks(
        self,
        track_meta: pd.DataFrame,
        tire_results: Optional[Dict] = None,
        pace_results: Optional[pd.DataFrame] = None,
        strategy_result: Optional[Any] = None
    ) -> List[SanityCheckResult]:

        results = []
        
        results.extend(self._check_track_metadata(track_meta))
        
        if tire_results:
            results.extend(self._check_tire_analysis(tire_results))
        
        if pace_results is not None and len(pace_results) > 0:
            results.extend(self._check_pace_analysis(pace_results))
        
        if strategy_result:
            results.extend(self._check_strategy(strategy_result))
        
        self.checks_run = results
        self.checks_passed = sum(1 for r in results if r.passed)
        self.checks_failed = len(results) - self.checks_passed
        
        return results
    
    def _check_track_metadata(self, df: pd.DataFrame) -> List[SanityCheckResult]:
        results = []
        
        monaco = df[df['track_name'].str.contains('Monaco', case=False)]
        if len(monaco) > 0:
            monaco_oi = monaco.iloc[0]['relative_overtaking_index']
            results.append(SanityCheckResult(
                name="Monaco high overtaking index",
                passed=monaco_oi > 0.8,
                expected="> 0.8",
                actual=str(monaco_oi),
                severity='critical',
                explanation="Monaco is notoriously difficult for overtaking. "
                           "Index < 0.8 suggests metadata error."
            ))
        
        monza = df[df['track_name'].str.contains('Monza', case=False)]
        if len(monza) > 0:
            monza_aero = monza.iloc[0]['aero_load_category']
            results.append(SanityCheckResult(
                name="Monza low aero category",
                passed=monza_aero == 'Low',
                expected="Low",
                actual=str(monza_aero),
                severity='warning',
                explanation="Monza is a low-downforce circuit. "
                           "Non-Low category suggests metadata error."
            ))
        
        pit_losses = df['estimated_green_flag_pit_loss_s'].values
        pit_check_passed = all(15 < p < 35 for p in pit_losses)
        results.append(SanityCheckResult(
            name="Pit loss in realistic range",
            passed=pit_check_passed,
            expected="15s < pit_loss < 35s for all tracks",
            actual=f"min={min(pit_losses):.1f}s, max={max(pit_losses):.1f}s",
            severity='critical',
            explanation="Pit lane time loss should be between 18-35 seconds for F1."
        ))
        
        lengths = df['length_km'].values
        length_check = all(3 < l < 8 for l in lengths)
        results.append(SanityCheckResult(
            name="Track lengths realistic",
            passed=length_check,
            expected="3km < length < 8km",
            actual=f"min={min(lengths):.2f}km, max={max(lengths):.2f}km",
            severity='warning',
            explanation="F1 tracks are typically 3-8km in length."
        ))
        
        sc_risks = df['historical_sc_risk_index'].values
        sc_check = all(0 <= r <= 1 for r in sc_risks)
        results.append(SanityCheckResult(
            name="SC risk index normalized",
            passed=sc_check,
            expected="0 <= sc_risk <= 1",
            actual=f"min={min(sc_risks):.2f}, max={max(sc_risks):.2f}",
            severity='critical',
            explanation="SC risk is a probability and must be 0-1."
        ))
        
        return results
    
    def _check_tire_analysis(self, tire_results: Dict) -> List[SanityCheckResult]:
        results = []
        
        for compound, result in tire_results.items():
            results.append(SanityCheckResult(
                name=f"{compound} positive degradation",
                passed=result.deg_rate_s_per_lap >= 0,
                expected=">= 0",
                actual=f"{result.deg_rate_s_per_lap:.4f} s/lap",
                severity='critical',
                explanation="Tires cannot get faster with age. "
                           "Negative degradation indicates data/model error."
            ))
            
            results.append(SanityCheckResult(
                name=f"{compound} realistic degradation",
                passed=result.deg_rate_s_per_lap < 0.25,
                expected="< 0.25 s/lap",
                actual=f"{result.deg_rate_s_per_lap:.4f} s/lap",
                severity='warning',
                explanation="Degradation > 0.25s/lap is extremely high. "
                           "Check for outliers or data quality issues."
            ))
            
            results.append(SanityCheckResult(
                name=f"{compound} model fit quality",
                passed=result.r_squared > 0.3,
                expected="R² > 0.3",
                actual=f"R² = {result.r_squared:.3f}",
                severity='info',
                explanation="Low R² suggests non-linear degradation or noisy data. "
                           "Consider piecewise models."
            ))
            
            if 'SOFT' in tire_results and 'HARD' in tire_results:
                soft_deg = tire_results['SOFT'].deg_rate_s_per_lap
                hard_deg = tire_results['HARD'].deg_rate_s_per_lap
                results.append(SanityCheckResult(
                    name="Soft > Hard degradation",
                    passed=soft_deg > hard_deg,
                    expected="Soft deg > Hard deg",
                    actual=f"Soft={soft_deg:.3f}, Hard={hard_deg:.3f}",
                    severity='warning',
                    explanation="Softer compounds usually degrade faster, but graining/track evolution can invert this."
                ))
                break 
        
        return results
    
    def _check_pace_analysis(self, pace_df: pd.DataFrame) -> List[SanityCheckResult]:
        results = []
        
        if 'CorrectedPace' in pace_df.columns:
            times = pace_df['CorrectedPace'].values
            time_check = all(60 < t < 150 for t in times)
            results.append(SanityCheckResult(
                name="Lap times in realistic range",
                passed=time_check,
                expected="60s < lap_time < 150s",
                actual=f"min={min(times):.1f}s, max={max(times):.1f}s",
                severity='critical',
                explanation="F1 lap times are typically 60-150 seconds."
            ))
        
        if 'CorrectedPace' in pace_df.columns and len(pace_df) > 1:
            spread = pace_df['CorrectedPace'].max() - pace_df['CorrectedPace'].min()
            results.append(SanityCheckResult(
                name="Pace spread realistic",
                passed=spread < 10,
                expected="< 10s spread",
                actual=f"{spread:.2f}s",
                severity='warning',
                explanation="Excessive pace spread suggests outliers or data issues."
            ))
        
        return results
    
    def _check_strategy(self, strategy) -> List[SanityCheckResult]:
        results = []
        
        if hasattr(strategy, 'to_dict'):
            s = strategy.to_dict()
        else:
            s = strategy
        
        stop_count = s.get('stop_count', 0)
        results.append(SanityCheckResult(
            name="Stop count realistic",
            passed=1 <= stop_count <= 3,
            expected="1-3 stops",
            actual=str(stop_count),
            severity='critical',
            explanation="Modern F1 races typically require 1-3 pit stops."
        ))
        
        uncertainty = s.get('uncertainty', 0)
        results.append(SanityCheckResult(
            name="Positive uncertainty",
            passed=uncertainty > 0,
            expected="> 0",
            actual=f"{uncertainty}s",
            severity='warning',
            explanation="Zero uncertainty is unrealistic. "
                       "All predictions have inherent uncertainty."
        ))
        
        confidence = s.get('confidence', 0)
        results.append(SanityCheckResult(
            name="Confidence normalized",
            passed=0 <= confidence <= 1,
            expected="0 <= confidence <= 1",
            actual=str(confidence),
            severity='critical',
            explanation="Confidence is a probability and must be 0-1."
        ))
        
        return results
    
    def print_report(self) -> str:
        report = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        WTF1 SANITY CHECK REPORT                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        report += f"\nTotal checks: {len(self.checks_run)}\n"
        report += f"Passed: {self.checks_passed} | Failed: {self.checks_failed}\n"
        report += "\n" + "─" * 78 + "\n"
        
        for severity in ['critical', 'warning', 'info']:
            checks = [c for c in self.checks_run if c.severity == severity]
            if not checks:
                continue
            
            report += f"\n{severity.upper()} ({len(checks)} checks):\n"
            for check in checks:
                status = "✓" if check.passed else "✗"
                report += f"  {status} {check.name}\n"
                if not check.passed:
                    report += f"      Expected: {check.expected}\n"
                    report += f"      Actual:   {check.actual}\n"
                    report += f"      → {check.explanation}\n"
        
        report += "\n" + "═" * 78 + "\n"
        
        if self.checks_failed == 0:
            report += "All sanity checks passed! \n"
        else:
            report += f"⚠ {self.checks_failed} checks failed. Review and investigate.\n"
        
        return report


class Backtester:
    def __init__(self):
        self.train_years = list(range(2018, 2024))
        self.test_years = [2024]
        self.results = []
    
    def run_backtest(
        self,
        races: List[Dict[str, Any]],
        strategy_gen,
        track_meta: pd.DataFrame
    ) -> Dict[str, Any]:
    
        predictions = []
        actuals = []
        
        for race in races:
            if race['year'] not in self.test_years:
                continue
            
            track_row = track_meta[
                track_meta['track_name'].str.contains(race['gp'], case=False)
            ]
            if len(track_row) == 0:
                continue
            
            try:
                predicted_strategy = strategy_gen.generate(
                    track_row.iloc[0],
                    race.get('total_laps', 57),
                    sc_risk=track_row.iloc[0].get('historical_sc_risk_index', 0.35)
                )
                
                predictions.append({
                    'race': race['gp'],
                    'predicted_stops': predicted_strategy.stop_count,
                    'predicted_compounds': predicted_strategy.compound_sequence,
                    'confidence': predicted_strategy.confidence
                })
                
                actuals.append({
                    'race': race['gp'],
                    'actual_stops': race.get('actual_winner_strategy', {}).get('stops', None),
                    'actual_compounds': race.get('actual_winner_strategy', {}).get('compounds', None)
                })
            except Exception as e:
                warnings.warn(f"Failed to predict {race['gp']}: {str(e)}")
                continue
        
        correct_stops = 0
        total = len(predictions)
        
        for pred, actual in zip(predictions, actuals):
            if actual['actual_stops'] == pred['predicted_stops']:
                correct_stops += 1
        
        stop_accuracy = correct_stops / total if total > 0 else 0
        
        return {
            'total_races': total,
            'stop_count_accuracy': round(stop_accuracy, 2),
            'predictions': predictions,
            'actuals': actuals,
            'interpretation': self._interpret_accuracy(stop_accuracy)
        }
    
    def _interpret_accuracy(self, accuracy: float) -> str:
        if accuracy > 0.8:
            return "Excellent: Model matches actual winner strategies >80% of time"
        elif accuracy > 0.6:
            return "Good: Model is reasonably aligned with race outcomes"
        elif accuracy > 0.4:
            return "Fair: Model captures general trends but misses nuances"
        else:
            return "Poor: Model needs significant improvement or different approach"

class SensitivityAnalyzer:
    
    def __init__(self):
        pass
    
    def analyze_deg_sensitivity(
        self,
        base_strategy,
        strategy_gen,
        track_meta: pd.Series,
        total_laps: int,
        perturbations: List[float] = [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]
    ) -> Dict[str, Any]:

        results = []
        
        for pert in perturbations:
            perturbed_tire = {
                'SOFT': type('Result', (), {'deg_rate_s_per_lap': 0.08 + pert})(),
                'MEDIUM': type('Result', (), {'deg_rate_s_per_lap': 0.05 + pert})(),
                'HARD': type('Result', (), {'deg_rate_s_per_lap': 0.03 + pert})()
            }
            
            try:
                strategy = strategy_gen.generate(
                    track_meta, total_laps, perturbed_tire
                )
                
                results.append({
                    'perturbation': pert,
                    'stop_count': strategy.stop_count,
                    'compounds': strategy.compound_sequence,
                    'time_delta': strategy.expected_time_s - base_strategy.expected_time_s
                })
            except:
                continue
        
        stop_changes = len(set(r['stop_count'] for r in results)) > 1
        
        return {
            'parameter': 'degradation_rate',
            'results': results,
            'strategy_changes': stop_changes,
            'interpretation': (
                "Strategy is SENSITIVE to degradation assumptions"
                if stop_changes else
                "Strategy is ROBUST to degradation changes"
            )
        }
    
    def analyze_sc_sensitivity(
        self,
        base_strategy,
        strategy_gen,
        track_meta: pd.Series,
        total_laps: int,
        sc_risks: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8]
    ) -> Dict[str, Any]:
        results = []
        
        track_copy = track_meta.copy()
        
        for sc_risk in sc_risks:
            track_copy['historical_sc_risk_index'] = sc_risk
            
            try:
                strategy = strategy_gen.generate(
                    track_copy, total_laps
                )
                
                results.append({
                    'sc_risk': sc_risk,
                    'stop_count': strategy.stop_count,
                    'time_delta': strategy.expected_time_s - base_strategy.expected_time_s,
                    'envelope_width': strategy.worst_case_s - strategy.best_case_s
                })
            except:
                continue
        
        return {
            'parameter': 'sc_risk',
            'results': results,
            'interpretation': (
                "Higher SC risk widens confidence envelope but typically "
                "doesn't change optimal stop count"
            )
        }

class ModelValidator:

    def __init__(self):
        self.validation_results = []
    
    def validate_race(
        self,
        race_data: Dict[str, Any],
        predicted_strategy,
        actual_winner_strategy: Dict[str, Any]
    ) -> ValidationResult:

        if hasattr(predicted_strategy, 'to_dict'):
            pred = predicted_strategy.to_dict()
        else:
            pred = predicted_strategy
        
        pred_stops = pred.get('stop_count', 0)
        actual_stops = actual_winner_strategy.get('stops', 0)
        
        stops_match = pred_stops == actual_stops
        
        result = ValidationResult(
            metric='stop_count_match',
            value=1.0 if stops_match else 0.0,
            benchmark=1.0,
            passed=stops_match,
            context=(
                f"{race_data.get('gp', 'Unknown')} {race_data.get('year', '')}: "
                f"Predicted {pred_stops}-stop, Actual {actual_stops}-stop"
            )
        )
        
        self.validation_results.append(result)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.validation_results:
            return {'error': 'No validation results available'}
        
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.passed)
        
        return {
            'total_races': total,
            'correct_predictions': passed,
            'accuracy': round(passed / total, 2) if total > 0 else 0,
            'failures': [r.context for r in self.validation_results if not r.passed]
        }

class WTF1Validator:
    
    def __init__(self):
        self.sanity = SanityChecker()
        self.backtest = Backtester()
        self.sensitivity = SensitivityAnalyzer()
        self.model = ModelValidator()
    
    def run_full_validation(
        self,
        track_meta: pd.DataFrame,
        tire_results: Optional[Dict] = None,
        pace_results: Optional[pd.DataFrame] = None,
        strategy_result = None
    ) -> Dict[str, Any]:
    
        sanity_results = self.sanity.run_all_checks(
            track_meta, tire_results, pace_results, strategy_result
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'sanity_checks': {
                'total': len(sanity_results),
                'passed': self.sanity.checks_passed,
                'failed': self.sanity.checks_failed,
                'details': [
                    {
                        'name': r.name,
                        'passed': r.passed,
                        'severity': r.severity
                    }
                    for r in sanity_results
                ]
            },
            'report': self.sanity.print_report()
        }
    
    def run_sensitivity_analysis(
        self,
        base_strategy,
        strategy_gen,
        track_meta: pd.Series,
        total_laps: int
    ) -> Dict[str, Any]:
        deg_sensitivity = self.sensitivity.analyze_deg_sensitivity(
            base_strategy, strategy_gen, track_meta, total_laps
        )
        
        sc_sensitivity = self.sensitivity.analyze_sc_sensitivity(
            base_strategy, strategy_gen, track_meta, total_laps
        )
        
        return {
            'degradation': deg_sensitivity,
            'safety_car': sc_sensitivity
        }

if __name__ == "__main__":
    print("WTF1 Validation Module - Testing")
    print("=" * 60)
    
    track_meta = pd.DataFrame([
        {
            'track_name': 'Bahrain International Circuit',
            'length_km': 5.412,
            'estimated_green_flag_pit_loss_s': 23.6,
            'relative_overtaking_index': 0.35,
            'aero_load_category': 'Medium',
            'historical_sc_risk_index': 0.43
        },
        {
            'track_name': 'Circuit de Monaco',
            'length_km': 3.337,
            'estimated_green_flag_pit_loss_s': 22.8,
            'relative_overtaking_index': 0.92,
            'aero_load_category': 'Very High',
            'historical_sc_risk_index': 0.82
        },
        {
            'track_name': 'Autodromo Nazionale Monza',
            'length_km': 5.793,
            'estimated_green_flag_pit_loss_s': 26.5,
            'relative_overtaking_index': 0.18,
            'aero_load_category': 'Low',
            'historical_sc_risk_index': 0.28
        }
    ])
    
    tire_results = {
        'SOFT': type('Result', (), {
            'deg_rate_s_per_lap': 0.082,
            'r_squared': 0.72,
            'cliff_lap': 18
        })(),
        'MEDIUM': type('Result', (), {
            'deg_rate_s_per_lap': 0.051,
            'r_squared': 0.68,
            'cliff_lap': 28
        })(),
        'HARD': type('Result', (), {
            'deg_rate_s_per_lap': 0.031,
            'r_squared': 0.65,
            'cliff_lap': 42
        })()
    }
    
    validator = WTF1Validator()
    results = validator.run_full_validation(track_meta, tire_results)
    print(results['report'])
    print("\n" + "=" * 60)
    print("Validation module tests complete!")