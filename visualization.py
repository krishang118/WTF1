# WTF1 Visualization Module
from typing import Dict, Any, List, Optional, Tuple
import warnings
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not installed. Run: pip install matplotlib seaborn")

COMPOUND_COLORS = {
    'SOFT': '#FF0000',       
    'MEDIUM': '#FFD700',     
    'HARD': '#FFFFFF',       
    'INTERMEDIATE': '#00FF00',  
    'WET': '#0000FF'         
}
STRATEGY_COLORS = {
    '1-stop': '#4CAF50',    
    '2-stop': '#2196F3',    
    '3-stop': '#FF9800'     
}
WTF1_PALETTE = {
    'primary': '#E10600',    
    'secondary': '#1E1E1E',  
    'accent': '#00D2BE',     
    'background': '#000000', 
    'text': '#FFFFFF',
    'grid': '#333333'
}

def apply_wtf1_style():
    if not MATPLOTLIB_AVAILABLE:
        return
    
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': WTF1_PALETTE['background'],
        'axes.facecolor': WTF1_PALETTE['background'],
        'axes.edgecolor': WTF1_PALETTE['grid'],
        'axes.labelcolor': WTF1_PALETTE['text'],
        'text.color': WTF1_PALETTE['text'],
        'xtick.color': WTF1_PALETTE['text'],
        'ytick.color': WTF1_PALETTE['text'],
        'grid.color': WTF1_PALETTE['grid'],
        'grid.alpha': 0.3,
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.figsize': (12, 6)
    })

def plot_tire_degradation(
    laps: pd.DataFrame,
    tire_results: Dict[str, Any],
    title: str = "Tire Degradation Analysis",
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    apply_wtf1_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    compound_col = 'CompoundNorm' if 'CompoundNorm' in laps.columns else 'Compound'
    time_col = 'LapTimeSeconds'
    age_col = 'TireAge' if 'TireAge' in laps.columns else 'StintLap'
    
    for compound, color in COMPOUND_COLORS.items():
        if compound not in laps[compound_col].unique():
            continue
        
        comp_data = laps[laps[compound_col] == compound]
        if len(comp_data) < 3:
            continue
        
        x = comp_data[age_col].values
        y = comp_data[time_col].values
        
        ax.scatter(x, y, c=color, alpha=0.5, s=20, label=f'{compound} (actual)')
        
        if compound in tire_results:
            result = tire_results[compound]
            deg_rate = result.deg_rate_s_per_lap
            
            x_line = np.linspace(x.min(), x.max(), 50)
            intercept = y.mean() - deg_rate * x.mean()
            y_line = intercept + deg_rate * x_line
            
            ax.plot(x_line, y_line, color=color, linewidth=2,
                   label=f'{compound} fit (R²={result.r_squared:.2f})')
            
            if result.cliff_lap:
                edge_color = 'black' if compound == 'HARD' else color
                ax.axvline(result.cliff_lap, color=color, linestyle='--',
                          alpha=0.7, label=f'{compound} cliff ~L{result.cliff_lap}')
    
    ax.set_xlabel('Tire Age (laps)')
    ax.set_ylabel('Lap Time (seconds)')
    ax.set_title(title)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    ax.text(0.02, 0.02, 
            "[MODEL-DERIVED] Degradation rates from linear regression",
            transform=ax.transAxes, fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_fuel_corrected_pace(
    pace_data: pd.DataFrame,
    title: str = "Fuel-Corrected Pace Comparison",
    top_n: int = 10,
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:

    if not MATPLOTLIB_AVAILABLE:
        return None
    
    if len(pace_data) == 0:
        return None
    
    apply_wtf1_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = pace_data.head(top_n).copy()
    
    fastest = data['CorrectedPace'].min()
    data['Delta'] = data['CorrectedPace'] - fastest
    
    y_pos = np.arange(len(data))
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(data)))
    
    bars = ax.barh(y_pos, data['Delta'], color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data['Driver'])
    
    for i, (delta, pace) in enumerate(zip(data['Delta'], data['CorrectedPace'])):
        ax.text(delta + 0.02, i, f'+{delta:.3f}s ({pace:.2f}s)',
                va='center', fontsize=9)
    
    ax.set_xlabel('Delta to Fastest (seconds)')
    ax.set_title(title)
    ax.axvline(0, color=WTF1_PALETTE['primary'], linewidth=2)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # Fastest at top
    
    ax.text(0.02, 0.98, 
            "[FUEL-CORRECTED] Removes fuel weight effect for fair comparison",
            transform=ax.transAxes, fontsize=8, alpha=0.7, va='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_strategy_comparison(
    strategies: List[Dict[str, Any]],
    title: str = "Strategy Comparison",
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    apply_wtf1_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    strategy_names = [s.get('compounds', f"Strategy {i}") for i, s in enumerate(strategies)]
    times = [s.get('expected_time', 0) for s in strategies]
    uncertainties = [s.get('uncertainty', 0) for s in strategies]
    
    colors = [STRATEGY_COLORS.get(f"{s.get('stops', 1)}-stop", '#666666') 
              for s in strategies]
    
    bars = ax1.barh(range(len(strategies)), times, xerr=uncertainties,
                    color=colors, edgecolor='white', linewidth=0.5, capsize=3)
    
    ax1.set_yticks(range(len(strategies)))
    ax1.set_yticklabels(strategy_names)
    ax1.set_xlabel('Expected Race Time (seconds)')
    ax1.set_title('Strategy Times (with uncertainty)')
    ax1.invert_yaxis()
    
    best_idx = np.argmin(times)
    bars[best_idx].set_edgecolor(WTF1_PALETTE['primary'])
    bars[best_idx].set_linewidth(3)
    
    best_time = min(times)
    deltas = [t - best_time for t in times]
    
    ax2.barh(range(len(strategies)), deltas, color=colors,
             edgecolor='white', linewidth=0.5)
    ax2.set_yticks(range(len(strategies)))
    ax2.set_yticklabels(strategy_names)
    ax2.set_xlabel('Delta to Best Strategy (seconds)')
    ax2.set_title('Gap to Optimal')
    ax2.axvline(0, color=WTF1_PALETTE['primary'], linewidth=2)
    ax2.invert_yaxis()
    
    for i, delta in enumerate(deltas):
        ax2.text(delta + 0.2, i, f'+{delta:.1f}s' if delta > 0 else 'BEST',
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_confidence_envelope(
    envelope_data: Dict[str, Any],
    title: str = "Strategy Confidence Envelope",
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    apply_wtf1_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    hist_data = envelope_data.get('histogram_data', None)
    if hist_data is not None:
        counts, bins = hist_data
        ax1.bar(bins[:-1], counts, width=np.diff(bins), 
                color=WTF1_PALETTE['accent'], edgecolor='white',
                alpha=0.8, align='edge')
        
        percentiles = envelope_data.get('percentiles', {})
        for p_name, p_val in percentiles.items():
            ax1.axvline(p_val, color=WTF1_PALETTE['primary'], 
                       linestyle='--' if p_name != 'p50' else '-',
                       alpha=0.8, label=f'{p_name}: {p_val}s')
        
        ax1.set_xlabel('Race Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Race Time Distribution (1000 simulations)')
        ax1.legend(fontsize=8)
    
    mean = envelope_data.get('mean', 0)
    best = envelope_data.get('best_case', 0)
    worst = envelope_data.get('worst_case', 0)
    robustness = envelope_data.get('robustness', 'Unknown')
    
    ax2.fill_between([0, 1], [best, best], [worst, worst],
                     color=WTF1_PALETTE['accent'], alpha=0.3,
                     label='Confidence envelope')
    ax2.axhline(mean, color=WTF1_PALETTE['primary'], linewidth=2,
               label=f'Mean: {mean}s')
    ax2.axhline(best, color='green', linestyle='--', 
               label=f'Best: {best}s')
    ax2.axhline(worst, color='red', linestyle='--',
               label=f'Worst: {worst}s')
    
    ax2.set_ylabel('Race Time (seconds)')
    ax2.set_title(f'Confidence Envelope (Robustness: {robustness})')
    ax2.set_xticks([])
    ax2.legend()
    
    width = envelope_data.get('envelope_width', 0)
    ax2.annotate(f'Width: {width}s',
                xy=(0.5, (best + worst) / 2),
                fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_race_summary(
    laps: pd.DataFrame,
    driver: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:

    if not MATPLOTLIB_AVAILABLE:
        return None
    
    driver_laps = laps[laps['Driver'] == driver].copy()
    if len(driver_laps) == 0:
        return None
    
    apply_wtf1_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ax1 = axes[0]
    compound_col = 'CompoundNorm' if 'CompoundNorm' in driver_laps.columns else 'Compound'
    
    for compound in driver_laps[compound_col].unique():
        mask = driver_laps[compound_col] == compound
        color = COMPOUND_COLORS.get(compound, '#888888')
        ax1.scatter(driver_laps.loc[mask, 'LapNumber'],
                   driver_laps.loc[mask, 'LapTimeSeconds'],
                   c=color, s=30, label=compound, edgecolors='white', linewidth=0.5)
    
    ax1.set_ylabel('Lap Time (seconds)')
    ax1.set_title(title or f'{driver} Race Summary')
    ax1.legend(title='Compound', loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    if 'Position' in driver_laps.columns:
        ax2.plot(driver_laps['LapNumber'], driver_laps['Position'],
                color=WTF1_PALETTE['accent'], linewidth=2, marker='o', markersize=3)
        ax2.set_ylabel('Position')
        ax2.invert_yaxis()
        ax2.set_ylim(20, 0)
        ax2.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Lap Number')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_stint_timeline(
    stint_data: pd.DataFrame,
    title: str = "Race Stint Timeline",
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    if len(stint_data) == 0:
        return None
    
    apply_wtf1_style()
    
    drivers = stint_data['Driver'].unique()
    n_drivers = len(drivers)
    
    fig, ax = plt.subplots(figsize=(14, max(6, n_drivers * 0.4)))
    
    for i, driver in enumerate(sorted(drivers)):
        driver_stints = stint_data[stint_data['Driver'] == driver].sort_values('Stint')
        
        for _, stint in driver_stints.iterrows():
            start = stint['StartLap']
            end = stint['EndLap']
            compound = stint['Compound']
            color = COMPOUND_COLORS.get(compound, '#888888')            
            ax.barh(i, end - start, left=start, height=0.6,
                   color=color, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(n_drivers))
    ax.set_yticklabels(sorted(drivers))
    ax.set_xlabel('Lap Number')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    legend_elements = [
        mpatches.Patch(facecolor=color, edgecolor='white', label=compound)
        for compound, color in COMPOUND_COLORS.items()
        if compound in stint_data['Compound'].values
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Compound')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def format_strategy_output(strategy_result) -> str:
    if hasattr(strategy_result, 'to_dict'):
        s = strategy_result.to_dict()
    else:
        s = strategy_result
    
    output = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEUTRAL STRATEGY (Model-Optimal)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Optimal Stop Count: {s.get('stop_count', 'N/A')}-stop
Compound Sequence:  {s.get('compound_sequence', 'N/A')}
Pit Windows:        {_format_pit_windows(s.get('pit_windows', []))}

Expected Race Time: {s.get('expected_time', 'N/A')}s ±{s.get('uncertainty', 'N/A')}s

Confidence Envelope:
  ├── Best case:  {s.get('best_case', 'N/A')}s
  ├── Mean:       {s.get('expected_time', 'N/A')}s
  └── Worst case: {s.get('worst_case', 'N/A')}s

Confidence Level: {s.get('confidence', 0)*100:.0f}%

Reasoning:
{s.get('reasoning', 'No reasoning available')}

Alternative Strategies:
"""
    
    for i, alt in enumerate(s.get('alternatives', []), 1):
        output += f"  {i}. {alt.get('compounds', 'N/A')} ({alt.get('stops', '?')}-stop): "
        output += f"+{alt.get('delta_to_best', '?')}s\n"
    
    output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[NOTE: This is MODEL-OPTIMAL based on neutral conditions.
 Actual team strategies may differ due to car-specific factors,
 driver preferences, and tactical considerations.]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return output


def format_sc_decision(sc_result) -> str:
    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAFETY CAR DECISION ADVISOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recommendation: {sc_result.recommendation}
Confidence: {sc_result.confidence * 100:.0f}%

{sc_result.reasoning}

Alternative: {sc_result.alternative_scenario}

Expected Position:
  • If PIT:  P{sc_result.expected_positions.get('after_pit', '?')}
  • If STAY: P{sc_result.expected_positions.get('if_stay', '?')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

def format_undercut_analysis(undercut_result) -> str:
    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNDERCUT/OVERCUT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recommendation: {undercut_result.recommendation}

Undercut Probability: {undercut_result.undercut_probability * 100:.0f}%
Required Out-lap Delta: {undercut_result.required_outlap_delta:.2f}s

Overcut Window: {_format_window(undercut_result.overcut_window)}

{undercut_result.reasoning}

Gap Evolution (next 10 laps): {undercut_result.gap_evolution}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

def _format_pit_windows(windows: List[Tuple[int, int]]) -> str:
    if not windows:
        return "N/A"
    return ", ".join([f"L{start}-L{end}" for start, end in windows])

def _format_window(window: Tuple[int, int]) -> str:
    if not window or window == (0, 0):
        return "Not available"
    return f"L{window[0]}-L{window[1]}"

def generate_race_report(
    track_name: str,
    year: int,
    strategy_result,
    tire_analysis: Dict[str, Any],
    pace_analysis: pd.DataFrame,
    envelope_data: Dict[str, Any],
    save_dir: Optional[str] = None
) -> str:
    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        WTF1 RACE ANALYSIS REPORT                             ║
║                        {track_name} {year}                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

{format_strategy_output(strategy_result)}

═══════════════════════════════════════════════════════════════════════════════
TIRE DEGRADATION ANALYSIS
═══════════════════════════════════════════════════════════════════════════════
"""
    
    for compound, result in tire_analysis.items():
        report += f"""
{compound}:
  Degradation Rate: {result.deg_rate_s_per_lap:.4f} s/lap
  Model Fit (R²): {result.r_squared:.3f}
  Estimated Cliff: Lap {result.cliff_lap or 'N/A'}
  Sample Size: {result.sample_size} laps
  {result.reasoning}
"""
    report += """
═══════════════════════════════════════════════════════════════════════════════
PACE COMPARISON (Fuel-Corrected)
═══════════════════════════════════════════════════════════════════════════════
"""
    
    if len(pace_analysis) > 0:
        for i, row in pace_analysis.head(5).iterrows():
            gap = row['CorrectedPace'] - pace_analysis['CorrectedPace'].min()
            report += f"  {row['Driver']}: {row['CorrectedPace']:.3f}s (+{gap:.3f}s)\n"
    
    report += f"""
═══════════════════════════════════════════════════════════════════════════════
CONFIDENCE ENVELOPE
═══════════════════════════════════════════════════════════════════════════════
  Robustness: {envelope_data.get('robustness', 'Unknown')}
  Mean Time: {envelope_data.get('mean', 'N/A')}s
  Best Case (5th %ile): {envelope_data.get('best_case', 'N/A')}s
  Worst Case (95th %ile): {envelope_data.get('worst_case', 'N/A')}s
  Envelope Width: {envelope_data.get('envelope_width', 'N/A')}s
  SC Impact: {envelope_data.get('sc_impact', 'N/A')}s (with SC vs without)

═══════════════════════════════════════════════════════════════════════════════
DISCLAIMER
═══════════════════════════════════════════════════════════════════════════════
This report is generated by WTF1 using public FastF1 data.
All values are MODEL-DERIVED estimates with explicit uncertainty.
This is NOT official team data and should not be used for betting/trading.
This analysis is for educational and portfolio demonstration purposes only.
═══════════════════════════════════════════════════════════════════════════════
"""
    return report

if __name__ == "__main__":
    print("WTF1 Visualization Module - Testing")
    print("=" * 60)
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
    else:
        apply_wtf1_style()
        print("[OK] Style applied successfully")
        
        print("\nTesting text formatters...")
        
        mock_strategy = {
            'stop_count': 2,
            'compound_sequence': 'SOFT-MEDIUM-HARD',
            'pit_windows': [(15, 18), (35, 38)],
            'expected_time': 5500,
            'uncertainty': 8,
            'best_case': 5492,
            'worst_case': 5508,
            'confidence': 0.75,
            'reasoning': '  • 2-stop minimizes degradation cost\n  • Traffic risk is low',
            'alternatives': [
                {'compounds': 'SOFT-HARD', 'stops': 1, 'delta_to_best': 12.5},
                {'compounds': 'MEDIUM-HARD', 'stops': 1, 'delta_to_best': 8.3}
            ]
        }
        output = format_strategy_output(mock_strategy)
        print(output[:500] + "...")
        
        print("\n" + "=" * 60)
        print("Visualization module tests complete! [OK]")