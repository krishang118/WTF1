# WTF1 Data Engine
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    warnings.warn("FastF1 not installed. Run: pip install fastf1")

class TrackStatus(Enum):
    GREEN = 1       
    YELLOW = 2      
    SC = 4          
    RED = 5         
    VSC = 6         
    VSC_ENDING = 7  

class LapType(Enum):
    GREEN_FLAG = "green_flag"
    SAFETY_CAR = "safety_car"
    VSC = "vsc"
    PIT_IN = "pit_in"
    PIT_OUT = "pit_out"
    FIRST_LAP = "first_lap"
    INVALID = "invalid"

COMPOUND_MAPPING = {
    'SOFT': 'SOFT', 'S': 'SOFT', 'soft': 'SOFT',
    'MEDIUM': 'MEDIUM', 'M': 'MEDIUM', 'medium': 'MEDIUM',
    'HARD': 'HARD', 'H': 'HARD', 'hard': 'HARD',
    'INTERMEDIATE': 'INTERMEDIATE', 'I': 'INTERMEDIATE', 'inter': 'INTERMEDIATE',
    'WET': 'WET', 'W': 'WET', 'wet': 'WET',
    None: 'UNKNOWN', 'nan': 'UNKNOWN', '': 'UNKNOWN'
}

class DataEngine:
    
    def __init__(self, cache_dir: Optional[Path] = None):

        if not FASTF1_AVAILABLE:
            raise RuntimeError(
                "FastF1 is required. Install with: pip install fastf1"
            )
        
        self.cache_dir = cache_dir or Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))
        
        self.MAX_FUEL_KG = 110.0
        self.FUEL_BURN_RATE = 1.8 
        self.FUEL_PENALTY_PER_KG = 0.035  
        self.DIRTY_AIR_THRESHOLD = 1.2 
        self.DIRTY_AIR_PENALTY = 0.3  
    
    def load_session(
        self,
        year: int,
        gp: str,
        session: str = 'R',
        load_telemetry: bool = False,
        load_weather: bool = True
    ) -> Any:
        if year < 2018 or year > 2025:
            warnings.warn(
                f"Year {year} may have limited data. Best results: 2018-2025"
            )
        try:
            session_obj = fastf1.get_session(year, gp, session)
            session_obj.load(
                telemetry=load_telemetry,
                weather=load_weather,
                messages=False
            )
            return session_obj
        except Exception as e:
            raise ValueError(
                f"Failed to load session {gp} {year} {session}: {str(e)}"
            )
    def get_available_sessions(self, year: int) -> pd.DataFrame:
        try:
            schedule = fastf1.get_event_schedule(year)
            return schedule[['RoundNumber', 'EventName', 'Country', 'EventDate']]
        except Exception as e:
            raise ValueError(f"Failed to get schedule for {year}: {str(e)}")

    def clean_laps(
        self,
        laps: pd.DataFrame,
        remove_pit_laps: bool = True,
        remove_sc_laps: bool = False,
        remove_first_lap: bool = True,
        remove_outliers: bool = True,
        outlier_threshold: float = 1.5
    ) -> pd.DataFrame:
        if laps is None or len(laps) == 0:
            return pd.DataFrame()
        
        df = laps.copy()
        
        df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
        
        df['CompoundNorm'] = df['Compound'].map(
            lambda x: COMPOUND_MAPPING.get(str(x), 'UNKNOWN')
        )
        
        df['LapType'] = LapType.GREEN_FLAG.value
        
        if remove_first_lap:
            df = df[df['LapNumber'] > 1]
        else:
            df.loc[df['LapNumber'] == 1, 'LapType'] = LapType.FIRST_LAP.value
        
        pit_in_mask = df['PitInTime'].notna()
        pit_out_mask = df['PitOutTime'].notna()
        
        df.loc[pit_in_mask, 'LapType'] = LapType.PIT_IN.value
        df.loc[pit_out_mask, 'LapType'] = LapType.PIT_OUT.value
        
        if remove_pit_laps:
            df = df[~(pit_in_mask | pit_out_mask)]
        
        if 'TrackStatus' in df.columns:
            sc_mask = df['TrackStatus'].apply(
                lambda x: self._is_sc_status(x) if pd.notna(x) else False
            )
            df.loc[sc_mask, 'LapType'] = LapType.SAFETY_CAR.value
            
            if remove_sc_laps:
                df = df[~sc_mask]
        
        if 'Deleted' in df.columns:
            df.loc[df['Deleted'] == True, 'LapType'] = LapType.INVALID.value
            df = df[df['Deleted'] != True]
        
        if remove_outliers and 'LapTimeSeconds' in df.columns:
            df = self._remove_outliers(df, 'LapTimeSeconds', outlier_threshold)
        
        df = df.dropna(subset=['LapTimeSeconds', 'LapNumber'])
        
        return df.reset_index(drop=True)
    
    def _is_sc_status(self, status: Any) -> bool:
        try:
            status_str = str(status)
            return any(code in status_str for code in ['4', '6', '7'])
        except:
            return False
    
    def _remove_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        threshold: float = 1.5
    ) -> pd.DataFrame:
        if column not in df.columns:
            return df
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    def engineer_features(
        self,
        laps: pd.DataFrame,
        session: Optional[Any] = None
    ) -> pd.DataFrame:
        if laps is None or len(laps) == 0:
            return pd.DataFrame()
        
        df = laps.copy()
        
        if 'TyreLife' in df.columns:
            df['TireAge'] = df['TyreLife']
        elif 'Stint' in df.columns:
            df['TireAge'] = df.groupby(['Driver', 'Stint']).cumcount() + 1
        else:
            df['TireAge'] = df.groupby('Driver').cumcount() + 1
        
        df['FuelEstimate'] = df['LapNumber'].apply(
            lambda lap: max(0, self.MAX_FUEL_KG - lap * self.FUEL_BURN_RATE)
        )
        
        df['FuelPenalty'] = df['FuelEstimate'] * self.FUEL_PENALTY_PER_KG
        
        if 'LapTimeSeconds' in df.columns:
            race_midpoint_fuel = self.MAX_FUEL_KG / 2
            df['FuelCorrectedTime'] = (
                df['LapTimeSeconds'] + 
                (race_midpoint_fuel - df['FuelEstimate']) * self.FUEL_PENALTY_PER_KG
            )
        df = self._calculate_gaps(df)
        
        if 'GapAhead' in df.columns:
            df['CleanAirFlag'] = df['GapAhead'] > self.DIRTY_AIR_THRESHOLD
            
            df['TrafficPenalty'] = df.apply(
                lambda row: self._estimate_traffic_penalty(row),
                axis=1
            )
            
            df['DRSAvailable'] = df['GapAhead'].apply(
                lambda gap: 0 < gap <= 1.0 if pd.notna(gap) else False
            )
        else:
            df['CleanAirFlag'] = True
            df['TrafficPenalty'] = 0.0
            df['DRSAvailable'] = False
        if 'Stint' in df.columns:
            df['StintLap'] = df.groupby(['Driver', 'Stint']).cumcount() + 1
        
        return df
    
    def _calculate_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Position' not in df.columns or 'LapNumber' not in df.columns:
            return df
        
        gaps_ahead = []
        gaps_behind = []
        
        for _, lap in df.iterrows():
            lap_num = lap['LapNumber']
            driver = lap['Driver']
            position = lap.get('Position', None)
            
            if pd.isna(position) or position <= 1:
                gaps_ahead.append(np.nan)
            else:
                car_ahead = df[
                    (df['LapNumber'] == lap_num) & 
                    (df['Position'] == position - 1)
                ]
                if len(car_ahead) > 0:
                    gap = abs(
                        lap['LapTimeSeconds'] - 
                        car_ahead.iloc[0]['LapTimeSeconds']
                    )
                    gaps_ahead.append(gap)
                else:
                    gaps_ahead.append(np.nan)
            
            car_behind = df[
                (df['LapNumber'] == lap_num) & 
                (df['Position'] == position + 1 if pd.notna(position) else False)
            ]
            if len(car_behind) > 0:
                gap = abs(
                    lap['LapTimeSeconds'] - 
                    car_behind.iloc[0]['LapTimeSeconds']
                )
                gaps_behind.append(gap)
            else:
                gaps_behind.append(np.nan)
        
        df['GapAhead'] = gaps_ahead
        df['GapBehind'] = gaps_behind
        
        return df
    
    def _estimate_traffic_penalty(self, row: pd.Series) -> float:
        gap = row.get('GapAhead', None)
        
        if pd.isna(gap) or gap > self.DIRTY_AIR_THRESHOLD:
            return 0.0
        if gap < 0.5:
            return self.DIRTY_AIR_PENALTY * 1.5  
        elif gap < 1.0:
            return self.DIRTY_AIR_PENALTY * 1.2
        else:
            return self.DIRTY_AIR_PENALTY

    def get_driver_laps(
        self,
        laps: pd.DataFrame,
        driver: str
    ) -> pd.DataFrame:
        return laps[laps['Driver'] == driver].copy()
    
    def get_stint_summary(self, laps: pd.DataFrame) -> pd.DataFrame:
        if laps is None or len(laps) == 0:
            return pd.DataFrame()
        
        summary = []
        compound_col = 'CompoundNorm' if 'CompoundNorm' in laps.columns else 'Compound'
        
        for (driver, stint), group in laps.groupby(['Driver', 'Stint']):
            stint_data = {
                'Driver': driver,
                'Stint': stint,
                'Compound': group[compound_col].iloc[0] if compound_col in group.columns else 'UNKNOWN',
                'StartLap': group['LapNumber'].min(),
                'EndLap': group['LapNumber'].max(),
                'TotalLaps': len(group),
                'AvgLapTime': group['LapTimeSeconds'].mean() if 'LapTimeSeconds' in group.columns else None
            }
            
            if 'LapTimeSeconds' in group.columns and len(group) > 3:
                times = group['LapTimeSeconds'].values
                laps_arr = np.arange(len(times))
                if len(times) > 1:
                    slope, _ = np.polyfit(laps_arr, times, 1)
                    stint_data['DegRate'] = max(0, slope) 
                else:
                    stint_data['DegRate'] = None
            else:
                stint_data['DegRate'] = None
            
            summary.append(stint_data)
        
        return pd.DataFrame(summary)
    
    def get_compound_statistics(
        self,
        laps: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        compound_col = 'CompoundNorm' if 'CompoundNorm' in laps.columns else 'Compound'
        time_col = 'LapTimeSeconds'
        
        if time_col not in laps.columns:
            return {}
        
        stats = {}
        for compound in laps[compound_col].unique():
            if compound == 'UNKNOWN':
                continue
            
            compound_laps = laps[laps[compound_col] == compound]
            times = compound_laps[time_col].dropna()
            
            if len(times) > 0:
                stats[compound] = {
                    'avg_time': times.mean(),
                    'min_time': times.min(),
                    'max_time': times.max(),
                    'std_time': times.std() if len(times) > 1 else 0,
                    'lap_count': len(times),
                    'drivers': compound_laps['Driver'].nunique()
                }
        
        return stats
    
    def extract_weather_data(self, session: Any) -> Dict[str, Any]:
        try:
            weather = session.weather_data
            if weather is None or len(weather) == 0:
                return {'available': False}
            
            return {
                'available': True,
                'avg_air_temp': weather['AirTemp'].mean(),
                'avg_track_temp': weather['TrackTemp'].mean(),
                'avg_humidity': weather['Humidity'].mean() if 'Humidity' in weather.columns else None,
                'rain_detected': weather['Rainfall'].any() if 'Rainfall' in weather.columns else False,
                'wind_speed_avg': weather['WindSpeed'].mean() if 'WindSpeed' in weather.columns else None
            }
        except:
            return {'available': False}

def load_session(year: int, gp: str, session: str = 'R') -> Any:
    engine = DataEngine()
    return engine.load_session(year, gp, session)

def quick_clean(laps: pd.DataFrame) -> pd.DataFrame:
    engine = DataEngine()
    return engine.clean_laps(laps)

def quick_features(laps: pd.DataFrame) -> pd.DataFrame:
    engine = DataEngine()
    cleaned = engine.clean_laps(laps)
    return engine.engineer_features(cleaned)

if __name__ == "__main__":
    print("WTF1 Data Engine - Testing")
    print("=" * 60)
    
    try:
        engine = DataEngine()
        
        print("\n1. Testing session loading (Bahrain 2024 Race)...")
        session = engine.load_session(2024, 'Bahrain', 'R')
        print(f"   Loaded: {session.event['EventName']}")
        print(f"   Laps available: {len(session.laps)}")
        
        print("\n2. Testing data cleaning...")
        clean_laps = engine.clean_laps(session.laps)
        print(f"   Clean laps: {len(clean_laps)}")
        print(f"   Lap types: {clean_laps['LapType'].value_counts().to_dict()}")
        
        print("\n3. Testing feature engineering...")
        featured = engine.engineer_features(clean_laps, session)
        new_features = ['TireAge', 'FuelEstimate', 'FuelPenalty', 'CleanAirFlag']
        available = [f for f in new_features if f in featured.columns]
        print(f"   Features added: {available}")
        
        print("\n4. Testing stint summary...")
        stints = engine.get_stint_summary(featured)
        print(f"   Total stints: {len(stints)}")
        if len(stints) > 0:
            print(f"   Compounds used: {stints['Compound'].unique().tolist()}")
        
        print("\n5. Testing compound statistics...")
        stats = engine.get_compound_statistics(featured)
        for compound, info in stats.items():
            print(f"   {compound}: avg={info['avg_time']:.3f}s, n={info['lap_count']}")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nMake sure FastF1 is installed: pip install fastf1")