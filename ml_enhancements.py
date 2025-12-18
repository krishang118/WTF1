# WTF1 Advanced ML Enhancements
import warnings
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. ML enhancements disabled.")

from sklearn.model_selection import train_test_split

class XGBoostTireEngine:

    def __init__(self):
        self.models = {}
        self.feature_cols = ['TireAge', 'FuelEstimate', 'TrafficIndex', 'TrackStatus']
    
    def train_model(self, laps: pd.DataFrame) -> Dict[str, float]:
        if not XGBOOST_AVAILABLE:
            return {}
        
        results = {}
        
        df = laps.copy()
        if 'TrafficIndex' not in df.columns:
            df['TrafficIndex'] = 0.0
        if 'TrackStatus' not in df.columns:
            df['TrackStatus'] = 1
            
        compounds = df['Compound'].unique()
        
        for compound in compounds:
            comp_data = df[df['Compound'] == compound].dropna(subset=self.feature_cols + ['LapTimeSeconds'])
            
            if len(comp_data) < 20: 
                continue
            
            X = comp_data[self.feature_cols].copy()
            
            for col in self.feature_cols:
                if col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            y = comp_data['LapTimeSeconds']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                objective='reg:squarederror',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - preds)**2))
            
            self.models[compound] = model
            results[compound] = float(rmse)
    
        return results

    def predict_degradation_curve(
        self, 
        compound: str, 
        max_laps: int = 40,
        start_fuel: float = 30.0
    ) -> Dict[str, List[float]]:
        if not XGBOOST_AVAILABLE or compound not in self.models:
            return {}
        
        model = self.models[compound]
        
        lap_range = range(1, max_laps + 1)
        synthetic_data = pd.DataFrame({
            'TireAge': lap_range,
            'FuelEstimate': [max(0, start_fuel - 1.8 * l) for l in lap_range],
            'TrafficIndex': [0.0] * max_laps,
            'TrackStatus': [1] * max_laps
        })
        
        preds = model.predict(synthetic_data[self.feature_cols])
        
        return {
            'laps': list(lap_range),
            'predicted_times': [round(float(p), 3) for p in preds]
        }

if __name__ == "__main__":
    print("WTF1 ML Enhancements Test")
    print("=" * 60)
    
    print("Generating mock data...")
    mock_laps = pd.DataFrame({
        'Compound': ['SOFT'] * 50 + ['HARD'] * 50,
        'TireAge': list(range(1, 51)) * 2,
        'FuelEstimate': [100 - x*1.5 for x in range(100)],
        'TrafficIndex': np.random.uniform(0, 1, 100),
        'TrackStatus': [1] * 100,
        'LapTimeSeconds': [90 + x*0.05 - (100-x)*0.03 for x in range(100)]
    })    
    if XGBOOST_AVAILABLE:
        print("\nTesting XGBoostTireEngine...")
        xgb_engine = XGBoostTireEngine()
        scores = xgb_engine.train_model(mock_laps)
        print(f"  Training RMSE: {scores}")
        
        curve = xgb_engine.predict_degradation_curve('SOFT')
        print(f"  Predicted curve (first 5): {curve.get('predicted_times', [])[:5]}")
    else:
        print("\nXGBoost skipped.")
    print("\nTest Complete.")