"""
KrishiMitra Weather Risk Predictor Module
Loads the trained XGBoost model and computes risk probability scores and primary drivers.
"""

import os
import pickle
import numpy as np
import pandas as pd

class WeatherRiskPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'weather_model.pkl')
        
        self.model_path = model_path
        self.package = None
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.package = pickle.load(f)
        else:
            self.package = None

    def is_ready(self):
        return self.package is not None

    def predict(self, temp_max, temp_min, humidity_morning, humidity_evening, 
                rainfall_mm, consecutive_wet_days, wind_speed_kmh, crop_stage=1):
        """
        Takes microclimate parameters and returns risk level, probability distribution, and key driver.
        """
        if self.package is None:
            # Fallback heuristic calculation if model not trained yet
            heuristic_score = (
                0.4 * (humidity_morning / 100.0) +
                0.3 * (consecutive_wet_days / 7.0) +
                0.2 * (humidity_evening / 100.0) +
                0.1 * (rainfall_mm > 5.0)
            )
            level = "High Risk" if heuristic_score > 0.65 else ("Moderate Risk" if heuristic_score > 0.45 else "Low Risk")
            probs = [0.1, 0.3, 0.6] if level == "High Risk" else ([0.2, 0.6, 0.2] if level == "Moderate Risk" else [0.7, 0.2, 0.1])
            return {
                'risk_level': level,
                'risk_percentage': round(float(heuristic_score * 100), 1),
                'probabilities': {'Low Risk': probs[0], 'Moderate Risk': probs[1], 'High Risk': probs[2]},
                'primary_driver': "High relative humidity and sustained leaf wetness"
            }
        
        model = self.package['model']
        feature_names = self.package['feature_names']
        
        input_data = pd.DataFrame([{
            'temp_max': float(temp_max),
            'temp_min': float(temp_min),
            'humidity_morning': float(humidity_morning),
            'humidity_evening': float(humidity_evening),
            'rainfall_mm': float(rainfall_mm),
            'consecutive_wet_days': int(consecutive_wet_days),
            'wind_speed_kmh': float(wind_speed_kmh),
            'crop_stage': int(crop_stage)
        }])[feature_names]
        
        probs = model.predict_proba(input_data)[0]
        class_idx = int(np.argmax(probs))
        class_names = self.package['class_names']
        risk_level = class_names[class_idx]
        
        # Calculate primary risk driver based on input thresholds
        drivers = []
        if humidity_morning >= 80:
            drivers.append(f"High morning humidity ({humidity_morning}%) favoring fungal spore proliferation")
        if consecutive_wet_days >= 3:
            drivers.append(f"Extended wet foliage over {consecutive_wet_days} consecutive days")
        if 20 <= temp_max <= 30 and humidity_morning >= 75:
            drivers.append(f"Optimal insect reproduction thermal band ({temp_max}°C)")
        if rainfall_mm >= 10:
            drivers.append("Recent precipitation washing away protective cuticle wax")
            
        primary_driver = "; ".join(drivers) if drivers else "Favorable climatic equilibrium; low pest pressure"
        risk_pct = round(float(probs[2] * 100 + probs[1] * 40), 1)
        risk_pct = min(max(risk_pct, 5.0), 99.0)

        return {
            'risk_level': risk_level,
            'risk_percentage': risk_pct,
            'probabilities': {
                'Low Risk': round(float(probs[0]), 3),
                'Moderate Risk': round(float(probs[1]), 3),
                'High Risk': round(float(probs[2]), 3)
            },
            'primary_driver': primary_driver
        }
