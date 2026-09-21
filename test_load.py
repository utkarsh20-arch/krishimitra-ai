"""
Quick sanity test script for KrishiMitra components.
Run: python test_load.py
"""

import sys
import os

print("🌾 Testing KrishiMitra Core Modules...")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vision_detector import CropVisionDetector
from models.weather_risk_predictor import WeatherRiskPredictor
from models.advisory_engine import GraniteAgriCopilot
from models.weather_service import LiveWeatherService
from models.llm_service import DynamicLLMAlertService

# 1. Test Weather Service
print("\n1. Testing Real-time Weather Service:")
weather_svc = LiveWeatherService()
w_data = weather_svc.get_weather_by_city("Nashik")
print(f"   ✅ Fetched Live Weather for: {w_data['location']}")
print(f"      Temp: {w_data['temp_current']}°C | Rain Prob: {w_data['rain_probability']}%")

# 2. Test Weather Risk Predictor
print("\n2. Testing ML Pest Outbreak Predictor:")
predictor = WeatherRiskPredictor()
res = predictor.predict(
    temp_max=w_data['temp_max'],
    temp_min=w_data['temp_min'],
    humidity_morning=w_data['humidity_morning'],
    humidity_evening=w_data['humidity_evening'],
    rainfall_mm=w_data['rainfall_mm'],
    consecutive_wet_days=w_data['consecutive_wet_days'],
    wind_speed_kmh=w_data['wind_speed_kmh'],
    crop_stage=2
)
print(f"   ✅ Outbreak Risk: {res['risk_level']} ({res['risk_percentage']}%)")
print(f"      Driver: {res['primary_driver']}")

# 3. Test Dynamic Alert Service (Gemini / Adaptive)
print("\n3. Testing Dynamic AI Alert Service:")
alert_svc = DynamicLLMAlertService()
alert = alert_svc.generate_weather_alert(
    location=w_data['location'],
    crop="Tomato",
    weather_data=w_data,
    risk_data=res,
    language="English"
)
print("   ✅ Generated Dynamic Alert:\n   " + alert[:150].replace("\n", " ") + "...")

# 4. Test Granite Copilot
print("\n4. Testing IBM Granite RAG Copilot:")
copilot = GraniteAgriCopilot()
adv = copilot.generate_advisory("Tomato", "yellow spots on leaves")
print(f"   ✅ Advisory Generated ({len(adv)} chars)")

print("\n🎉 ALL COMPONENTS VERIFIED AND OPERATIONAL!")
