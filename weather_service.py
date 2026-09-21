"""
KrishiMitra Real-Time Weather Service
Fetches live microclimate weather and forecast data for any district/location using Open-Meteo.
Requires ZERO API keys.
"""

import requests

class LiveWeatherService:
    def __init__(self):
        self.geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"

    def get_weather_by_city(self, city_name="Nashik"):
        """
        Geocodes the city name and retrieves real-time temperature, humidity,
        wind speed, precipitation, and rain probability for the next 24-48 hours.
        """
        try:
            # 1. Geocode city name to lat/lon
            geo_params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
            geo_resp = requests.get(self.geo_url, params=geo_params, timeout=5)
            
            if geo_resp.status_code != 200 or not geo_resp.json().get("results"):
                return self._fallback_weather(city_name, error="City not found, using regional baseline.")
                
            loc_data = geo_resp.json()["results"][0]
            lat = loc_data["latitude"]
            lon = loc_data["longitude"]
            resolved_name = f"{loc_data.get('name')}, {loc_data.get('admin1', '')} ({loc_data.get('country', '')})"

            # 2. Fetch live weather & forecast
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
                "timezone": "auto"
            }
            w_resp = requests.get(self.weather_url, params=weather_params, timeout=5)
            if w_resp.status_code != 200:
                return self._fallback_weather(city_name, error="Weather service timeout.")

            data = w_resp.json()
            curr = data.get("current", {})
            daily = data.get("daily", {})

            temp_current = curr.get("temperature_2m", 28.0)
            humidity_current = curr.get("relative_humidity_2m", 70.0)
            precipitation_current = curr.get("precipitation", 0.0)
            wind_speed = curr.get("wind_speed_10m", 10.0)

            temp_max = daily.get("temperature_2m_max", [temp_current + 3])[0]
            temp_min = daily.get("temperature_2m_min", [temp_current - 6])[0]
            rain_prob = daily.get("precipitation_probability_max", [20])[0]
            rain_sum = daily.get("precipitation_sum", [precipitation_current])[0]

            return {
                "success": True,
                "location": resolved_name,
                "latitude": lat,
                "longitude": lon,
                "temp_current": float(temp_current),
                "temp_max": float(temp_max),
                "temp_min": float(temp_min),
                "humidity_morning": float(min(humidity_current + 10, 98.0)),
                "humidity_evening": float(max(humidity_current - 15, 25.0)),
                "rainfall_mm": float(rain_sum),
                "rain_probability": int(rain_prob),
                "wind_speed_kmh": float(wind_speed),
                "consecutive_wet_days": 2 if rain_prob > 60 else (1 if rain_prob > 30 else 0)
            }

        except Exception as e:
            return self._fallback_weather(city_name, error=str(e))

    def _fallback_weather(self, city_name, error=""):
        return {
            "success": False,
            "error": error,
            "location": f"{city_name} (Simulated Agro-Climatic Data)",
            "latitude": 20.0,
            "longitude": 75.0,
            "temp_current": 29.5,
            "temp_max": 32.0,
            "temp_min": 22.0,
            "humidity_morning": 82.0,
            "humidity_evening": 60.0,
            "rainfall_mm": 5.0,
            "rain_probability": 35,
            "wind_speed_kmh": 12.0,
            "consecutive_wet_days": 1
        }
