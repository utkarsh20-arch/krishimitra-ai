"""
KrishiMitra Dynamic AI Alert & Advisory Service
Integrates Google Gemini and OpenAI ChatGPT APIs to generate unique,
context-aware, real-time weather and pest outbreak alerts for farmers.
"""

import os
import json
import random
import requests

# Load .env automatically if present
try:
    from dotenv import load_dotenv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(base_dir, '.env'))
except ImportError:
    pass

class DynamicLLMAlertService:
    def __init__(self):
        pass

    def generate_weather_alert(self, location, crop, weather_data, risk_data, 
                               language="English", provider="Gemini", api_key=None):
        """
        Generates a unique, dynamic, highly contextual agricultural alert based on real-time microclimate feeds.
        Supports Google Gemini API, OpenAI ChatGPT API, and an intelligent dynamic local fallback.
        """
        # 1. Try Google Gemini API if key is available
        gemini_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if provider.lower() == "gemini" and gemini_key and len(gemini_key) > 10:
            result = self._call_gemini_api(location, crop, weather_data, risk_data, language, gemini_key)
            if result:
                return result

        # 2. Try OpenAI API if selected
        openai_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if provider.lower() == "openai" and openai_key and len(openai_key) > 10:
            result = self._call_openai_api(location, crop, weather_data, risk_data, language, openai_key)
            if result:
                return result

        # 3. Dynamic Local Intelligent Fallback (Generates unique, varied alerts every time)
        return self._generate_dynamic_local_alert(location, crop, weather_data, risk_data, language)

    def _call_gemini_api(self, location, crop, weather, risk, language, api_key):
        """
        Calls Google Gemini API via lightweight HTTP REST.
        """
        # Try current models
        models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        prompt = self._build_prompt(location, crop, weather, risk, language)
        
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 600
                }
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        if content_parts:
                            return f"✨ **[Live Gemini AI Advisory]**\n\n" + content_parts[0].get("text", "")
            except Exception:
                continue
        return None

    def _call_openai_api(self, location, crop, weather, risk, language, api_key):
        """
        Calls OpenAI ChatGPT API via HTTP REST.
        """
        prompt = self._build_prompt(location, crop, weather, risk, language)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are KrishiMitra, an empathetic expert agronomist providing organic, climate-aware pest alerts to farmers."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 600
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return f"✨ **[Live ChatGPT AI Advisory]**\n\n" + text
        except Exception:
            pass
        return None

    def _build_prompt(self, location, crop, weather, risk, language):
        return f"""
You are KrishiMitra, an expert AI agricultural scientist specializing in sustainable, low-cost, bio-organic farming for smallholder farmers.

Analyze the following real-time environmental snapshot and generate a unique, practical, and highly empathetic advisory for the farmer:

- Location: {location}
- Target Crop: {crop}
- Current Temp: {weather.get('temp_current', 28)}°C (High: {weather.get('temp_max', 32)}°C, Low: {weather.get('temp_min', 22)}°C)
- Relative Humidity: Morning {weather.get('humidity_morning', 80)}%, Evening {weather.get('humidity_evening', 60)}%
- Rain Probability in next 24h: {weather.get('rain_probability', 20)}% (Volume: {weather.get('rainfall_mm', 0)} mm)
- Wind Speed: {weather.get('wind_speed_kmh', 10)} km/h
- Consecutive Wet Days: {weather.get('consecutive_wet_days', 0)} days
- ML Pest Outbreak Risk: {risk.get('risk_level', 'Moderate')} ({risk.get('risk_percentage', 50)}%)
- Primary Weather Driver: {risk.get('primary_driver', 'Normal')}

Guidelines:
1. Start with a direct, conversational verdict (e.g., whether to spray today or hold off due to rain/humidity).
2. Explain specifically HOW today's humidity and temperature influence pest and fungal behavior for {crop}.
3. Give 2-3 step-by-step, low-cost bio-control actions (e.g. Neem oil, Trichoderma, pheromone traps, pruning).
4. Add an important safety/responsible AI caution (avoiding hazardous chemicals).
5. Language to respond in: {language}.
6. Keep it actionable, structured with bullet points, and under 250 words.
"""

    def _generate_dynamic_local_alert(self, location, crop, weather, risk, language):
        """
        Dynamically generates varied, non-repetitive agronomic alerts locally
        when no external API key is provided.
        """
        city = location.split(",")[0].strip()
        rain_prob = weather.get('rain_probability', 0)
        humidity = weather.get('humidity_morning', 75)
        temp = weather.get('temp_current', 28)
        risk_pct = risk.get('risk_percentage', 50)
        risk_level = risk.get('risk_level', 'Moderate Risk')
        
        # Varied greetings
        greetings_en = [
            f"🌾 **Field Advisory for {city} farmers growing {crop}:**",
            f"📢 **Real-Time Climate & Pest Bulletin | {city} ({crop}):**",
            f"🚜 **KrishiMitra Diagnostic Alert for {crop} in {city}:**"
        ]
        greetings_hi = [
            f"🌾 **{city} के {crop} उत्पादक किसान भाइयों के लिए विशेष मौसम सलाह:**",
            f"📢 **ताज़ा मौसम व कीट सुरक्षा बुलेटिन | {city} ({crop}):**",
            f"🚜 **कृषि-मित्र दैनिक फसल सुरक्षा परामर्श ({city}):**"
        ]
        
        greeting = random.choice(greetings_hi if language == "Hindi" else greetings_en)

        if rain_prob >= 45:
            # Rain dominant
            if language == "Hindi":
                action = (
                    f"⛔ **तत्काल छिड़काव रोकें!**\n"
                    f"अगले 24 घंटों में {city} में **{rain_prob}% बारिश की संभावना** है। आज पत्तों पर कोई भी दवा या जैविक घोल न छिड़कें, अन्यथा बारिश से पूरी दवा बह जाएगी।\n\n"
                    f"🔍 **आज क्या करें:**\n"
                    f"1. खेत के जल-निकासी (drainage) नालों को साफ करें ताकि जड़ों में पानी न ठहरे।\n"
                    f"2. बारिश थमने के बाद सुबह के समय **नीम तेल (5ml/लीटर)** का छिड़काव शेड्यूल करें।\n"
                    f"3. मिट्टी में फंगस रोकने हेतु जड़ के पास ट्राइकोडर्मा विरिडी का बुरकाव करें।"
                )
            else:
                action = (
                    f"⛔ **HALT SPRAYING IMMEDIATELY!**\n"
                    f"{city} has a **{rain_prob}% probability of rainfall** in the next 24 hours. Any foliar spray applied today will be completely washed away by rainwater, causing wasted input costs.\n\n"
                    f"🔍 **Recommended Action Plan:**\n"
                    f"1. Clear field drainage channels to prevent waterlogging around {crop} roots.\n"
                    f"2. Prepare a 5% Neem Seed Kernel Extract (NSKE) or cold-pressed Neem Oil (5ml/L) to spray once weather clears on day 2.\n"
                    f"3. Drench root zones with *Trichoderma viride* to resist soil-borne moisture pathogens."
                )
        elif humidity >= 80:
            # High humidity dominant
            if language == "Hindi":
                action = (
                    f"⚠️ **उच्च आर्द्रता (High Humidity) कीट चेतावनी!**\n"
                    f"वर्तमान में {city} में सुबह की नमी **{humidity}%** और तापमान **{temp}°C** है। यह मौसम ब्लाइट व फफूंद बीजाणुओं (fungal spores) के पनपने के लिए बेहद अनुकूल है (जोखिम: {risk_pct}%)।\n\n"
                    f"🌿 **जैविक रोकथाम कदम:**\n"
                    f"1. रोग के लक्षण दिखने से पहले ही **ट्राइकोडर्मा विरिडी (5 ग्राम/लीटर पानी)** का छिड़काव सुबह 7-9 बजे के बीच करें।\n"
                    f"2. यदि रस-चूसक कीट दिखें, तो 15-20 पीले चिपचिपे कार्ड (Yellow Sticky Traps) प्रति एकड़ लगाएं।\n"
                    f"3. हवा के संचार के लिए निचली रोगग्रस्त पत्तियों की छंटाई करें।"
                )
            else:
                action = (
                    f"⚠️ **ELEVATED HUMIDITY SPORE ALERT!**\n"
                    f"Morning relative humidity in {city} is at **{humidity}%** with a temperature of **{temp}°C**. This specific thermal-moisture band accelerates spore germination in {crop} (Outbreak Risk: {risk_pct}%).\n\n"
                    f"🌿 **Proactive Bio-Control Actions:**\n"
                    f"1. Apply prophylactic *Trichoderma viride* (5g/L water) or *Pseudomonas fluorescens* during early morning hours.\n"
                    f"2. Install 15–20 Yellow Sticky Traps per acre at canopy height to curb whitefly/aphid vectors.\n"
                    f"3. Prune bottom leaves to improve canopy aeration and limit soil splash."
                )
        else:
            # Stable dry window
            if language == "Hindi":
                action = (
                    f"✅ **अनुकूल छिड़काव खिड़की (Favorable Window)**\n"
                    f"{city} में आगामी 48 घंटों तक मौसम साफ व स्थिर बना हुआ है (तापमान {temp}°C, नमी {humidity}%)।\n\n"
                    f"🌱 **सुझाव:**\n"
                    f"1. यह नियमित जैविक पोषण (जीवामृत/पंचगव्य) देने का सबसे बेहतरीन समय है।\n"
                    f"2. फसल की नियमित निगरानी (Scouting) करें और मित्र कीटों (लेडीबर्ड बीटल) का संरक्षण करें।"
                )
            else:
                action = (
                    f"✅ **OPTIMAL SPRAYING WINDOW DETECTED**\n"
                    f"Atmospheric conditions across {city} are currently stable with moderate humidity ({humidity}%) and clear skies.\n\n"
                    f"🌱 **Agronomic Recommendation:**\n"
                    f"1. Ideal 48-hour window for foliar application of bio-fertilizers or organic plant growth promoters.\n"
                    f"2. Conduct weekly field scouting; protect beneficial predator insects (e.g. ladybird beetles, spiders)."
                )

        footer = (
            "\n\n🛡️ *सुरक्षा सलाह: रासायनिक कीटनाशकों का अंधाधुंध उपयोग न करें। मित्र कीटों और मिट्टी के सूक्ष्मजीवों की रक्षा करें।*"
            if language == "Hindi" else
            "\n\n🛡️ *Safety Guardrail: Avoid prophylactic synthetic chemicals. Safeguard beneficial pollinators and soil microbiome.*"
        )

        return f"{greeting}\n\n{action}{footer}"
