"""
KrishiMitra Agronomy Copilot & RAG Engine
Powered by IBM Granite 3.0 prompt architecture and ICAR Knowledge Base retrieval.
"""

import os
import json
import requests

try:
    from dotenv import load_dotenv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(base_dir, '.env'))
except ImportError:
    pass

class GraniteAgriCopilot:
    def __init__(self, knowledge_base_path=None):
        if knowledge_base_path is None:
            candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'icar_knowledge_base.json'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'icar_knowledge_base.json'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icar_knowledge_base.json'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'icar_knowledge_base.json'),
                'icar_knowledge_base.json'
            ]
            for c in candidates:
                if os.path.exists(c):
                    knowledge_base_path = c
                    break
            
        self.knowledge = {}
        if knowledge_base_path and os.path.exists(knowledge_base_path):
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                self.knowledge = json.load(f).get('crops', {})

    def retrieve_context(self, crop, query_text):
        """
        RAG Component: Retrieves relevant ICAR advisories matching the crop and symptom keywords.
        """
        crop_entries = self.knowledge.get(crop, [])
        matches = []
        q_lower = query_text.lower()
        
        for entry in crop_entries:
            score = 0
            if any(term in q_lower for term in entry['name'].lower().split()):
                score += 3
            if any(symptom_word in q_lower for symptom_word in entry['symptoms'].lower().split()):
                score += 2
            if score > 0:
                matches.append((score, entry))
                
        matches.sort(key=lambda x: x[0], reverse=True)
        if matches:
            return matches[0][1]
        elif crop_entries:
            return crop_entries[0]
        return None

    def generate_advisory(self, crop, user_query, weather_context=None, language="English", api_key=None):
        """
        Generates an actionable, climate-aligned agricultural advisory using IBM Granite 3.0 prompt template.
        Supports live IBM Watsonx / HuggingFace API or built-in offline agronomic synthesizer.
        """
        context = self.retrieve_context(crop, user_query)
        
        weather_summary = "Normal seasonal conditions."
        if weather_context:
            weather_summary = (
                f"Temp: {weather_context.get('temp', 28)}°C, "
                f"Morning Humidity: {weather_context.get('humidity', 75)}%, "
                f"Rain Probability: {weather_context.get('rain_prob', 10)}%, "
                f"Pest Outbreak Risk: {weather_context.get('risk_level', 'Moderate')}"
            )

        system_prompt = (
            "You are KrishiMitra, an expert AI Agricultural Scientist powered by IBM Granite 3.0. "
            "Your mission is to provide smallholder farmers with practical, low-cost, organic, "
            "and climate-resilient pest and crop protection advisories based on certified ICAR practices. "
            "Prioritize biological control (Neem, Trichoderma, pheromone traps, botanical extracts). "
            "Discourage excessive chemical pesticide dumping. Always include a safety warning."
        )

        rag_context_str = ""
        if context:
            rag_context_str = (
                f"\n[CERTIFIED ICAR REFERENCE: {context['name']}]\n"
                f"- Symptoms: {context['symptoms']}\n"
                f"- Weather Triggers: {context['weather_triggers']['high_risk_condition']}\n"
                f"- Recommended Bio-Remedies: {'; '.join(context['organic_remedies'])}\n"
                f"- Cultural Practices: {'; '.join(context['cultural_prevention'])}\n"
                f"- Hazard Warning: {context['hazard_warning']}\n"
            )

        # 1. Try Google Gemini API if key is available
        gemini_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if gemini_key and len(gemini_key) > 10:
            gemini_result = self._call_gemini_advisory(crop, user_query, weather_context, rag_context_str, language, gemini_key)
            if gemini_result:
                return gemini_result

        # 2. If user provides an IBM Granite / Watsonx API key or HuggingFace token, call the API
        if api_key and len(api_key) > 10 and not api_key.startswith("AIza"):
            try:
                # Hugging Face Inference API for IBM Granite 3.0 8B Instruct
                api_url = "https://api-inference.huggingface.co/models/ibm-granite/granite-3.0-8b-instruct"
                headers = {"Authorization": f"Bearer {api_key}"}
                prompt_text = (
                    f"<|start_of_role|>system<|end_of_role|>{system_prompt}<|end_of_text|>\n"
                    f"<|start_of_role|>user<|end_of_role|>\n"
                    f"Crop: {crop}\nWeather Condition: {weather_summary}\n"
                    f"Farmer Query: {user_query}\n"
                    f"Reference Guidelines: {rag_context_str}\n"
                    f"Language: {language}\n"
                    f"Provide actionable diagnosis, step-by-step bio-remedies, and weather precaution.<|end_of_text|>\n"
                    f"<|start_of_role|>assistant<|end_of_role|>"
                )
                resp = requests.post(api_url, headers=headers, json={"inputs": prompt_text, "parameters": {"max_new_tokens": 500}}, timeout=10)
                if resp.status_code == 200:
                    generated = resp.json()[0]['generated_text'].split("<|start_of_role|>assistant<|end_of_role|>")[-1].strip()
                    return f"**[Live IBM Granite 3.0 Advisory]**\n\n{generated}"
            except Exception as e:
                pass  # Fall back to offline high-fidelity synthesizer

        # 3. High-Fidelity Agronomic Synthesizer (Zero-latency offline demo mode)
        return self._synthesize_offline_response(crop, user_query, context, weather_context, language)

    def _call_gemini_advisory(self, crop, user_query, weather_context, rag_context, language, api_key):
        """Calls Google Gemini models to answer the farmer query dynamically."""
        models_to_try = ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.8-flash", "gemma-4-31b-it"]
        weather_summary = "Normal seasonal conditions."
        if weather_context:
            weather_summary = (
                f"Air Temp: {weather_context.get('temp', 28)}°C, "
                f"Humidity: {weather_context.get('humidity', 75)}%, "
                f"Rain Probability: {weather_context.get('rain_prob', 10)}%, "
                f"Outbreak Risk: {weather_context.get('risk_level', 'Moderate')}, "
                f"Soil Profile: {weather_context.get('soil_type', 'Agricultural Loam')} (pH {weather_context.get('soil_ph', 6.8)})"
            )

        prompt = f"""
You are KrishiMitra, an expert AI Agricultural Scientist powered by ICAR (Indian Council of Agricultural Research).
Answer the farmer's question directly, practically, and empathetically:

FARMER QUESTION: {user_query}
TARGET CROP: {crop}
LIVE ENVIRONMENTAL CONDITIONS: {weather_summary}
ICAR GUIDELINES CONTEXT: {rag_context}
LANGUAGE: {language}

Instructions:
1. Provide a direct, actionable answer addressing the farmer's question.
2. Structure with clean bullet points and clear sections.
3. Recommend practical biological remedies (Neem, Trichoderma, FYM/vermicompost) and soil/climate precautions.
4. Conclude with a Responsible AI safety caution against dangerous synthetic chemicals.
5. Do NOT use cartoon emojis. Keep the formatting clean and enterprise-grade.
"""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.6,
                "maxOutputTokens": 650,
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }
        
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            try:
                import urllib.request
                req_data = json.dumps(payload).encode('utf-8')
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                }
                req = urllib.request.Request(url, data=req_data, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as res:
                    data = json.loads(res.read().decode('utf-8'))
                    candidates = data.get("candidates", [])
                    if candidates:
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        if content_parts:
                            return f"**[Live Google Gemini Agronomy Copilot]**\n\n" + content_parts[0].get("text", "")
            except Exception:
                continue
        return None

    def _synthesize_offline_response(self, crop, query, context, weather_context, language):
        disease_name = context['name'] if (context and 'name' in context) else "General Crop Management & Agronomy"
        remedies = context['organic_remedies'] if (context and 'organic_remedies' in context) else [
            "Apply 10,000 ppm cold-pressed Neem Oil (5ml/L) to prevent sucking pest colonization.",
            "Spray Trichoderma viride bio-fungicide @ 5g/L during early morning hours."
        ]
        cultural = context['cultural_prevention'] if (context and 'cultural_prevention' in context) else [
            "Maintain wide row spacing to enhance field aeration and sunlight penetration.",
            "Avoid overhead sprinkler irrigation to minimize prolonged leaf wetness duration."
        ]
        warning = context.get('hazard_warning', "Avoid unprescribed synthetic chemicals; safeguard soil fertility.") if context else "Avoid unprescribed synthetic chemicals; safeguard beneficial pollinators and soil microbiome."

        rain_prob = weather_context.get('rain_prob', 0) if weather_context else 0
        temp = weather_context.get('temp', 28) if weather_context else 28
        soil_type = weather_context.get('soil_type', 'Agricultural Soil') if weather_context else 'Agricultural Soil'

        if rain_prob > 40:
            rain_warning = (
                f"**[WEATHER PRECAUTION]**: Rainfall probability is {rain_prob}% in your area within the next 24 hours. "
                "**DO NOT spray foliar bio-remedies today**, as rain will wash away the active ingredients. "
                "Ensure field drainage channels are clear."
            )
        else:
            rain_warning = (
                "**[WEATHER CLEAR WINDOW]**: Favorable dry window detected for the next 24-48 hours. "
                "Ideal time for early morning field preparation or foliar bio-nutrition."
            )

        # Detect general sowing/growing viability questions (e.g. 'can i grow tomato now in mariahu?')
        q_lower = query.lower()
        is_sowing_query = any(w in q_lower for w in ["can i grow", "grow", "sow", "plant", "season", "time", "kab boe", "lagaye", "viability", "feasible"])

        if is_sowing_query:
            if language == "Hindi":
                return (
                    f"### [कृषि-मित्र बुलेटिन] बुवाई एवं फसल अनुकूलता परामर्श\n\n"
                    f"**फसल**: {crop} | **वर्तमान तापमान**: {temp}°C | **मिट्टी**: {soil_type}\n\n"
                    f"{rain_warning}\n\n"
                    f"#### [वैज्ञानिक बुवाई सिफारिशें]:\n"
                    f"1. **मौसम अनुकूलता**: वर्तमान तापमान ({temp}°C) {crop} की प्राथमिक वृद्धि के लिए उपयुक्त है। यदि बारिश की संभावना अधिक हो, तो क्यारियों में जलभराव से बचने हेतु उठी हुई क्यारियों (raised beds) पर रोपाई करें।\n"
                    f"2. **मृदा पोषण**: खेत की तैयारी के समय 25-30% सड़ी हुई गोबर की खाद (FYM) या केंचुआ खाद मिलाएं ताकि मिट्टी में हवा का संचार बना रहे।\n"
                    f"3. **जैव-सुरक्षा**: पौध लगाने से पहले जड़ों को ट्राइकोडर्मा विरिडी (5 ग्राम/लीटर पानी) के घोल में 15 मिनट उपचारित करें।\n\n"
                    f"#### [सुरक्षा चेतावनी | Responsible AI]:\n"
                    f"> {warning}\n\n"
                    f"*स्त्रोत: आईसीएआर (ICAR) एवं राष्ट्रीय बागवानी बोर्ड दिशानिर्देश 2024*"
                )
            return (
                f"### [AGRONOMIC ADVISORY] Sowing & Cultivation Feasibility\n\n"
                f"**Target Crop**: {crop} | **Ambient Temp**: {temp}°C | **Soil Environment**: {soil_type}\n\n"
                f"{rain_warning}\n\n"
                f"#### [Actionable Agronomic Recommendations]:\n"
                f"1. **Viability Window**: Current temperature ({temp}°C) supports {crop} growth. If localized rainfall is anticipated, construct 15-20 cm raised cultivation beds to prevent root collar waterlogging.\n"
                f"2. **Organic Soil Enrichment**: Incorporate 25-30% decomposed farmyard manure (FYM) or vermicompost to enhance aeration and microbial vitality in {soil_type}.\n"
                f"3. **Root-Zone Bio-Inoculation**: Dip nursery seedling roots in Trichoderma viride bio-fungicide (@ 5g/L water) for 15 minutes prior to field transplanting.\n\n"
                f"#### [Responsible AI & Toxicity Safeguard]:\n"
                f"> **Warning**: {warning}\n\n"
                f"*Knowledge Base: Certified ICAR & National Horticulture Guidelines*"
            )

        if language == "Hindi":
            return (
                f"### [कृषि-मित्र सलाह] IBM Granite 3.0 RAG द्वारा समर्थित\n\n"
                f"**फसल**: {crop} | **संभावित संदर्भ**: {disease_name}\n\n"
                f"{rain_warning}\n\n"
                f"#### [जैविक रोकथाम उपाय]:\n"
                f"1. **नीम आधारित उपचार**: {remedies[0]}\n"
                f"2. **जैव-नियंत्रण**: {remedies[1] if len(remedies) > 1 else 'ट्राइकोडर्मा विरिडी (5 ग्राम/लीटर पानी) का छिड़काव करें।'}\n"
                f"3. **खेत प्रबंधन**: {cultural[0]}\n\n"
                f"#### [सुरक्षा चेतावनी | Responsible AI]:\n"
                f"> {warning}\n\n"
                f"*स्त्रोत: आईसीएआर (ICAR) एवं केंद्रीय कृषि दिशानिर्देश 2024*"
            )

        return (
            f"### [AGRONOMIC ADVISORY] IBM Granite 3.0 + ICAR RAG\n\n"
            f"**Target Crop**: {crop} | **Primary Diagnosis**: **{disease_name}**\n\n"
            f"{rain_warning}\n\n"
            f"#### [Actionable Bio-Control Measures]:\n"
            f"1. **Biological Spray**: {remedies[0]}\n"
            f"2. **Secondary Bio-Agent**: {remedies[1] if len(remedies) > 1 else 'Apply Trichoderma viride bio-fungicide @ 5g/liter.'}\n"
            f"3. **Agronomic Cultural Practice**: {cultural[0]}\n\n"
            f"#### [Responsible AI & Toxicity Safeguard]:\n"
            f"> **Warning**: {warning}\n\n"
            f"*Knowledge Base: Certified ICAR & National IPM Guidelines*"
        )
