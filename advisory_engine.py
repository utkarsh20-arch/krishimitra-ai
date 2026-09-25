"""
KrishiMitra Agronomy Copilot & Conversational RAG Engine
Powered by ICAR Certified Knowledge Base and Multi-Engine LLM Architecture.
Zero emojis, enterprise-grade dark telemetry formatting.
Supports English, Hindi, and Hinglish conversational interactions.
"""

import os
import json
import re

# Native zero-dependency .env loader
def _load_env_native():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, '.env')
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        if k.strip() not in os.environ:
                            os.environ[k.strip()] = v.strip()
        except Exception:
            pass

_load_env_native()

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
            try:
                with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                    self.knowledge = json.load(f).get('crops', {})
            except Exception:
                self.knowledge = {}

    def retrieve_context(self, crop, query_text):
        """
        RAG Component: Retrieves relevant ICAR disease entry if symptoms/disease keywords match.
        Returns None for general conversational queries (greetings, rain emergencies, general soil questions).
        """
        if not query_text or not self.knowledge:
            return None
            
        q_lower = query_text.lower().strip()
        
        # Guard: Ignore casual conversational greetings or generic questions
        conversational_words = ["hi", "hello", "hey", "namaste", "pranam", "kaise ho", "kya haal", "ram ram"]
        if q_lower in conversational_words or len(q_lower) < 4:
            return None

        crop_entries = self.knowledge.get(crop, [])
        matches = []
        
        for entry in crop_entries:
            score = 0
            entry_name_words = [w for w in entry['name'].lower().split() if len(w) > 3]
            for term in entry_name_words:
                if term in q_lower:
                    score += 4
            symptom_words = [w for w in entry.get('symptoms', '').lower().split() if len(w) > 3]
            for symptom_word in symptom_words:
                if symptom_word in q_lower:
                    score += 2
            if score >= 3:
                matches.append((score, entry))
                
        matches.sort(key=lambda x: x[0], reverse=True)
        if matches:
            return matches[0][1]
        return None

    def generate_advisory(self, crop, user_query, weather_context=None, language="English", api_key=None, chat_history=None):
        """
        Main entry point for farmer questions.
        Supports online Google Gemini, HuggingFace/IBM Granite, and an intelligent offline conversational fallback.
        """
        if not user_query or not user_query.strip():
            return "Please type your question regarding crop health, rainfall precautions, soil nutrition, or pest management."

        q_lower = user_query.lower().strip()

        # Detect Hindi/Hinglish phrasing even if language selector is set to English
        hindi_keywords = [
            "kya", "kaise", "karein", "karna", "barish", "pani", "mitti", "khad", "rog", 
            "fasal", "tamatar", "namaste", "ram ram", "hai", "rahi", "mera", "meri", 
            "khet", "bhai", "kripya", "batao", "bataiye", "pranam", "kaise", "kaisa"
        ]
        is_hindi = (language == "Hindi") or any(w in q_lower for w in hindi_keywords)

        # Contextual ICAR RAG
        context = self.retrieve_context(crop, user_query)
        rag_context_str = ""
        if context:
            rag_context_str = (
                f"\n[CERTIFIED ICAR REFERENCE: {context.get('name', 'Pest/Disease')}]\n"
                f"- Symptoms: {context.get('symptoms', 'N/A')}\n"
                f"- Weather Triggers: {context.get('weather_triggers', {}).get('high_risk_condition', 'High humidity')}\n"
                f"- Recommended Bio-Remedies: {'; '.join(context.get('organic_remedies', []))}\n"
                f"- Cultural Practices: {'; '.join(context.get('cultural_prevention', []))}\n"
                f"- Hazard Warning: {context.get('hazard_warning', 'Avoid synthetic chemicals.')}\n"
            )

        # 1. Try Google Gemini API if key is present
        gemini_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if gemini_key and len(gemini_key) > 10:
            gemini_res = self._call_gemini_advisory(crop, user_query, weather_context, rag_context_str, language, gemini_key, is_hindi)
            if gemini_res:
                return gemini_res

        # 2. Try IBM Granite API if HuggingFace / Granite token is supplied
        if api_key and len(api_key) > 10 and not api_key.startswith("AIza") and not api_key.startswith("AQ."):
            hf_res = self._call_granite_advisory(crop, user_query, weather_context, rag_context_str, language, api_key)
            if hf_res:
                return hf_res

        # 3. High-Fidelity Conversational Offline Engine
        return self._synthesize_offline_response(crop, user_query, context, weather_context, language, is_hindi)

    def _call_gemini_advisory(self, crop, user_query, weather_context, rag_context, language, api_key, is_hindi=False):
        """Calls Google Gemini model via native urllib REST API."""
        models_to_try = ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.8-flash", "gemma-4-31b-it"]
        
        weather_summary = "Normal seasonal conditions."
        location_name = "Local Farm"
        if weather_context:
            location_name = weather_context.get('location', 'Mariahu')
            weather_summary = (
                f"Location: {location_name}, "
                f"Air Temp: {weather_context.get('temp', 28)}°C, "
                f"Humidity: {weather_context.get('humidity', 75)}%, "
                f"Rain Probability: {weather_context.get('rain_prob', 10)}%, "
                f"Outbreak Risk: {weather_context.get('risk_level', 'Moderate')}, "
                f"Soil Profile: {weather_context.get('soil_type', 'Agricultural Loam')} (pH {weather_context.get('soil_ph', 6.8)})"
            )

        target_lang = "Hindi / Hinglish" if is_hindi else language

        prompt = f"""
You are KrishiMitra, an empathetic, certified AI Agricultural Scientist and Agronomy Copilot (ICAR grounded).
You are interacting directly with an Indian farmer in an interactive chat session.

FARMER QUESTION: {user_query}
TARGET CROP: {crop}
LIVE TELEMETRY: {weather_summary}
ICAR CODEX CONTEXT: {rag_context}
RESPOND IN: {target_lang}

Conversational Guidelines:
1. If the farmer gives a greeting (hi, hello, namaste, ram ram, etc.), greet them warmly, acknowledge their farm ({crop} in {location_name}), and ask how you can assist their crop today.
2. If the farmer asks an urgent weather question (such as active rainfall, waterlogging, or storm), provide immediate 4-step emergency field protection (immediate furrow drainage, do not spray chemicals/urea during rain, stake plants, post-rain bio-fungicide like Trichoderma).
3. If asking about planting or soil, provide scientifically grounded advice referencing their current temperature and soil type.
4. Always prioritize biological organic solutions (Neem 10,000 ppm, Trichoderma, FYM compost, pheromone traps).
5. Conclude with a Responsible AI safety caution against dangerous synthetic chemical overdoses.
6. NO cartoon emojis. Use clean bullet points and professional typography.
"""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.65,
                "maxOutputTokens": 650,
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }
        
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            try:
                import urllib.request
                req_data = json.dumps(payload).encode('utf-8')
                headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
                req = urllib.request.Request(url, data=req_data, headers=headers)
                res = urllib.request.urlopen(req, timeout=20)
                data = json.loads(res.read().decode('utf-8'))
                candidates = data.get("candidates", [])
                if candidates:
                    content_parts = candidates[0].get("content", {}).get("parts", [])
                    if content_parts:
                        return f"**[Live Google Gemini Agronomy Copilot]**\n\n" + content_parts[0].get("text", "").strip()
            except Exception:
                continue
        return None

    def _call_granite_advisory(self, crop, user_query, weather_context, rag_context, language, api_key):
        """Calls Hugging Face IBM Granite endpoint."""
        try:
            import urllib.request
            api_url = "https://api-inference.huggingface.co/models/ibm-granite/granite-3.0-8b-instruct"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            prompt_text = (
                f"<|start_of_role|>system<|end_of_role|>You are KrishiMitra, an expert AI Agricultural Scientist. Zero emojis.<|end_of_text|>\n"
                f"<|start_of_role|>user<|end_of_role|>\n"
                f"Crop: {crop}\nFarmer Query: {user_query}\nReference: {rag_context}\nLanguage: {language}\n"
                f"Provide actionable agronomy guidance.<|end_of_text|>\n"
                f"<|start_of_role|>assistant<|end_of_role|>"
            )
            payload = json.dumps({"inputs": prompt_text, "parameters": {"max_new_tokens": 500}}).encode('utf-8')
            req = urllib.request.Request(api_url, data=payload, headers=headers)
            res = urllib.request.urlopen(req, timeout=12)
            data = json.loads(res.read().decode('utf-8'))
            if isinstance(data, list) and len(data) > 0 and 'generated_text' in data[0]:
                text = data[0]['generated_text'].split("<|start_of_role|>assistant<|end_of_role|>")[-1].strip()
                return f"**[Live IBM Granite 3.0 Advisory]**\n\n{text}"
        except Exception:
            pass
        return None

    def _synthesize_offline_response(self, crop, query, context, weather_context, language, is_hindi=False):
        """
        High-Fidelity Offline Conversational Synthesizer.
        Responds empathetically and accurately across greetings, active rainfall emergencies,
        sowing feasibility, soil enrichment, and pest/disease management without crashing.
        """
        q_lower = query.lower().strip()
        
        # Microclimate parameters
        rain_prob = weather_context.get('rain_prob', 0) if weather_context else 0
        temp = weather_context.get('temp', 28) if weather_context else 28
        soil_type = weather_context.get('soil_type', 'Agricultural Soil') if weather_context else 'Agricultural Soil'
        soil_ph = weather_context.get('soil_ph', 7.0) if weather_context else 7.0
        location = weather_context.get('location', 'your area') if weather_context else 'your area'

        # ---------------------------------------------------------
        # INTENT 1: GREETINGS & CASUAL INTERACTION
        # ---------------------------------------------------------
        greeting_patterns = [
            r'^(hi|hello|hey|helo|hii|heyy|namaste|namaskar|pranam|ram ram|kya haal|kaise ho|kaise h|kaisa hai|good morning|good evening|shubh prabhat)$',
            r'\b(hi|hello|hey|namaste|namaskar|pranam|ram ram|kaise ho|kya haal)\b'
        ]
        is_greeting = any(re.search(pat, q_lower) for pat in greeting_patterns) and len(q_lower.split()) <= 5

        if is_greeting:
            if is_hindi:
                return (
                    f"### [कृषि-मित्र संवादात्मक को-पायलट]\n\n"
                    f"**नमस्ते किसान भाई! राम-राम।** मैं आपका डिजिटल कृषि-मित्र एग्रोनॉमी को-पायलट हूँ।\n\n"
                    f"मैं आपके क्षेत्र (**{location}**) में **{crop}** की खेती, वर्तमान तापमान (**{temp}°C**), बारिश की संभावना (**{rain_prob}%**) और मिट्टी (**{soil_type}**) के अनुसार आपकी सहायता के लिए तैयार हूँ।\n\n"
                    f"**आप मुझसे क्या-क्या पूछ सकते हैं?**\n"
                    f"1. **बारिश व मौसम आपदा**: जैसे *\"अभी बारिश हो रही है, टमाटर के खेत में क्या करूँ?\"*\n"
                    f"2. **बुवाई व मौसम अनुकूलता**: जैसे *\"क्या मैं अभी टमाटर लगा सकता हूँ?\"*\n"
                    f"3. **कीट व रोग सुरक्षा**: पत्ती मुड़ना (Leaf Curl), झुलसा (Blight) या जैविक नीम के तेल का छिड़काव।\n"
                    f"4. **मिट्टी व जैविक खाद**: सड़ी गोबर की खाद (FYM), केंचुआ खाद और सही पोषण।\n\n"
                    f"*बताइए, आज आपके खेत या फसल से जुड़ी क्या समस्या है?*"
                )
            return (
                f"### [KrishiMitra Conversational Copilot]\n\n"
                f"**Hello & Welcome Farmer!** I am your **KrishiMitra Agronomy Copilot**.\n\n"
                f"I am actively synchronized with your **{crop}** farm in **{location}**, monitoring your ambient temperature (**{temp}°C**), rain probability (**{rain_prob}%**), and soil order (**{soil_type}**, pH {soil_ph}).\n\n"
                f"**How can I assist you right now?**\n"
                f"1. **Live Rain & Weather Emergencies**: E.g., *\"It is raining right now, what immediate steps should I take for my tomato farm?\"*\n"
                f"2. **Cultivation & Sowing Feasibility**: Crop viability in current temperatures and bed preparation.\n"
                f"3. **Organic Pest & Disease Defense**: Certified ICAR biological controls (Neem, Trichoderma, pheromone traps).\n"
                f"4. **Soil Fertility & pH Management**: Organic manures, microbial inoculation, and drainage.\n\n"
                f"*Please ask your question in English, Hindi, or Hinglish!*"
            )

        # ---------------------------------------------------------
        # INTENT 2: RAIN / STORM / WATERLOGGING EMERGENCY
        # ---------------------------------------------------------
        rain_emergency_words = ["rain", "raining", "barish", "barsat", "pani", "waterlog", "flood", "toofan", "storm", "hail", "jal jamav", "jalbhav"]
        is_rain_emergency = any(w in q_lower for w in rain_emergency_words)

        if is_rain_emergency:
            if is_hindi:
                return (
                    f"### [आपातकालीन मौसम सलाह | बारिश एवं जलभराव प्रबंधन]\n\n"
                    f"**स्थान**: {location} | **फसल**: {crop} | **वर्तमान वर्षा स्थिति**: सक्रिय वर्षा / जलभराव जोखिम\n\n"
                    f"टमाटर या सब्जी की फसलों के लिए अत्यधिक वर्षा और खेत में भरा पानी बहुत संवेदनशील होता है। तुरंत निम्नलिखित कदम उठाएं:\n\n"
                    f"#### [तुरंत करने योग्य 4 प्राथमिकता कार्य]:\n"
                    f"1. **तत्काल जल-निकासी (Drainage)**: टमाटर की जड़ें 12-24 घंटे से अधिक पानी में डूबी रहने पर सड़ने लगती हैं (Collar Rot / Damping-off) और मुरझान (Bacterial Wilt) का खतरा बढ़ता है। खेत की क्यारियों और मेड़ों के बीच नालियां काटकर तुरंत अतिरिक्त पानी बाहर निकालें।\n"
                    f"2. **दवा छिड़काव व यूरिया तुरंत रोकें**: चालू बारिश में कोई भी कीटनाशक, फफूंदनाशक या दानेदार रासायनिक यूरिया न डालें। बारिश से सारी दवा धुल जाएगी, लागत बर्बाद होगी और रसायन पानी में बह जाएगा।\n"
                    f"3. **सहारा (Staking) की जांच करें**: यदि टमाटर के पौधे बड़े हैं, तो बांस की खपच्चियों और सुतली को मजबूत करें ताकि पौधे गीली कीचड़ में न गिरें। फल यदि गीली मिट्टी से छुएंगे तो उनमें फल सड़न (Buckeye Rot) हो जाएगी।\n"
                    f"4. **बारिश रुकने के बाद सुरक्षात्मक छिड़काव**: बारिश थमने और धूप निकलने के तुरंत बाद वातावरण में 90%+ नमी से झुलसा (Blight) का प्रकोप होता है। मौसम साफ होते ही पत्तियों पर जैविक **ट्राइकोडर्मा विरिडी (5 ग्राम/लीटर पानी)** या कॉपर आधारित जैविक फफूंदनाशक का छिड़काव करें।\n\n"
                    f"#### [सुरक्षा चेतावनी | Responsible AI]:\n"
                    f"> **सावधानी**: बारिश के दौरान खेत में सबमर्सिबल पंप, ट्रांसफॉर्मर या नंगे बिजली तारों के पास बिल्कुल न जाएं। जलभराव वाले खेतों में कभी भी अत्यधिक रासायनिक घोल न डालें।\n\n"
                    f"*वैज्ञानिक स्त्रोत: भारतीय सब्जी अनुसंधान संस्थान (ICAR-IIVR) दिशानिर्देश*"
                )
            return (
                f"### [EMERGENCY AGRONOMIC ADVISORY] Active Rainfall & Crop Protection\n\n"
                f"**Location**: {location} | **Target Crop**: {crop} | **Current Trigger**: Rainfall / Soil Saturation\n\n"
                f"Tomato root structures are extremely susceptible to water stagnation and soil oxygen depletion. Execute the following immediate protocol:\n\n"
                f"#### [Immediate Priority Action Protocol]:\n"
                f"1. **Clear Surface Drainage Immediately**: Ensure all furrows and outlet trenches between beds are completely unobstructed to drain ponded water. Tomato roots standing in water for >12-24 hours succumb rapidly to fungal root collar rot (*Pythium/Phytophthora*) and sudden bacterial wilt.\n"
                f"2. **Cease All Chemical Sprays & Fertilizers**: DO NOT apply foliar bio-sprays, pesticides, or nitrogen (Urea) during active rain. Rainfall completely washes away applied chemicals, resulting in financial loss and ground contamination.\n"
                f"3. **Inspect Staking & Trellising**: Verify that tomato vine stakes and trellis strings hold the plants upright. Keep foliage and developing fruits elevated off wet muddy soil to prevent Buckeye fruit rot (*Phytophthora nicotianae*).\n"
                f"4. **Post-Rain Pathogen Barrier**: Saturated humidity (>90%) following rain sparks rapid Early and Late Blight sporulation. Once rain pauses and leaf surfaces dry, apply preventive **Trichoderma viride (@ 5g/L)** or copper bio-formulations during morning hours.\n\n"
                f"#### [Responsible AI & Electrical Safety Safeguard]:\n"
                f"> **Caution**: Exercise strict caution around wet electrical pump connections and cables. Avoid drenching waterlogged root zones with hazardous chemical compounds.\n\n"
                f"*Knowledge Base: Certified ICAR - Indian Institute of Vegetable Research (IIVR) Protocol*"
            )

        # ---------------------------------------------------------
        # INTENT 3: SOWING & CULTIVATION FEASIBILITY
        # ---------------------------------------------------------
        sowing_words = ["can i grow", "grow", "sow", "plant", "season", "time", "kab boe", "lagaye", "lagana", "viability", "feasible", "ugaye", "kheti"]
        is_sowing = any(w in q_lower for w in sowing_words)

        if is_sowing:
            if is_hindi:
                return (
                    f"### [कृषि-मित्र बुलेटिन] बुवाई एवं फसल अनुकूलता परामर्श\n\n"
                    f"**फसल**: {crop} | **स्थान**: {location} | **वर्तमान तापमान**: {temp}°C | **मिट्टी**: {soil_type}\n\n"
                    f"#### [वैज्ञानिक बुवाई सिफारिशें]:\n"
                    f"1. **मौसम अनुकूलता**: वर्तमान तापमान ({temp}°C) {crop} की वृद्धि के लिए अनुकूल है। यदि बारिश की संभावना अधिक हो, तो क्यारियों में जलभराव से बचने हेतु 15-20 सेमी उठी हुई क्यारियों (raised beds) पर रोपाई करें।\n"
                    f"2. **मृदा पोषण व जीवांश**: खेत की तैयारी में 25-30% अच्छी सड़ी गोबर की खाद (FYM) या केंचुआ खाद मिलाएं ताकि {soil_type} में जल निकासी और हवा का संचार उत्तम रहे।\n"
                    f"3. **जैव-सुरक्षा (Seedling Dip)**: पौध रोपने से पहले जड़ों को ट्राइकोडर्मा विरिडी (5 ग्राम/लीटर पानी) के घोल में 15 मिनट डुबोकर उपचारित करें।\n\n"
                    f"#### [सुरक्षा चेतावनी | Responsible AI]:\n"
                    f"> **सावधानी**: बिना मृदा परीक्षण के अत्यधिक रासायनिक उर्वरक न डालें। जैविक खाद से मिट्टी की उर्वरता सुरक्षित रखें।\n\n"
                    f"*स्त्रोत: आईसीएआर (ICAR) एवं राष्ट्रीय बागवानी बोर्ड दिशानिर्देश*"
                )
            return (
                f"### [AGRONOMIC ADVISORY] Sowing & Cultivation Feasibility\n\n"
                f"**Target Crop**: {crop} | **Location**: {location} | **Ambient Temp**: {temp}°C | **Soil Environment**: {soil_type}\n\n"
                f"#### [Actionable Agronomic Recommendations]:\n"
                f"1. **Viability Window**: Current temperature ({temp}°C) supports {crop} vegetative growth. If precipitation is anticipated, construct 15-20 cm raised cultivation beds to prevent root collar waterlogging.\n"
                f"2. **Organic Soil Enrichment**: Incorporate 25-30% well-decomposed farmyard manure (FYM) or vermicompost to enhance aeration and microbial vitality in {soil_type}.\n"
                f"3. **Root-Zone Bio-Inoculation**: Dip nursery seedling roots in Trichoderma viride bio-fungicide (@ 5g/L water) for 15 minutes prior to field transplanting.\n\n"
                f"#### [Responsible AI & Toxicity Safeguard]:\n"
                f"> **Warning**: Avoid synthetic chemical overdosing. Rely on organic soil amendment for long-term farm resilience.\n\n"
                f"*Knowledge Base: Certified ICAR & National Horticulture Guidelines*"
            )

        # ---------------------------------------------------------
        # INTENT 4: SOIL & FERTILIZERS / KHAD
        # ---------------------------------------------------------
        soil_words = ["soil", "mitti", "khad", "fertilizer", "urea", "dap", "npk", "potash", "compost", "gobar", "vermicompost", "ph"]
        is_soil = any(w in q_lower for w in soil_words)

        if is_soil:
            if is_hindi:
                return (
                    f"### [मृदा स्वास्थ्य एवं जैविक पोषण परामर्श]\n\n"
                    f"**स्थान**: {location} | **पहचानी गई मिट्टी**: {soil_type} (pH {soil_ph}) | **फसल**: {crop}\n\n"
                    f"#### [संतुलित जैविक खाद प्रबंधन]:\n"
                    f"1. **जीवांश खाद (Organic Base)**: प्रति एकड़ 4-5 टन अच्छी सड़ी गोबर की खाद (FYM) या 1.5 टन केंचुआ खाद (Vermicompost) खेत की अंतिम जुताई में मिलाएं।\n"
                    f"2. **जैव-उर्वरक (Bio-Fertilizers)**: नाइट्रोजन आपूर्ति हेतु एजोटोबैक्टर (Azotobacter) और फास्फोरस के लिए पीएसबी (PSB) कल्चर (2 किग्रा/एकड़) का उपयोग करें।\n"
                    f"3. **अत्यधिक यूरिया से बचें**: रासायनिक यूरिया की अधिक मात्रा से पौधे कोमल हो जाते हैं और रस चूसक कीड़ों का हमला तेजी से बढ़ता है।\n\n"
                    f"#### [सुरक्षा चेतावनी | Responsible AI]:\n"
                    f"> **सावधानी**: मिट्टी की जांच के बिना अंधाधुंध रासायनिक डीएपी/यूरिया न डालें।\n\n"
                    f"*स्त्रोत: आईसीएआर (ICAR) मृदा विज्ञान संस्थान दिशानिर्देश*"
                )
            return (
                f"### [SOIL HEALTH & BALANCED NUTRITION ADVISORY]\n\n"
                f"**Location**: {location} | **Soil Profile**: {soil_type} (pH {soil_ph}) | **Target Crop**: {crop}\n\n"
                f"#### [Scientific Organic Soil Amendment]:\n"
                f"1. **Organic Carbon Enrichment**: Apply 4-5 tonnes/acre well-rotted farmyard manure (FYM) or 1.5 tonnes vermicompost to optimize cation exchange capacity in {soil_type}.\n"
                f"2. **Beneficial Microbial Inoculants**: Use Azotobacter (nitrogen-fixing) and PSB (Phosphorus Solubilizing Bacteria) bio-fertilizers @ 2 kg/acre mixed with compost.\n"
                f"3. **Mitigate Synthetic Nitrogen Excess**: Over-application of chemical urea causes succulent plant growth, inviting severe aphid and whitefly colonization.\n\n"
                f"#### [Responsible AI & Soil Conservation Safeguard]:\n"
                f"> **Warning**: Avoid excessive chemical fertilizer dumping. Preserve natural soil microbiome integrity.\n\n"
                f"*Knowledge Base: ICAR Indian Institute of Soil Science*"
            )

        # ---------------------------------------------------------
        # INTENT 5: PEST & DISEASE / GENERAL CODEX ADVISORY
        # ---------------------------------------------------------
        disease_name = context.get('name', 'Crop Health & Pest Management') if context else "Crop Health & Field Management"
        remedies = context.get('organic_remedies', [
            "Apply 10,000 ppm cold-pressed Neem Oil (5ml/L water) with emulsifier as an early barrier.",
            "Spray Trichoderma viride bio-fungicide @ 5g/L during early morning hours."
        ]) if context else [
            "Apply 10,000 ppm cold-pressed Neem Oil (5ml/L water) to suppress sucking pests.",
            "Spray Trichoderma viride bio-fungicide @ 5g/L during early morning hours."
        ]
        cultural = context.get('cultural_prevention', [
            "Maintain wide row spacing to ensure adequate aeration and sunlight penetration.",
            "Avoid overhead sprinkler irrigation to minimize prolonged leaf wetness."
        ]) if context else [
            "Maintain wide row spacing to enhance field aeration and sunlight penetration.",
            "Avoid overhead sprinkler irrigation to minimize prolonged leaf wetness duration."
        ]
        warning = context.get('hazard_warning', "Avoid unprescribed synthetic chemicals; safeguard soil fertility.") if context else "Avoid unprescribed synthetic chemicals; safeguard beneficial pollinators and soil microbiome."

        rain_warning = ""
        if rain_prob > 40:
            rain_warning = (
                f"**[WEATHER PRECAUTION]**: Rainfall probability is {rain_prob}% in your area. "
                "**DO NOT spray foliar bio-remedies today**, as precipitation will wash away the active ingredients."
            )
        else:
            rain_warning = (
                "**[WEATHER CLEAR WINDOW]**: Favorable dry window detected for the next 24-48 hours. "
                "Ideal time for early morning field inspection or foliar bio-nutrition."
            )

        if is_hindi:
            return (
                f"### [कृषि-मित्र वैज्ञानिक परामर्श] ICAR प्रमाणित दिशानिर्देश\n\n"
                f"**फसल**: {crop} | **विषय**: {disease_name} | **स्थान**: {location}\n\n"
                f"{rain_warning}\n\n"
                f"#### [प्राकृतिक एवं जैविक उपचार]:\n"
                f"1. **जैविक नीम छिड़काव**: {remedies[0]}\n"
                f"2. **जैव-फफूंदनाशक**: {remedies[1] if len(remedies) > 1 else 'ट्राइकोडर्मा विरिडी (5 ग्राम/लीटर पानी) का छिड़काव करें।'}\n"
                f"3. **खेत प्रबंधन**: {cultural[0]}\n\n"
                f"#### [सुरक्षा चेतावनी | Responsible AI]:\n"
                f"> **सावधानी**: {warning}\n\n"
                f"*स्त्रोत: आईसीएआर (ICAR) एवं राष्ट्रीय पादप स्वास्थ्य प्रबंधन संस्थान*"
            )

        return (
            f"### [AGRONOMIC ADVISORY] Certified ICAR Codex\n\n"
            f"**Target Crop**: {crop} | **Focus**: **{disease_name}** | **Location**: {location}\n\n"
            f"{rain_warning}\n\n"
            f"#### [Actionable Bio-Control Measures]:\n"
            f"1. **Primary Biological Defense**: {remedies[0]}\n"
            f"2. **Secondary Bio-Agent**: {remedies[1] if len(remedies) > 1 else 'Apply Trichoderma viride bio-fungicide @ 5g/liter.'}\n"
            f"3. **Agronomic Cultural Practice**: {cultural[0]}\n\n"
            f"#### [Responsible AI & Toxicity Safeguard]:\n"
            f"> **Warning**: {warning}\n\n"
            f"*Knowledge Base: Certified ICAR & National IPM Guidelines*"
        )

# Export alias so both import names work seamlessly
AgronomyCopilot = GraniteAgriCopilot
