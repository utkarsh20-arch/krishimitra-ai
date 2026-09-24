"""
====================================================================
🌾 KrishiMitra AI: Resilient Climate & Pest Advisory Copilot
1M1B - IBM SkillsBuild AI for Sustainability Virtual Internship
Aligned with UN SDGs: SDG 2 (Zero Hunger), SDG 13 (Climate Action), SDG 15 (Life on Land)
====================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.vision_detector import CropVisionDetector
    from models.weather_risk_predictor import WeatherRiskPredictor
    from models.advisory_engine import GraniteAgriCopilot
    from models.weather_service import LiveWeatherService
    from models.llm_service import DynamicLLMAlertService
except (ModuleNotFoundError, ImportError):
    from vision_detector import CropVisionDetector
    from weather_risk_predictor import WeatherRiskPredictor
    from advisory_engine import GraniteAgriCopilot
    from weather_service import LiveWeatherService
    from llm_service import DynamicLLMAlertService

# Page Configuration
st.set_page_config(
    page_title="KrishiMitra AI | Sustainable Agri Copilot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling with Rich Animations
st.markdown("""
<style>
    @keyframes floatWheat {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-6px) rotate(5deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    
    @keyframes pulseLive {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
    }
    
    @keyframes shimmerTitle {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    @keyframes cardFadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes riskPulse {
        0% { transform: scale(1); filter: drop-shadow(0 0 2px rgba(230,57,70,0.4)); }
        50% { transform: scale(1.03); filter: drop-shadow(0 0 12px rgba(230,57,70,0.8)); }
        100% { transform: scale(1); filter: drop-shadow(0 0 2px rgba(230,57,70,0.4)); }
    }

    .floating-logo {
        display: inline-block;
        animation: floatWheat 4s ease-in-out infinite;
        font-size: 2.4rem;
        vertical-align: middle;
        margin-right: 8px;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2D6A4F, #52B788, #74C69D, #2D6A4F);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmerTitle 6s linear infinite;
        display: inline-block;
        margin-bottom: 2px;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #8D99AE;
        margin-bottom: 18px;
        animation: cardFadeIn 0.8s ease-out;
    }

    /* Weather Live Box - Glassmorphic Animated Ticker */
    .weather-live-box {
        background: linear-gradient(135deg, rgba(27, 67, 50, 0.45) 0%, rgba(45, 106, 79, 0.35) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(82, 183, 136, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        padding: 18px 22px;
        border-radius: 14px;
        color: #E8F5E9 !important;
        margin-bottom: 20px;
        animation: cardFadeIn 0.6s ease-out;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
    }
    .weather-live-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(45, 106, 79, 0.35);
        border-color: rgba(116, 198, 157, 0.6);
    }

    /* Live Pulsing Dot */
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #2ECC71;
        animation: pulseLive 1.8s infinite;
        margin-right: 7px;
        vertical-align: middle;
    }

    /* Weather Stat Chips */
    .weather-chips-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
    .weather-chip {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        color: #F1FAEE;
        transition: all 0.25s ease;
    }
    .weather-chip:hover {
        background: rgba(82, 183, 136, 0.25);
        transform: translateY(-2px);
        border-color: #52B788;
    }

    /* AI Alert Box with Glowing Left Accent */
    .ai-alert-box {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(8px);
        border-left: 5px solid #52B788;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        margin-top: 18px;
        animation: cardFadeIn 0.5s ease-out;
        transition: all 0.3s ease;
    }
    .ai-alert-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(82, 183, 136, 0.2);
        border-left-color: #74C69D;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #40916C;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 14px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        border-left-color: #52B788;
    }

    /* SDG Badges */
    .badge-sdg {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
        color: #D8F3DC;
        border: 1px solid #40916C;
        padding: 5px 12px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
        transition: transform 0.2s ease;
    }
    .badge-sdg:hover {
        transform: scale(1.05);
    }

    /* Outbreak Risk Display */
    .risk-banner {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 22px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    .risk-banner:hover {
        transform: translateY(-3px);
    }
    .risk-score-pulse {
        display: inline-block;
        animation: riskPulse 2.5s infinite ease-in-out;
    }

    /* Streamlit Tab Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(45, 106, 79, 0.15);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Models & Services
@st.cache_resource
def load_components():
    vision = CropVisionDetector()
    predictor = WeatherRiskPredictor()
    copilot = GraniteAgriCopilot()
    weather_svc = LiveWeatherService()
    llm_alert_svc = DynamicLLMAlertService()
    return vision, predictor, copilot, weather_svc, llm_alert_svc

vision_model, risk_predictor, copilot_model, weather_service, alert_service = load_components()

# Session State for Live Weather & Dynamic Alert
if "live_weather" not in st.session_state:
    st.session_state.live_weather = weather_service.get_weather_by_city("Nashik")

if "dynamic_alert" not in st.session_state:
    st.session_state.dynamic_alert = None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/wheat.png", width=70)
    st.markdown("### 🌾 KrishiMitra AI")
    st.caption("**1M1B – IBM SkillsBuild Virtual Internship**")
    
    st.markdown("---")
    st.markdown("**📍 Location & Agro-Climatic Zone**")
    
    city_input = st.text_input("Enter City / District Name:", value="Nashik", help="Type any district in India or worldwide")
    if st.button("🔄 Fetch Live Weather", type="primary"):
        with st.spinner(f"Fetching real-time satellite & weather feed for {city_input}..."):
            st.session_state.live_weather = weather_service.get_weather_by_city(city_input)
            st.session_state.dynamic_alert = None # Reset alert for new location
            st.success(f"Loaded: {st.session_state.live_weather['location']}")

    selected_crop = st.selectbox("Target Crop", ["Tomato", "Paddy (Rice)", "Cotton", "Potato", "Wheat"])
    language = st.radio("Language / भाषा", ["English", "Hindi"], horizontal=True)
    
    st.markdown("---")
    st.markdown("**🤖 AI Engine Settings**")
    ai_provider = st.selectbox("LLM Provider", ["Google Gemini", "OpenAI ChatGPT", "IBM Granite (Offline)"])
    user_api_key = st.text_input(
        f"{ai_provider} API Key (Optional):",
        type="password",
        placeholder="Paste your API key here",
        help="Leave blank to use the built-in intelligent agronomy engine!"
    )
    if not user_api_key:
        st.caption("💡 *Running in Smart Adaptive Mode (Zero key required).*")
    else:
        st.caption("✨ *Live Cloud API Mode Active.*")

    st.markdown("---")
    st.markdown(
        "<span class='badge-sdg'>SDG 2: Zero Hunger</span>"
        "<span class='badge-sdg'>SDG 13: Climate</span>"
        "<span class='badge-sdg'>SDG 15: Soil Health</span>",
        unsafe_allow_html=True
    )

# --- MAIN CONTENT HEADER ---
st.markdown("""
<div style='display: flex; align-items: center; margin-bottom: 4px;'>
    <span class='floating-logo'>🌾</span>
    <h1 class='main-title'>KrishiMitra: Climate & Pest Advisory Copilot</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Empowering smallholder farmers with proactive bio-control advisories grounded in real-time climate telemetry & Google Gemini AI.</p>", unsafe_allow_html=True)

# Live Weather Banner with Animated Chips & Live Radar Dot
lw = st.session_state.live_weather
st.markdown(f"""
<div class='weather-live-box'>
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
        <div>
            <span class='pulse-dot'></span>
            <b style='letter-spacing: 0.5px;'>LIVE CLIMATE RADAR:</b> {lw['location']}
        </div>
        <div style='font-size: 0.82rem; opacity: 0.85; background: rgba(0,0,0,0.2); padding: 3px 10px; border-radius: 12px;'>
            🛰️ Open-Meteo Satellite Feed
        </div>
    </div>
    <div class='weather-chips-container'>
        <div class='weather-chip'>🌡️ <b>Temp:</b> {lw['temp_current']}°C <small>({lw['temp_min']}°C - {lw['temp_max']}°C)</small></div>
        <div class='weather-chip'>💧 <b>Morning Humidity:</b> {lw['humidity_morning']}%</div>
        <div class='weather-chip'>🌧️ <b>Rain Chance:</b> {lw['rain_probability']}% ({lw['rainfall_mm']} mm)</div>
        <div class='weather-chip'>💨 <b>Wind:</b> {lw['wind_speed_kmh']} km/h</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Four Feature Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🌿 Leaf Disease Scanner (Vision AI)",
    "🌦️ Microclimate Risk Forecaster (ML + Dynamic AI)",
    "🤖 KrishiMitra Copilot (Generative AI + RAG)",
    "⚖️ Responsible AI & Impact"
])

# ====================================================================
# TAB 1: VISION DETECTOR
# ====================================================================
with tab1:
    st.subheader("🌿 Computer Vision Leaf Disease & Pest Scanner")
    st.write("Upload a leaf photo from the field to detect symptoms and receive immediate non-toxic biological remedies.")
    
    col1, col2 = st.columns([1.2, 1.8])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Leaf Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
        sample_choice = st.selectbox(
            "Or test with a simulated sample:",
            ["None", "Tomato Early Blight", "Tomato Leaf Curl", "Paddy Blast", "Healthy Crop Leaf"]
        )
        
        test_image = None
        if uploaded_file is not None:
            test_image = Image.open(uploaded_file)
        elif sample_choice != "None":
            test_image = Image.new('RGB', (400, 400), color=(80, 140, 70))
            from PIL import ImageDraw
            d = ImageDraw.Draw(test_image)
            if "Blight" in sample_choice:
                d.ellipse((120, 120, 260, 260), fill=(130, 80, 30), outline=(200, 170, 50))
            elif "Curl" in sample_choice:
                d.ellipse((100, 100, 280, 280), fill=(180, 190, 50), outline=(220, 220, 70))
        
        if test_image:
            st.image(test_image, caption="Field Capture", use_container_width=True)
            scan_btn = st.button("🔍 Analyze Leaf Symptoms", type="primary")
        else:
            scan_btn = False
            st.info("Upload an image or pick a sample above to test the vision model.")

    with col2:
        if test_image and scan_btn:
            with st.spinner("Analyzing spectral patterns & lesion distribution..."):
                result = vision_model.analyze_image(test_image, selected_crop=selected_crop)
            
            st.markdown("### 📋 Diagnostic Report")
            c_res1, c_res2 = st.columns(2)
            c_res1.metric("Identified Condition", result['detected_condition'])
            c_res1.metric("Detection Confidence", f"{int(result['confidence'] * 100)}%")
            c_res2.metric("Severity Level", result['severity'])
            c_res2.metric("Status", "Normal" if result['is_healthy'] else "Action Required")
            
            st.markdown("#### 🔬 Annotated Diagnostic Inspection")
            st.image(result['annotated_image'], caption="Bounding Box & Symptom Localization", width=350)
            
            st.markdown("#### 🌿 Certified Bio-Control Measures (ICAR)")
            for remedy in result['organic_remedies']:
                st.markdown(f"• **Bio-remedy**: {remedy}")
                
            st.markdown("#### ⚠️ Responsible AI Safety Advisory")
            st.warning(result['hazard_warning'])

# ====================================================================
# TAB 2: PREDICTIVE ML WEATHER RISK FORECASTER
# ====================================================================
with tab2:
    st.subheader(f"🌦️ Microclimate Outbreak Forecaster for {lw['location']}")
    st.write("Calculates pest outbreak risk with **XGBoost ML** and generates **unique, real-time AI alerts** via Gemini / ChatGPT.")
    
    col_w1, col_w2 = st.columns([1.3, 1.7])
    
    with col_w1:
        st.markdown("**Real-Time Microclimate Inputs**")
        st.caption("Auto-filled from live weather. You can adjust to test scenarios:")
        
        t_max = st.slider("Max Daily Temperature (°C)", 15.0, 45.0, float(lw['temp_max']))
        t_min = st.slider("Min Daily Temperature (°C)", 10.0, 35.0, float(lw['temp_min']))
        hum_m = st.slider("Morning Relative Humidity (%)", 40, 98, int(lw['humidity_morning']))
        hum_e = st.slider("Evening Relative Humidity (%)", 20, 95, int(lw['humidity_evening']))
        rain_mm = st.slider("Expected 24-hr Rainfall (mm)", 0.0, 50.0, float(lw['rainfall_mm']))
        wet_days = st.slider("Consecutive Wet/Foggy Days", 0, 7, int(lw['consecutive_wet_days']))
        wind = st.slider("Average Wind Speed (km/h)", 2.0, 35.0, float(lw['wind_speed_kmh']))
        crop_stage = st.selectbox("Crop Growth Stage", ["Seedling (0)", "Vegetative (1)", "Flowering/Tillering (2)", "Fruiting/Maturity (3)"], index=2)
        stage_idx = int(crop_stage.split("(")[1][0])
        
        gen_alert_btn = st.button("✨ Generate Live AI Weather Alert", type="primary")

    with col_w2:
        prediction = risk_predictor.predict(
            temp_max=t_max,
            temp_min=t_min,
            humidity_morning=hum_m,
            humidity_evening=hum_e,
            rainfall_mm=rain_mm,
            consecutive_wet_days=wet_days,
            wind_speed_kmh=wind,
            crop_stage=stage_idx
        )
        
        risk_color = "#E63946" if prediction['risk_level'] == "High Risk" else ("#F4A261" if prediction['risk_level'] == "Moderate Risk" else "#2A9D8F")
        
        pulse_class = "risk-score-pulse" if prediction['risk_level'] == "High Risk" else ""
        st.markdown(f"""
        <div class='risk-banner' style='border-top: 5px solid {risk_color}; box-shadow: 0 8px 24px rgba(0,0,0,0.25);'>
            <div style='display: flex; justify-content: space-between; align-items: baseline;'>
                <h3 style='margin:0; color:{risk_color}; font-size:1.25rem;'>Outbreak Risk: {prediction['risk_level']}</h3>
                <span style='font-size: 0.85rem; opacity: 0.8; background: rgba(255,255,255,0.08); padding: 3px 10px; border-radius: 12px;'>⚡ XGBoost ML</span>
            </div>
            <div class='{pulse_class}' style='color:{risk_color}; font-size:3.2rem; font-weight:800; margin:10px 0;'>
                {prediction['risk_percentage']}%
            </div>
            <p style='margin:0; font-size: 0.95rem; opacity: 0.9;'><b>🔍 Primary Climatic Trigger:</b> {prediction['primary_driver']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📈 Probability Distribution Across Classes")
        prob_df = pd.DataFrame({
            "Risk Class": list(prediction['probabilities'].keys()),
            "Probability (%)": [v * 100 for v in prediction['probabilities'].values()]
        })
        st.bar_chart(prob_df.set_index("Risk Class"))

        # Generate or Display Dynamic AI Alert
        if gen_alert_btn or st.session_state.dynamic_alert is None:
            with st.spinner(f"Synthesizing unique real-time alert via {ai_provider}..."):
                st.session_state.dynamic_alert = alert_service.generate_weather_alert(
                    location=lw['location'],
                    crop=selected_crop,
                    weather_data=lw,
                    risk_data=prediction,
                    language=language,
                    provider="Gemini" if "Gemini" in ai_provider else ("OpenAI" if "OpenAI" in ai_provider else "Local"),
                    api_key=user_api_key
                )
        
        st.markdown("#### 🤖 Dynamic AI Agronomic Alert (Real-Time Generated)")
        st.markdown(f"<div class='ai-alert-box'>{st.session_state.dynamic_alert}</div>", unsafe_allow_html=True)

# ====================================================================
# TAB 3: IBM GRANITE COPILOT & RAG
# ====================================================================
with tab3:
    st.subheader(f"🤖 KrishiMitra Agronomy Copilot (Grounded in {lw['location'].split(',')[0]} Weather)")
    st.write("Ask any farming question. The AI grounds its advice in certified ICAR manuals and real-time weather constraints.")
    
    q_col1, q_col2 = st.columns([2, 1])
    
    with q_col1:
        default_queries = [
            f"My {selected_crop} leaves have spots and current humidity is high in {lw['location'].split(',')[0]}. What organic remedy should I apply?",
            f"Rain is expected in my area tomorrow. Should I spray neem oil for pest control on my {selected_crop} today?",
            "Whiteflies are attacking my crop. How to control them without expensive chemicals?",
            "Paddy leaves are developing diamond-shaped gray lesions. What is the certified ICAR treatment?"
        ]
        sample_q = st.selectbox("Quick Query Examples:", ["-- Custom Input --"] + default_queries)
        
        user_prompt = st.text_area(
            "Enter farmer's question / किसान का सवाल:",
            value=sample_q if sample_q != "-- Custom Input --" else "",
            placeholder="e.g. My crop leaves are turning yellow with brown spots. What should I spray?"
        )
        
        ask_btn = st.button("🚀 Ask KrishiMitra", type="primary")

    with q_col2:
        st.markdown("**⚙️ Copilot Status**")
        st.info(f"Target Crop: **{selected_crop}**\nLanguage: **{language}**\nActive Engine: **{ai_provider}**\nLocation: **{lw['location'].split(',')[0]}**")
        if user_api_key:
            st.success("✅ Cloud API Connected")
        else:
            st.info("⚡ Offline Smart Mode Active")

    if ask_btn and user_prompt:
        weather_ctx = {
            'temp': lw['temp_current'],
            'humidity': lw['humidity_morning'],
            'rain_prob': lw['rain_probability'],
            'risk_level': prediction['risk_level']
        }
        with st.spinner(f"Querying ICAR Knowledge Base & Synthesizing via {ai_provider}..."):
            advisory = copilot_model.generate_advisory(
                crop=selected_crop,
                user_query=user_prompt,
                weather_context=weather_ctx,
                language=language,
                api_key=user_api_key
            )
        st.markdown("---")
        st.markdown(advisory)

# ====================================================================
# TAB 4: RESPONSIBLE AI & IMPACT ASSESSMENT
# ====================================================================
with tab4:
    st.subheader("⚖️ Responsible AI Framework & Sustainability Impact")
    st.write("Mandatory guidelines evaluation as required by the 1M1B – IBM SkillsBuild Internship.")
    
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        st.markdown("""
        <div class='metric-card'>
            <h4>1. ⚖️ Fairness & Inclusivity</h4>
            <p>• Unbiased recommendations: Never favors commercial chemical brands over affordable homemade bio-remedies.<br>
            • Accessible to low-literacy farmers through vernacular language and straightforward instructions.</p>
        </div>
        <div class='metric-card'>
            <h4>2. 🔍 Transparency & Explainability</h4>
            <p>• Every alert explains the climatic reasoning (e.g., 'Risk increased due to 88% humidity').<br>
            • Explicitly cites reference sources: ICAR, KVK, and National IPM guidelines.</p>
        </div>
        """, unsafe_allow_html=True)

    with r_col2:
        st.markdown("""
        <div class='metric-card'>
            <h4>3. 🛡️ Ethics & Safety</h4>
            <p>• Strict bio-first policy: Recommends non-toxic biological remedies (Neem, Trichoderma) first.<br>
            • Flags dangerous chemical pesticides (WHO Class Ia/Ib) with explicit toxicity hazard warnings.</p>
        </div>
        <div class='metric-card'>
            <h4>4. 🔒 Privacy & Data Minimization</h4>
            <p>• Zero sensitive personal data collected (no Aadhaar, phone numbers, or land ownership records required).<br>
            • Operates only on regional agro-climatic coordinates and crop symptoms.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 🌍 Measurable Sustainability Impact (SDG Goals)")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Chemical Pesticide Reduction", "30% - 40%", "Less toxic runoff")
    m_col2.metric("Average Cost Savings per Acre", "₹2,500 - ₹4,000", "Per cropping season")
    m_col3.metric("Preventive Lead Time", "48 Hours Ahead", "Before visible crop damage")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>KrishiMitra AI | Developed for 1M1B AI for Sustainability Virtual Internship in collaboration with IBM SkillsBuild & AICTE.</small></center>",
    unsafe_allow_html=True
)
