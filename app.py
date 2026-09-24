"""
====================================================================
🌾 KrishiMitra AI: Resilient Climate & Pest Advisory Copilot
1M1B - IBM SkillsBuild AI for Sustainability Virtual Internship
Aligned with UN SDGs: SDG 2 (Zero Hunger), SDG 13 (Climate Action), SDG 15 (Life on Land)
====================================================================
"""

import streamlit as st
import streamlit.components.v1 as components
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

# Custom Styling with Anime & Visual Novel Effects
st.markdown("""
<style>
    /* Forest & Agricultural Farm Landscape Background */
    [data-testid="stAppViewContainer"] {
        background: 
            linear-gradient(135deg, rgba(8, 24, 16, 0.88) 0%, rgba(4, 14, 9, 0.94) 100%),
            url("https://images.unsplash.com/photo-1625246333195-78d9c38ad449?q=80&w=1920&auto=format&fit=crop") center center / cover no-repeat fixed !important;
    }
    [data-testid="stSidebar"] {
        background: 
            linear-gradient(180deg, rgba(9, 25, 16, 0.94) 0%, rgba(4, 15, 9, 0.97) 100%),
            url("https://images.unsplash.com/photo-1625246333195-78d9c38ad449?q=80&w=1920&auto=format&fit=crop") center center / cover no-repeat fixed !important;
        border-right: 1px solid rgba(82, 183, 136, 0.3) !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Anime Floating & Bobbing Animations */
    @keyframes animeBob {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-7px) rotate(2deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    
    @keyframes eyeBlink {
        0%, 90%, 100% { transform: scaleY(1); }
        95% { transform: scaleY(0.1); }
    }
    
    @keyframes animeAura {
        0% { filter: drop-shadow(0 0 5px rgba(82, 183, 136, 0.5)); }
        50% { filter: drop-shadow(0 0 16px rgba(82, 183, 136, 0.9)) drop-shadow(0 0 25px rgba(255, 215, 0, 0.6)); }
        100% { filter: drop-shadow(0 0 5px rgba(82, 183, 136, 0.5)); }
    }
    
    @keyframes superAuraRed {
        0% { box-shadow: 0 0 10px #E63946, 0 0 20px #FF758F; }
        50% { box-shadow: 0 0 28px #E63946, 0 0 50px #FF4D6D; transform: scale(1.015); }
        100% { box-shadow: 0 0 10px #E63946, 0 0 20px #FF758F; }
    }
    
    @keyframes superAuraGold {
        0% { box-shadow: 0 0 10px #F4A261, 0 0 20px #E76F51; }
        50% { box-shadow: 0 0 25px #F4A261, 0 0 45px #E9C46A; transform: scale(1.01); }
        100% { box-shadow: 0 0 10px #F4A261, 0 0 20px #E76F51; }
    }

    @keyframes superAuraGreen {
        0% { box-shadow: 0 0 10px #2A9D8F, 0 0 18px #52B788; }
        50% { box-shadow: 0 0 22px #52B788, 0 0 35px #74C69D; transform: scale(1.01); }
        100% { box-shadow: 0 0 10px #2A9D8F, 0 0 18px #52B788; }
    }

    @keyframes laserSweep {
        0% { top: 2%; opacity: 0.9; }
        50% { top: 94%; opacity: 1; }
        100% { top: 2%; opacity: 0.9; }
    }

    @keyframes shimmerTitle {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes hudPulse {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; filter: drop-shadow(0 0 8px #00FFFF); }
    }

    /* Falling Anime Leaves / Sakura Particles */
    @keyframes animeFall1 {
        0% { top: -8%; transform: translateX(0vw) rotate(0deg); opacity: 0.85; }
        50% { transform: translateX(10vw) rotate(180deg); opacity: 0.7; }
        100% { top: 105%; transform: translateX(-4vw) rotate(360deg); opacity: 0; }
    }
    @keyframes animeFall2 {
        0% { top: -8%; transform: translateX(0vw) rotate(0deg); opacity: 0.8; }
        50% { transform: translateX(-12vw) rotate(220deg); opacity: 0.6; }
        100% { top: 105%; transform: translateX(6vw) rotate(440deg); opacity: 0; }
    }
    @keyframes animeFall3 {
        0% { top: -8%; transform: translateX(0vw) rotate(0deg); opacity: 0.85; }
        50% { transform: translateX(14vw) rotate(140deg); opacity: 0.55; }
        100% { top: 105%; transform: translateX(-8vw) rotate(300deg); opacity: 0; }
    }

    .anime-particle {
        position: fixed;
        z-index: 99999;
        pointer-events: none;
        user-select: none;
        font-size: 1.4rem;
    }
    .p1 { left: 12%; animation: animeFall1 12s linear infinite; animation-delay: 0s; }
    .p2 { left: 42%; animation: animeFall2 15s linear infinite; animation-delay: 2.5s; font-size: 1.6rem; }
    .p3 { left: 72%; animation: animeFall3 13s linear infinite; animation-delay: 5s; }
    .p4 { left: 88%; animation: animeFall1 17s linear infinite; animation-delay: 1.5s; font-size: 1.2rem; }
    .p5 { left: 28%; animation: animeFall2 14s linear infinite; animation-delay: 7s; }

    /* Anime Mascot Styling */
    .chibi-mascot-container {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(135deg, rgba(82, 183, 136, 0.22) 0%, rgba(45, 106, 79, 0.38) 100%);
        border: 2px solid #52B788;
        border-radius: 16px;
        padding: 12px 14px;
        margin-bottom: 18px;
        animation: animeAura 4s ease-in-out infinite;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .chibi-avatar {
        animation: animeBob 3.2s ease-in-out infinite;
        flex-shrink: 0;
    }
    .chibi-dialogue {
        font-size: 0.88rem;
        color: #E8F5E9;
        line-height: 1.35;
    }
    .chibi-name {
        font-size: 0.76rem;
        font-weight: 800;
        color: #74C69D;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    /* Cyber-Mecha Scanner HUD for Tab 1 */
    .hud-scanner-wrapper {
        position: relative;
        border-radius: 14px;
        overflow: hidden;
        border: 2px solid #52B788;
        box-shadow: 0 0 20px rgba(82, 183, 136, 0.35);
        background: rgba(10, 25, 18, 0.6);
        padding: 6px;
        margin-top: 10px;
    }
    .hud-laser-line {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, transparent, #2ECC71, #00FFFF, #2ECC71, transparent);
        box-shadow: 0 0 14px #00FFFF, 0 0 28px #2ECC71;
        animation: laserSweep 2.6s ease-in-out infinite;
        z-index: 100;
        pointer-events: none;
    }
    .hud-corner {
        position: absolute;
        width: 16px;
        height: 16px;
        border-color: #00FFFF;
        border-style: solid;
        z-index: 99;
        pointer-events: none;
    }
    .corner-tl { top: 8px; left: 8px; border-width: 3px 0 0 3px; }
    .corner-tr { top: 8px; right: 8px; border-width: 3px 3px 0 0; }
    .corner-bl { bottom: 8px; left: 8px; border-width: 0 0 3px 3px; }
    .corner-br { bottom: 8px; right: 8px; border-width: 0 3px 3px 0; }
    
    .hud-badge {
        position: absolute;
        bottom: 12px;
        right: 12px;
        background: rgba(10, 25, 18, 0.88);
        color: #00FFFF;
        font-family: monospace;
        font-size: 0.72rem;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #00FFFF;
        letter-spacing: 1px;
        z-index: 101;
        animation: hudPulse 2s infinite;
    }

    /* Anime Alert Banners */
    .anime-alert-banner {
        padding: 14px 18px;
        border-radius: 14px;
        margin-bottom: 14px;
        font-size: 1.02rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .danger-alert {
        background: linear-gradient(135deg, rgba(230, 57, 70, 0.25) 0%, rgba(150, 20, 30, 0.4) 100%);
        border: 2px solid #E63946;
        color: #FFCCD5;
        animation: superAuraRed 2s infinite;
    }
    .healthy-alert {
        background: linear-gradient(135deg, rgba(42, 157, 143, 0.25) 0%, rgba(20, 80, 70, 0.4) 100%);
        border: 2px solid #2A9D8F;
        color: #D8F3DC;
        animation: superAuraGreen 2.5s infinite;
    }
    .anime-ability-item {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #52B788;
        border-radius: 0 10px 10px 0;
        padding: 10px 14px;
        margin-bottom: 8px;
        transition: transform 0.2s ease, background 0.2s ease;
    }
    .anime-ability-item:hover {
        transform: translateX(6px);
        background: rgba(82, 183, 136, 0.18);
        border-left-color: #74C69D;
    }

    /* Anime RPG Visual Novel Dialogue Box */
    .anime-rpg-dialogue {
        background: linear-gradient(135deg, rgba(20, 42, 32, 0.95) 0%, rgba(10, 26, 19, 0.98) 100%);
        border: 2px solid #52B788;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.4), inset 0 0 15px rgba(82, 183, 136, 0.18);
        position: relative;
        margin-top: 18px;
        transition: all 0.3s ease;
    }
    .anime-rpg-dialogue:hover {
        border-color: #74C69D;
        box-shadow: 0 14px 45px rgba(82, 183, 136, 0.3);
        transform: translateY(-2px);
    }
    .rpg-badge {
        display: inline-block;
        background: #2D6A4F;
        color: #D8F3DC;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid #74C69D;
    }

    /* Grimoire Codex Styling for Tab 3 */
    .anime-grimoire-box {
        background: linear-gradient(135deg, rgba(18, 38, 28, 0.95) 0%, rgba(8, 20, 14, 0.98) 100%);
        border: 2px solid #52B788;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5), 0 0 20px rgba(82, 183, 136, 0.2);
        margin-top: 18px;
        position: relative;
    }
    .grimoire-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(82, 183, 136, 0.3);
        padding-bottom: 10px;
        margin-bottom: 14px;
    }

    /* Quest Cards for Tab 4 */
    .quest-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1.5px solid rgba(82, 183, 136, 0.3);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 14px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .quest-card:hover {
        transform: translateY(-4px);
        border-color: #74C69D;
        box-shadow: 0 10px 25px rgba(82, 183, 136, 0.25);
    }
    .quest-rank {
        display: inline-block;
        background: linear-gradient(135deg, #FFB703, #FB8500);
        color: #000;
        font-weight: 800;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 6px;
        margin-bottom: 6px;
    }

    /* Outbreak Auras */
    .danger-aura {
        animation: superAuraRed 2s infinite ease-in-out !important;
        border-color: #E63946 !important;
    }
    .moderate-aura {
        animation: superAuraGold 2.2s infinite ease-in-out !important;
        border-color: #F4A261 !important;
    }
    .healthy-aura {
        animation: superAuraGreen 2.5s infinite ease-in-out !important;
        border-color: #2A9D8F !important;
    }

    /* Core Title & Typography */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #52B788, #95D5B2, #74C69D, #00FFFF, #52B788);
        background-size: 250% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmerTitle 6s linear infinite;
        display: inline-block;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 1.01rem;
        color: #94D2BD;
        margin-bottom: 18px;
    }

    /* Weather Live Box - Glassmorphic */
    .weather-live-box {
        background: linear-gradient(135deg, rgba(27, 67, 50, 0.5) 0%, rgba(45, 106, 79, 0.4) 100%);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(82, 183, 136, 0.35);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        padding: 18px 22px;
        border-radius: 16px;
        color: #E8F5E9 !important;
        margin-bottom: 20px;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .weather-live-box:hover {
        transform: translateY(-3px);
        border-color: #74C69D;
    }

    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #2ECC71;
        box-shadow: 0 0 10px #2ECC71;
        margin-right: 7px;
        vertical-align: middle;
    }

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
        background: rgba(82, 183, 136, 0.3);
        transform: translateY(-2px);
        border-color: #52B788;
    }

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
        transform: scale(1.06);
    }

    /* Shiny Anime Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%) !important;
        color: #D8F3DC !important;
        border: 1.5px solid #52B788 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        cursor: pointer !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.03) !important;
        box-shadow: 0 0 18px rgba(82, 183, 136, 0.85), 0 0 32px rgba(0, 255, 200, 0.45) !important;
        border-color: #74C69D !important;
    }
    .stButton > button:active {
        transform: scale(0.96) !important;
    }

    /* Handcrafted Nature Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(10, 25, 18, 0.95);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #2D6A4F, #52B788);
        border-radius: 8px;
        border: 1px solid rgba(116, 198, 157, 0.3);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #52B788, #74C69D);
        box-shadow: 0 0 10px #52B788;
    }

    /* Floating Glass Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 35, 25, 0.65);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(82, 183, 136, 0.3);
        backdrop-filter: blur(12px);
        margin-bottom: 22px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 9px 18px !important;
        font-weight: 700;
        color: #A7C4B5 !important;
        border: 1px solid transparent;
        transition: all 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(82, 183, 136, 0.16) !important;
        color: #D8F3DC !important;
        transform: translateY(-2px);
        border-color: rgba(82, 183, 136, 0.4);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #52B788 !important;
        box-shadow: 0 4px 18px rgba(82, 183, 136, 0.4) !important;
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #52B788 !important;
        height: 3px;
        border-radius: 3px;
        box-shadow: 0 0 10px #52B788;
    }

    /* Bespoke Input, Select & Textarea Styling */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stTextArea textarea, .stTextInput input {
        background: rgba(14, 32, 23, 0.8) !important;
        border: 1.5px solid rgba(82, 183, 136, 0.35) !important;
        border-radius: 12px !important;
        color: #E8F5E9 !important;
        transition: all 0.25s ease;
    }
    div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover, .stTextArea textarea:hover, .stTextInput input:hover {
        border-color: #52B788 !important;
        box-shadow: 0 0 12px rgba(82, 183, 136, 0.35) !important;
    }
    div[data-baseweb="select"]:focus-within > div, div[data-baseweb="input"]:focus-within > div, .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #74C69D !important;
        box-shadow: 0 0 16px rgba(116, 198, 157, 0.5) !important;
    }

    /* Neon Emerald Sliders */
    .stSlider div[data-baseweb="slider"] div[role="slider"] {
        background-color: #52B788 !important;
        border: 2px solid #D8F3DC !important;
        box-shadow: 0 0 12px #52B788 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: grab !important;
    }
    .stSlider div[data-baseweb="slider"] div[role="slider"]:hover {
        transform: scale(1.25) !important;
        box-shadow: 0 0 18px #74C69D, 0 0 30px #52B788 !important;
    }

    /* Custom Radio Pill Badges */
    .stRadio div[role="radiogroup"] {
        gap: 10px;
    }
    .stRadio label {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(82, 183, 136, 0.25);
        padding: 5px 14px;
        border-radius: 20px;
        transition: all 0.2s ease;
        cursor: pointer !important;
    }
    .stRadio label:hover {
        background: rgba(82, 183, 136, 0.22);
        border-color: #52B788;
        transform: translateY(-2px);
    }

    /* Glossy Card Sheen Reflection */
    .weather-live-box, .risk-banner, .quest-card, .anime-rpg-dialogue, .anime-grimoire-box {
        position: relative;
        overflow: hidden;
    }
    .weather-live-box::after, .risk-banner::after, .quest-card::after, .anime-rpg-dialogue::after, .anime-grimoire-box::after {
        content: '';
        position: absolute;
        top: -60%;
        left: -60%;
        width: 220%;
        height: 220%;
        background: linear-gradient(60deg, transparent 40%, rgba(255, 255, 255, 0.05) 50%, transparent 60%);
        transform: rotate(25deg);
        transition: transform 0.8s ease;
        pointer-events: none;
    }
    .weather-live-box:hover::after, .risk-banner:hover::after, .quest-card:hover::after, .anime-rpg-dialogue:hover::after, .anime-grimoire-box:hover::after {
        transform: rotate(25deg) translate(30%, 30%);
    }

    /* Hide iframe container of custom script */
    iframe[title="streamlit.components.v1.html"], [data-testid="stCustomComponentV1"] {
        position: fixed !important;
        top: -200px !important;
        left: -200px !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
</style>

<!-- Nature Falling Leaves & Sprout Particles (Zero Pink) -->
<div class="anime-particle p1">🍃</div>
<div class="anime-particle p2">🌱</div>
<div class="anime-particle p3">🌿</div>
<div class="anime-particle p4">✨</div>
<div class="anime-particle p5">🌾</div>
""", unsafe_allow_html=True)

# --- INTERACTIVE LUMINOUS GREEN LIGHT MOUSE FOLLOWER & CLICK BURST (JAVASCRIPT) ---
INTERACTIVE_FX_JS = """
<script>
(function() {
    function initKrishiInteractive() {
        try {
            const parentDoc = (window.parent && window.parent.document) ? window.parent.document : document;
            if (!parentDoc || !parentDoc.body) {
                setTimeout(initKrishiInteractive, 200);
                return;
            }

            if (parentDoc.getElementById('krishi-green-light')) return;

            // Luminous Green Light Halo (प्यारी सी हरी रोशनी)
            const lightOrb = parentDoc.createElement('div');
            lightOrb.id = 'krishi-green-light';
            lightOrb.style.position = 'fixed';
            lightOrb.style.pointerEvents = 'none';
            lightOrb.style.zIndex = '99999999';
            lightOrb.style.width = '64px';
            lightOrb.style.height = '64px';
            lightOrb.style.borderRadius = '50%';
            lightOrb.style.background = 'radial-gradient(circle, rgba(46, 234, 128, 0.45) 0%, rgba(82, 183, 136, 0.22) 42%, rgba(46, 204, 113, 0) 70%)';
            lightOrb.style.boxShadow = '0 0 25px rgba(46, 234, 128, 0.65), 0 0 55px rgba(0, 255, 170, 0.35)';
            lightOrb.style.display = 'flex';
            lightOrb.style.alignItems = 'center';
            lightOrb.style.justifyContent = 'center';
            lightOrb.style.transform = 'translate(-50%, -50%)';
            lightOrb.style.transition = 'opacity 0.25s ease, width 0.2s ease, height 0.2s ease, box-shadow 0.2s ease';
            lightOrb.style.opacity = '0';

            const leaf = parentDoc.createElement('span');
            leaf.id = 'krishi-leaf-core';
            leaf.innerHTML = '🍃';
            leaf.style.fontSize = '22px';
            leaf.style.filter = 'drop-shadow(0 0 6px #00FF88) drop-shadow(0 0 12px rgba(82, 183, 136, 0.95))';
            leaf.style.display = 'inline-block';
            leaf.style.transition = 'transform 0.08s ease-out';
            lightOrb.appendChild(leaf);

            parentDoc.body.appendChild(lightOrb);

            let mouseX = -200, mouseY = -200;
            let leafX = -200, leafY = -200;
            let lastPollenTime = 0;
            let isHoveringInteractive = false;

            parentDoc.addEventListener('mousemove', function(e) {
                mouseX = e.clientX;
                mouseY = e.clientY;
                lightOrb.style.opacity = '1';

                const target = e.target;
                if (target && (
                    target.tagName === 'BUTTON' || 
                    target.tagName === 'INPUT' || 
                    target.tagName === 'SELECT' || 
                    target.tagName === 'A' || 
                    target.closest('button') || 
                    target.closest('[role="tab"]') ||
                    target.closest('.stButton') ||
                    target.closest('.stSelectbox') ||
                    target.closest('.stRadio') ||
                    target.closest('.risk-banner') ||
                    target.closest('.quest-card')
                )) {
                    if (!isHoveringInteractive) {
                        isHoveringInteractive = true;
                        lightOrb.style.width = '84px';
                        lightOrb.style.height = '84px';
                        lightOrb.style.boxShadow = '0 0 35px rgba(46, 234, 128, 0.85), 0 0 70px rgba(0, 255, 200, 0.5)';
                        leaf.style.fontSize = '26px';
                    }
                } else {
                    if (isHoveringInteractive) {
                        isHoveringInteractive = false;
                        lightOrb.style.width = '64px';
                        lightOrb.style.height = '64px';
                        lightOrb.style.boxShadow = '0 0 25px rgba(46, 234, 128, 0.65), 0 0 55px rgba(0, 255, 170, 0.35)';
                        leaf.style.fontSize = '22px';
                    }
                }

                const now = Date.now();
                if (now - lastPollenTime > 45) {
                    lastPollenTime = now;
                    spawnGreenFireflySpark(e.clientX, e.clientY);
                }
            }, { passive: true });

            parentDoc.addEventListener('mouseleave', function() {
                lightOrb.style.opacity = '0';
            });

            parentDoc.addEventListener('click', function(e) {
                triggerGreenLightBurst(e.clientX, e.clientY);
            });

            parentDoc.addEventListener('touchstart', function(e) {
                if (e.touches && e.touches[0]) {
                    mouseX = e.touches[0].clientX;
                    mouseY = e.touches[0].clientY;
                    triggerGreenLightBurst(mouseX, mouseY);
                }
            }, { passive: true });

            function spawnGreenFireflySpark(x, y) {
                const p = parentDoc.createElement('div');
                const icons = ['✨', '🌱', '•', '🍃', '🌾'];
                p.innerText = icons[Math.floor(Math.random() * icons.length)];
                p.style.position = 'fixed';
                p.style.left = (x + (Math.random() * 16 - 8)) + 'px';
                p.style.top = (y + (Math.random() * 16 - 8)) + 'px';
                p.style.pointerEvents = 'none';
                p.style.zIndex = '99999998';
                p.style.fontSize = (Math.random() * 6 + 10) + 'px';
                p.style.color = '#74C69D';
                p.style.opacity = '0.9';
                p.style.filter = 'drop-shadow(0 0 4px #00FF88)';
                p.style.transition = 'all 0.55s cubic-bezier(0.2, 0.8, 0.3, 1)';
                parentDoc.body.appendChild(p);

                requestAnimationFrame(function() {
                    const driftX = (Math.random() * 24 - 12);
                    const driftY = -(Math.random() * 26 + 12);
                    p.style.transform = 'translate(' + driftX + 'px, ' + driftY + 'px) scale(0.3) rotate(' + (Math.random() * 120) + 'deg)';
                    p.style.opacity = '0';
                });

                setTimeout(function() { p.remove(); }, 560);
            }

            function triggerGreenLightBurst(x, y) {
                const ring = parentDoc.createElement('div');
                ring.style.position = 'fixed';
                ring.style.left = x + 'px';
                ring.style.top = y + 'px';
                ring.style.width = '14px';
                ring.style.height = '14px';
                ring.style.borderRadius = '50%';
                ring.style.border = '2.5px solid #2ECC71';
                ring.style.boxShadow = '0 0 20px #00FF88, inset 0 0 12px #2ECC71';
                ring.style.transform = 'translate(-50%, -50%)';
                ring.style.pointerEvents = 'none';
                ring.style.zIndex = '99999999';
                ring.style.transition = 'all 0.5s cubic-bezier(0.1, 0.7, 0.1, 1)';
                parentDoc.body.appendChild(ring);

                requestAnimationFrame(function() {
                    ring.style.width = '90px';
                    ring.style.height = '90px';
                    ring.style.opacity = '0';
                    ring.style.borderColor = '#00FFFF';
                });
                setTimeout(function() { ring.remove(); }, 520);

                const count = 7;
                const items = ['🍃', '✨', '🌿', '🌾', '🌱', '⚡'];
                for (let i = 0; i < count; i++) {
                    const part = parentDoc.createElement('div');
                    part.innerText = items[i % items.length];
                    part.style.position = 'fixed';
                    part.style.left = x + 'px';
                    part.style.top = y + 'px';
                    part.style.pointerEvents = 'none';
                    part.style.zIndex = '99999999';
                    part.style.fontSize = '16px';
                    part.style.filter = 'drop-shadow(0 0 5px #00FF88)';
                    part.style.transform = 'translate(-50%, -50%)';
                    part.style.transition = 'all 0.65s cubic-bezier(0.12, 0.82, 0.32, 1.25)';
                    parentDoc.body.appendChild(part);

                    const angle = (i / count) * 2 * Math.PI + (Math.random() * 0.3);
                    const dist = Math.random() * 60 + 35;
                    const destX = Math.cos(angle) * dist;
                    const destY = Math.sin(angle) * dist;

                    requestAnimationFrame(function() {
                        part.style.transform = 'translate(calc(-50% + ' + destX + 'px), calc(-50% + ' + destY + 'px)) scale(0.35) rotate(' + (Math.random() * 360) + 'deg)';
                        part.style.opacity = '0';
                    });
                    setTimeout(function() { part.remove(); }, 680);
                }
            }

            function animateFollower() {
                const dx = mouseX - leafX;
                const dy = mouseY - leafY;
                leafX += dx * 0.18;
                leafY += dy * 0.18;

                const speed = Math.sqrt(dx * dx + dy * dy);
                let angle = Math.atan2(dy, dx) * (180 / Math.PI) + 45;
                
                if (speed < 2) {
                    const sway = Math.sin(Date.now() / 260) * 12;
                    angle += sway;
                }

                lightOrb.style.left = leafX + 'px';
                lightOrb.style.top = leafY + 'px';
                leaf.style.transform = 'rotate(' + angle + 'deg)';

                requestAnimationFrame(animateFollower);
            }
            requestAnimationFrame(animateFollower);

        } catch (err) {
            console.warn('KrishiMitra Green Light FX:', err);
        }
    }

    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        initKrishiInteractive();
    } else {
        document.addEventListener('DOMContentLoaded', initKrishiInteractive);
    }
})();
</script>
"""
components.html(INTERACTIVE_FX_JS, height=0, width=0)

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
    # Anime Chibi Mascot: Kisan-chan
    st.markdown("""
    <div class="chibi-mascot-container">
        <div class="chibi-avatar">
            <svg width="60" height="68" viewBox="0 0 80 85" fill="none" xmlns="http://www.w3.org/2000/svg">
                <!-- Straw Hat -->
                <ellipse cx="40" cy="22" rx="36" ry="12" fill="#E9C46A" stroke="#D4A373" stroke-width="2"/>
                <path d="M22,22 C22,8 58,8 58,22 Z" fill="#F4A261"/>
                <path d="M20,22 Q40,26 60,22" stroke="#BC6C25" stroke-width="3"/>
                <!-- Green Sprout on Hat -->
                <path d="M40,10 C46,2 54,6 40,10 Z" fill="#52B788"/>
                <path d="M40,10 C34,2 26,6 40,10 Z" fill="#74C69D"/>
                <!-- Head -->
                <ellipse cx="40" cy="46" rx="25" ry="22" fill="#FFE5D9"/>
                <!-- Anime Hair Bangs -->
                <path d="M19,38 Q28,50 36,40 Q44,52 52,40 Q57,48 61,38" fill="#583101"/>
                <!-- Eyes with Sparkles -->
                <ellipse cx="30" cy="46" rx="4.5" ry="6" fill="#1B4332"/>
                <circle cx="31.5" cy="44" r="2.2" fill="#FFFFFF"/>
                <ellipse cx="50" cy="46" rx="4.5" ry="6" fill="#1B4332"/>
                <circle cx="51.5" cy="44" r="2.2" fill="#FFFFFF"/>
                <!-- Anime Blush -->
                <ellipse cx="23" cy="53" rx="4.5" ry="2.5" fill="#FF8FA3" opacity="0.65"/>
                <ellipse cx="57" cy="53" rx="4.5" ry="2.5" fill="#FF8FA3" opacity="0.65"/>
                <!-- Cheerful Mouth -->
                <path d="M36,54 Q40,59 44,54" stroke="#583101" stroke-width="2.2" fill="none" stroke-linecap="round"/>
                <!-- Farmer Overalls -->
                <path d="M20,68 Q40,62 60,68 L58,84 L22,84 Z" fill="#2D6A4F"/>
                <!-- Straps & Button -->
                <line x1="28" y1="65" x2="28" y2="82" stroke="#1B4332" stroke-width="3"/>
                <line x1="52" y1="65" x2="52" y2="82" stroke="#1B4332" stroke-width="3"/>
                <circle cx="28" cy="72" r="2.5" fill="#E9C46A"/>
                <circle cx="52" cy="72" r="2.5" fill="#E9C46A"/>
                <text x="35" y="81" font-size="11">🌾</text>
            </svg>
        </div>
        <div class="chibi-dialogue">
            <div class="chibi-name" style="font-size: 0.86rem; color: #74C69D; font-weight: 700;">🌾 किसान मित्र (KrishiMitra)</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-top: 3px;">"नमस्ते! 🙏🌾"</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
<div style='background: linear-gradient(135deg, rgba(20, 48, 35, 0.45) 0%, rgba(10, 25, 18, 0.65) 100%); border: 1px solid rgba(82, 183, 136, 0.35); border-radius: 18px; padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); backdrop-filter: blur(12px);'>
    <div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;'>
        <div style='display: flex; align-items: center; gap: 14px;'>
            <span style='font-size: 2.8rem; filter: drop-shadow(0 0 12px rgba(82, 183, 136, 0.8));'>🌾</span>
            <div>
                <h1 class='main-title' style='margin: 0; font-size: 2.2rem;'>KrishiMitra AI</h1>
                <div style='font-size: 0.95rem; color: #94D2BD; font-weight: 500; margin-top: 2px;'>
                    Resilient Microclimate & Bio-Pest Advisory Copilot
                </div>
            </div>
        </div>
        <div style='display: flex; gap: 8px; flex-wrap: wrap;'>
            <span style='background: rgba(82, 183, 136, 0.15); border: 1px solid #52B788; color: #D8F3DC; font-size: 0.76rem; font-weight: 700; padding: 4px 12px; border-radius: 20px;'>🌿 ICAR CERTIFIED KB</span>
            <span style='background: rgba(0, 255, 255, 0.1); border: 1px solid #00FFFF; color: #A0F0ED; font-size: 0.76rem; font-weight: 700; padding: 4px 12px; border-radius: 20px;'>🛰️ SATELLITE TELEMETRY</span>
            <span style='background: rgba(255, 183, 3, 0.15); border: 1px solid #FFB703; color: #FFE3A8; font-size: 0.76rem; font-weight: 700; padding: 4px 12px; border-radius: 20px;'>⚡ GEMINI 3.6 FLASH</span>
        </div>
    </div>
    <div style='margin-top: 12px; font-size: 0.92rem; color: #C7E8D6; border-top: 1px solid rgba(82, 183, 136, 0.2); padding-top: 10px;'>
        Empowering smallholder farmers with proactive bio-control advisories grounded in real-time microclimate feeds, automated XGBoost outbreak forecasting, and responsible AI safety rails.
    </div>
</div>
""", unsafe_allow_html=True)

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
            st.markdown("""
            <div class='hud-scanner-wrapper'>
                <div class='hud-laser-line'></div>
                <div class='hud-corner corner-tl'></div>
                <div class='hud-corner corner-tr'></div>
                <div class='hud-corner corner-bl'></div>
                <div class='hud-corner corner-br'></div>
                <div class='hud-badge'>⚡ SPECTRAL LOCK: ACTIVE</div>
            """, unsafe_allow_html=True)
            st.image(test_image, caption="Field Capture Target", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            scan_btn = st.button("🔍 Launch Cyber Leaf Scan", type="primary")
        else:
            scan_btn = False
            st.info("Upload an image or pick a sample above to test the vision model.")

    with col2:
        if test_image and scan_btn:
            with st.spinner("Analyzing spectral patterns & lesion distribution..."):
                result = vision_model.analyze_image(test_image, selected_crop=selected_crop)
            
            if not result['is_healthy']:
                st.markdown(f"""
                <div class='anime-alert-banner danger-alert'>
                    🚨 <b>BIO-HAZARD DETECTED:</b> {result['detected_condition'].upper()}!
                    <div style='font-size: 0.85rem; font-weight: normal; margin-top: 4px;'>Severity: {result['severity']} | Immediate ICAR bio-shield recommended.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='anime-alert-banner healthy-alert'>
                    ✨ <b>OPTIMAL CROP HEALTH DETECTED!</b>
                    <div style='font-size: 0.85rem; font-weight: normal; margin-top: 4px;'>No pathogenic spores found. Crop foliage shows peak vitality!</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📋 Diagnostic HUD Telemetry")
            c_res1, c_res2 = st.columns(2)
            c_res1.metric("Identified Condition", result['detected_condition'])
            c_res1.metric("Detection Confidence", f"{int(result['confidence'] * 100)}%")
            c_res2.metric("Severity Level", result['severity'])
            c_res2.metric("Status", "Normal" if result['is_healthy'] else "Action Required")
            
            st.markdown("#### 🔬 Annotated Diagnostic Inspection")
            st.image(result['annotated_image'], caption="Bounding Box & Symptom Localization", width=350)
            
            st.markdown("#### 🌿 Certified Bio-Control Techniques (ICAR)")
            for remedy in result['organic_remedies']:
                st.markdown(f"""
                <div class='anime-ability-item'>
                    <b style='color: #74C69D;'>🌟 ICAR Bio-Defense:</b> {remedy}
                </div>
                """, unsafe_allow_html=True)
                
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
        
        if prediction['risk_level'] == "High Risk":
            risk_color = "#E63946"
            aura_class = "danger-aura"
            status_text = "💥 CRITICAL OUTBREAK THREAT DETECTED!"
        elif prediction['risk_level'] == "Moderate Risk":
            risk_color = "#F4A261"
            aura_class = "moderate-aura"
            status_text = "⚡ ELEVATED RISK WARNING"
        else:
            risk_color = "#2A9D8F"
            aura_class = "healthy-aura"
            status_text = "🍃 ZEN HARMONY: LOW RISK"
        
        st.markdown(f"""
        <div class='risk-banner {aura_class}' style='border-top: 5px solid {risk_color}; box-shadow: 0 8px 24px rgba(0,0,0,0.25);'>
            <div style='display: flex; justify-content: space-between; align-items: baseline;'>
                <h3 style='margin:0; color:{risk_color}; font-size:1.25rem;'>{status_text}</h3>
                <span style='font-size: 0.85rem; opacity: 0.85; background: rgba(255,255,255,0.08); padding: 3px 10px; border-radius: 12px;'>⚡ XGBoost ML</span>
            </div>
            <div style='color:{risk_color}; font-size:3.4rem; font-weight:900; margin:10px 0; text-shadow: 0 0 20px {risk_color}88;'>
                {prediction['risk_percentage']}%
            </div>
            <p style='margin:0; font-size: 0.95rem; opacity: 0.9;'><b>🎯 Primary Microclimate Driver:</b> {prediction['primary_driver']}</p>
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
        st.markdown(f"""
        <div class='anime-rpg-dialogue'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                <span class='rpg-badge'>📡 TACTICAL CLIMATE TRANSMISSION</span>
                <span style='color: #74C69D; font-size: 0.82rem; font-weight: 700;'>● LIVE VIA {ai_provider.upper()}</span>
            </div>
            <div style='display: flex; gap: 14px; align-items: flex-start;'>
                <div style='font-size: 2.2rem; filter: drop-shadow(0 0 10px rgba(82, 183, 136, 0.8));'>🌾</div>
                <div style='font-size: 0.98rem; line-height: 1.6; color: #E8F5E9;'>
                    {st.session_state.dynamic_alert}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
        st.markdown(f"""
        <div class='anime-grimoire-box'>
            <div class='grimoire-header'>
                <span class='rpg-badge'>📜 ICAR TACTICAL AGRONOMY CODEX</span>
                <span style='color: #74C69D; font-weight: 700; font-size: 0.85rem;'>★ RANK S ADVISORY ★</span>
            </div>
            <div style='line-height: 1.6; color: #E8F5E9;'>
                {advisory}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ====================================================================
# TAB 4: RESPONSIBLE AI & IMPACT ASSESSMENT
# ====================================================================
with tab4:
    st.subheader("⚖️ Responsible AI Framework & Sustainability Impact")
    st.write("Mandatory guidelines evaluation as required by the 1M1B – IBM SkillsBuild Internship.")
    
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        st.markdown("""
        <div class='quest-card'>
            <span class='quest-rank'>RANK S MISSION</span>
            <h4 style='margin: 4px 0 8px 0; color: #74C69D;'>1. ⚖️ Fairness & Inclusivity</h4>
            <p style='margin: 0; font-size: 0.92rem; line-height: 1.5;'>• Unbiased recommendations: Never favors commercial chemical brands over affordable homemade bio-remedies.<br>
            • Accessible to low-literacy farmers through vernacular language and straightforward instructions.</p>
        </div>
        <div class='quest-card'>
            <span class='quest-rank'>RANK S MISSION</span>
            <h4 style='margin: 4px 0 8px 0; color: #74C69D;'>2. 🔍 Transparency & Explainability</h4>
            <p style='margin: 0; font-size: 0.92rem; line-height: 1.5;'>• Every alert explains the climatic reasoning (e.g., 'Risk increased due to 88% humidity').<br>
            • Explicitly cites reference sources: ICAR, KVK, and National IPM guidelines.</p>
        </div>
        """, unsafe_allow_html=True)

    with r_col2:
        st.markdown("""
        <div class='quest-card'>
            <span class='quest-rank'>RANK S MISSION</span>
            <h4 style='margin: 4px 0 8px 0; color: #74C69D;'>3. 🛡️ Ethics & Safety Shield</h4>
            <p style='margin: 0; font-size: 0.92rem; line-height: 1.5;'>• Strict bio-first policy: Recommends non-toxic biological remedies (Neem, Trichoderma) first.<br>
            • Flags dangerous chemical pesticides (WHO Class Ia/Ib) with explicit toxicity hazard warnings.</p>
        </div>
        <div class='quest-card'>
            <span class='quest-rank'>RANK S MISSION</span>
            <h4 style='margin: 4px 0 8px 0; color: #74C69D;'>4. 🔒 Privacy & Data Minimization</h4>
            <p style='margin: 0; font-size: 0.92rem; line-height: 1.5;'>• Zero sensitive personal data collected (no Aadhaar, phone numbers, or land ownership records required).<br>
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
