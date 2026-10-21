import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import best_model_name, load_model, predict_urls

st.set_page_config(
    page_title="PhishPop! | Playful Phishing URL Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# stop early if the models haven't been trained yet
if not os.path.exists("models/best_model.txt"):
    st.error("🚨 No trained model found. Please run `python -m src.train` first.")
    st.stop()


@st.cache_resource
def get_model(name):
    return load_model(name)


@st.cache_data
def get_importance():
    return pd.read_csv("models/feature_importance.csv")


@st.cache_data
def get_results():
    if os.path.exists("models/results.csv"):
        return pd.read_csv("models/results.csv")
    return None


MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
best_model = best_model_name()

FEATURE_META = {
    "has_https": {"name": "HTTPS Encryption", "category": "Security", "color": "#8B5CF6"},
    "subdomain_count": {"name": "Subdomain Depth", "category": "Structure", "color": "#F472B6"},
    "hostname_length": {"name": "Hostname Length", "category": "Lexical", "color": "#FBBF24"},
    "domain_length": {"name": "Domain Length", "category": "Lexical", "color": "#34D399"},
    "dot_count": {"name": "Dot (.) Count", "category": "Structure", "color": "#8B5CF6"},
    "hyphen_count": {"name": "Hyphen (-) Count", "category": "Lexical", "color": "#F472B6"},
    "domain_hyphen_count": {"name": "Domain Hyphens", "category": "Lexical", "color": "#FBBF24"},
    "digit_count": {"name": "Digit Count", "category": "Lexical", "color": "#34D399"},
    "digit_ratio": {"name": "Digit Ratio", "category": "Lexical", "color": "#8B5CF6"},
    "tld_length": {"name": "TLD Length", "category": "Structure", "color": "#F472B6"},
    "has_at": {"name": "@ Symbol Trap", "category": "Security", "color": "#EF4444"},
    "has_ip": {"name": "Direct IP Host", "category": "Security", "color": "#EF4444"},
    "risky_tld": {"name": "High-Risk TLD", "category": "Security", "color": "#FBBF24"},
    "is_shortener": {"name": "URL Shortener", "category": "Security", "color": "#8B5CF6"},
    "keyword_count": {"name": "Target Keywords", "category": "Lexical", "color": "#F472B6"},
}

# custom styling
st.markdown("""
<style>
    /* fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1E293B;
    }

    h1, h2, h3, h4, h5, h6, .pop-heading {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        color: #1E293B !important;
        letter-spacing: -0.02em;
    }

    /* page background */
    .stApp {
        background-color: #FFFDF5 !important;
        background-image: none !important;
        color: #1E293B;
    }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFDF5 !important;
        background-image: none !important;
        border-right: 2px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.9rem;
    }

    /* cards */
    .sticker-card {
        background: #FFFFFF;
        border: 2px solid #1E293B;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 6px 6px 0px 0px #1E293B;
        transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
    }
    .sticker-card:hover {
        transform: translate(-2px, -2px) rotate(-0.5deg);
        box-shadow: 8px 8px 0px 0px #1E293B;
    }

    /* hero banner */
    .hero-banner {
        background: #FFFFFF;
        border: 2.5px solid #1E293B;
        border-radius: 24px;
        padding: 32px 36px;
        margin-bottom: 28px;
        box-shadow: 8px 8px 0px 0px #8B5CF6, 8px 8px 0px 2.5px #1E293B;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        color: #1E293B;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 500;
        max-width: 820px;
        line-height: 1.5;
    }

    /* badges */
    .pop-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        border: 2px solid #1E293B;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 0.85rem;
        box-shadow: 3px 3px 0px 0px #1E293B;
        letter-spacing: 0.02em;
    }
    .badge-violet { background: #8B5CF6; color: #FFFFFF; }
    .badge-pink   { background: #F472B6; color: #FFFFFF; }
    .badge-yellow { background: #FBBF24; color: #1E293B; }
    .badge-mint   { background: #34D399; color: #1E293B; }
    .badge-white  { background: #FFFFFF; color: #1E293B; }

    /* tilted tag */
    .tape-tag {
        display: inline-block;
        padding: 4px 14px;
        background: #FBBF24;
        color: #1E293B;
        border: 2px solid #1E293B;
        font-family: 'Outfit', sans-serif;
        font-weight: 900;
        font-size: 0.8rem;
        text-transform: uppercase;
        transform: rotate(-3deg);
        box-shadow: 3px 3px 0px 0px #1E293B;
        border-radius: 6px;
    }

    /* verdict cards */
    .verdict-phish {
        background: #FFF1F2;
        border: 2.5px solid #1E293B;
        border-radius: 22px;
        padding: 28px 32px;
        margin-bottom: 26px;
        box-shadow: 8px 8px 0px 0px #F472B6, 8px 8px 0px 2.5px #1E293B;
        position: relative;
    }
    .verdict-safe {
        background: #ECFDF5;
        border: 2.5px solid #1E293B;
        border-radius: 22px;
        padding: 28px 32px;
        margin-bottom: 26px;
        box-shadow: 8px 8px 0px 0px #34D399, 8px 8px 0px 2.5px #1E293B;
        position: relative;
    }

    /* small stat tiles */
    .pop-tile {
        background: #FFFFFF;
        border: 2px solid #1E293B;
        border-radius: 16px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 4px 4px 0px 0px #1E293B;
        transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .pop-tile:hover {
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px 0px #1E293B;
    }
    .pop-tile-val {
        font-family: 'Outfit', sans-serif;
        font-size: 1.45rem;
        font-weight: 900;
        line-height: 1.2;
        margin: 6px 0 2px 0;
    }
    .pop-tile-lbl {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }

    /* url box */
    .pop-url-pill {
        font-family: 'JetBrains Mono', monospace;
        background: #F8FAFC;
        border: 2px solid #1E293B;
        border-radius: 12px;
        padding: 10px 16px;
        color: #0F172A;
        font-size: 0.95rem;
        word-break: break-all;
        margin: 12px 0 16px 0;
        box-shadow: 3px 3px 0px 0px #E2E8F0;
    }

    /* probability bar */
    .candy-meter-bg {
        width: 100%;
        height: 18px;
        background: #E2E8F0;
        border: 2px solid #1E293B;
        border-radius: 9999px;
        overflow: hidden;
        margin-top: 12px;
        box-shadow: inset 2px 2px 0px rgba(0,0,0,0.1);
    }
    .candy-meter-fill-red {
        height: 100%;
        background: repeating-linear-gradient(
            45deg,
            #EF4444,
            #EF4444 10px,
            #F87171 10px,
            #F87171 20px
        );
        border-radius: 9999px;
        border-right: 2px solid #1E293B;
    }
    .candy-meter-fill-green {
        height: 100%;
        background: repeating-linear-gradient(
            45deg,
            #10B981,
            #10B981 10px,
            #34D399 10px,
            #34D399 20px
        );
        border-radius: 9999px;
        border-right: 2px solid #1E293B;
    }

    /* tabs */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        padding: 10px 24px !important;
        border: 2px solid transparent !important;
        border-radius: 14px 14px 0 0 !important;
        color: #64748B !important;
        margin-right: 6px !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1E293B !important;
        background: #FFFFFF !important;
        border: 2px solid #1E293B !important;
        border-bottom: 2px solid #FFFFFF !important;
        box-shadow: 4px -4px 0px 0px #FBBF24 !important;
        transform: translateY(2px) !important;
    }

    /* buttons */
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        border: 2.5px solid #1E293B !important;
        border-radius: 9999px !important;
        padding: 10px 24px !important;
        box-shadow: 4px 4px 0px 0px #1E293B !important;
        transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }
    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div {
        color: #1E293B !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.92rem !important;
    }
    div.stButton > button:hover {
        transform: translate(-2px, -2px) !important;
        box-shadow: 6px 6px 0px 0px #1E293B !important;
        background-color: #FBBF24 !important;
        color: #1E293B !important;
    }
    div.stButton > button:hover p,
    div.stButton > button:hover span {
        color: #1E293B !important;
    }
    div.stButton > button:active {
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0px 0px #1E293B !important;
    }

    /* primary button */
    div.stButton > button[kind="primary"] {
        background-color: #8B5CF6 !important;
        color: #FFFFFF !important;
        border: 2.5px solid #1E293B !important;
        box-shadow: 4px 4px 0px 0px #1E293B !important;
    }
    div.stButton > button[kind="primary"] p,
    div.stButton > button[kind="primary"] span {
        color: #FFFFFF !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 900 !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #7C3AED !important;
        color: #FFFFFF !important;
        box-shadow: 6px 6px 0px 0px #1E293B !important;
    }

    /* text input */
    div[data-testid="stTextInputRootElement"],
    div[data-testid="stTextInput"] > div,
    div[data-testid="stTextInput"] > div > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 2.5px solid #1E293B !important;
        border-radius: 14px !important;
        box-shadow: 4px 4px 0px 0px #CBD5E1 !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="base-input"]:focus-within,
    div[data-baseweb="input"]:focus-within {
        border-color: #8B5CF6 !important;
        box-shadow: 4px 4px 0px 0px #8B5CF6 !important;
    }
    input[type="text"],
    input[data-testid="stTextInput"],
    div[data-baseweb="input"] input {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        -webkit-text-fill-color: #1E293B !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
    }

    /* selectbox */
    div[data-baseweb="select"] > div {
        border: 2.5px solid #1E293B !important;
        border-radius: 12px !important;
        background: #FFFFFF !important;
        box-shadow: 3px 3px 0px 0px #1E293B !important;
    }

    /* tables */
    div[data-testid="stDataFrame"] {
        border: 2px solid #1E293B !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 6px 6px 0px 0px #1E293B !important;
        background: #FFFFFF !important;
    }

    /* file uploader */
    div[data-testid="stFileUploader"] {
        background: #FFFFFF !important;
        border: 2px dashed #1E293B !important;
        border-radius: 18px !important;
        padding: 16px !important;
        box-shadow: 4px 4px 0px 0px #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)


# sidebar
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 8px;">
            <div style="background: #FBBF24; border: 2px solid #1E293B; padding: 12px 14px; border-radius: 18px; box-shadow: 4px 4px 0px 0px #1E293B;">
                <span style="font-size: 1.8rem;">⚡</span>
            </div>
            <div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 900; color: #1E293B; line-height: 1;">PhishPop!</div>
                <div style="font-size: 0.78rem; font-weight: 800; color: #8B5CF6; letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px;">Geometric AI Detector</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border: none; border-top: 2px solid #1E293B; margin: 16px 0;'>", unsafe_allow_html=True)

    st.markdown("#### 🎮 AI Engine Selection")
    model_name = st.selectbox(
        "Active Algorithm",
        MODEL_NAMES,
        index=MODEL_NAMES.index(best_model) if best_model in MODEL_NAMES else 0,
        help="Select which algorithm scores URL threat probabilities."
    )

    is_champion = (model_name == best_model)
    st.markdown(f"""
        <div style="background: #FFFFFF; border: 2px solid #1E293B; border-radius: 14px; padding: 14px; box-shadow: 4px 4px 0px 0px {'#F472B6' if is_champion else '#E2E8F0'}; margin-top: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.75rem; font-weight: 800; text-transform: uppercase; color: #64748B;">ACTIVE CLASSIFIER</span>
                <span class="pop-badge {'badge-pink' if is_champion else 'badge-yellow'}" style="padding: 2px 10px; font-size: 0.75rem; box-shadow: 2px 2px 0px 0px #1E293B;">
                    {'⭐ BEST PICK' if is_champion else 'TESTED'}
                </span>
            </div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.15rem; font-weight: 800; color: #1E293B; margin-top: 6px;">{model_name}</div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Top Benchmark Performer: <b style="color: #8B5CF6;">{best_model}</b></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border: none; border-top: 2px solid #1E293B; margin: 16px 0;'>", unsafe_allow_html=True)

    with st.expander("🎈 Threat Scale Guide", expanded=False):
        st.markdown("""
        - 🟢 **0% - 35%**: **Safe & Benign**
          Normal domain lengths, valid HTTPS, standard structure.
        - 🟡 **36% - 69%**: **Suspicious**
          Multiple subdomains, high digit ratios, or odd prefixes.
        - 🔴 **70% - 100%**: **Phishing Threat!**
          Deceptive brand keywords, suspicious TLDs, or credential traps.
        """)

    st.markdown("""
        <div style="margin-top: 28px; text-align: center;">
            <div class="tape-tag">✨ PLAYFUL GEOMETRIC v2.0</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 8px; font-weight: 600;">50,000+ Phishing & Safe URLs</div>
        </div>
    """, unsafe_allow_html=True)

model = get_model(model_name)


# header banner
st.markdown(f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span class="tape-tag">⚡ THREAT INTELLIGENCE</span>
                    <span class="pop-badge badge-yellow">LIVE ML ENGINE</span>
                </div>
                <div class="hero-title">
                    PhishPop! URL Threat Radar
                </div>
                <div class="hero-subtitle">
                    A punchy, tactile URL security playground. Parse domain anatomy, uncover sneaky typosquatters, and benchmark machine learning algorithms with energetic clarity!
                </div>
            </div>
            <div style="display: flex; gap: 10px; flex-direction: column; align-items: flex-end;">
                <span class="pop-badge badge-violet">ENGINE: {model_name.upper()}</span>
                <span class="pop-badge badge-mint">🛡️ 15 GEOMETRIC VECTORS</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


tab_single, tab_batch, tab_models = st.tabs([
    "🔍 Single URL Scanner",
    "📂 Batch URL Inspector",
    "📊 Model Performance & Benchmarks"
])


# single url tab
with tab_single:
    st.markdown("<h3 class='pop-heading'>🔍 Single URL Inspection</h3>", unsafe_allow_html=True)
    st.markdown("Enter any target web address below or click our quick test sticker samples to inspect the lexical and structural signals:")

    st.markdown("<p style='font-size: 0.82rem; font-weight: 800; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;'>QUICK TEST SAMPLES:</p>", unsafe_allow_html=True)
    sample_cols = st.columns(5)
    sample_urls = [
        ("🟢 Google (Safe)", "https://www.google.com"),
        ("🔴 PayPal Spoof", "http://secure-login.paypal.com.verify-id.xyz/update?id=123"),
        ("🔴 Raw IP Bank Trap", "http://192.168.10.5/bank/login.php"),
        ("🔴 SBI Verify (.tk)", "http://account-verify-sbi-online.tk/confirm/password"),
        ("🟢 Wikipedia ML", "https://www.wikipedia.org/wiki/Machine_learning"),
    ]

    if "current_url" not in st.session_state:
        st.session_state.current_url = "https://www.google.com"

    for idx, (label, sample_url) in enumerate(sample_urls):
        with sample_cols[idx]:
            if st.button(label, key=f"sample_{idx}", use_container_width=True):
                st.session_state.current_url = sample_url

    input_col, btn_col = st.columns([5, 1])
    with input_col:
        url_input = st.text_input(
            "Target Web Address",
            value=st.session_state.current_url,
            placeholder="e.g. http://secure-login.bank-update.top/signin",
            label_visibility="collapsed",
        )
    with btn_col:
        scan_triggered = st.button("🚀 SCAN URL", type="primary", use_container_width=True)

    active_url = url_input.strip()
    if active_url:
        with st.spinner("Parsing geometric & lexical vectors..."):
            result, features = predict_urls([active_url], model)
            label = result.loc[0, "Prediction"]
            prob = float(result.loc[0, "Phishing probability"])

        is_phishing = (label == "Phishing")
        verdict_class = "verdict-phish" if is_phishing else "verdict-safe"
        verdict_icon = "🚨" if is_phishing else "🎉"
        verdict_title = "MALICIOUS PHISHING DETECTED!" if is_phishing else "VERIFIED LEGITIMATE & SAFE!"
        badge_style = "badge-pink" if is_phishing else "badge-mint"

        # verdict card
        st.markdown(f"""
            <div class="{verdict_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 2.2rem;">{verdict_icon}</span>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 900; color: #1E293B;">
                                {verdict_title}
                            </div>
                        </div>
                        <div class="pop-url-pill">{active_url}</div>
                    </div>
                    <div style="text-align: right;">
                        <span class="pop-badge {badge_style}">
                            {label.upper()} VERDICT
                        </span>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 2.8rem; font-weight: 900; color: {'#EF4444' if is_phishing else '#10B981'}; line-height: 1; margin-top: 6px;">
                            {prob:.1%}
                        </div>
                        <div style="font-size: 0.78rem; font-weight: 800; text-transform: uppercase; color: #64748B;">Phishing Probability</div>
                    </div>
                </div>
                <div class="candy-meter-bg">
                    <div class="{'candy-meter-fill-red' if is_phishing else 'candy-meter-fill-green'}" style="width: {max(min(prob * 100, 100), 5):.1f}%;"></div>
                </div>
                <div style="margin-top: 16px; font-size: 0.95rem; font-weight: 600; color: #334155; line-height: 1.5;">
                    {'⚠️ <b>Security Alert:</b> This URL exhibits lexical markers characteristic of fraudulent websites, deceptive subdomains, or unencrypted credential traps. Do NOT submit passwords or personal information!'
                     if is_phishing else
                     '✅ <b>All Clear:</b> Domain structure conforms to standard legitimate domain conventions with expected lexical lengths, valid SSL/TLS attributes, and benign metrics.'}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # four signal tiles
        st.markdown("<h4 class='pop-heading'>🧩 Structural & Security Signals</h4>", unsafe_allow_html=True)
        ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)

        has_https = bool(features.loc[0, "has_https"])
        has_ip = bool(features.loc[0, "has_ip"])
        is_risky_tld = bool(features.loc[0, "risky_tld"])
        keywords_detected = int(features.loc[0, "keyword_count"])

        with ind_col1:
            st.markdown(f"""
                <div class="pop-tile">
                    <div style="font-size: 1.8rem;">{'🔒' if has_https else '⚠️'}</div>
                    <div class="pop-tile-val" style="color: {'#10B981' if has_https else '#EF4444'};">
                        {'HTTPS Secure' if has_https else 'Plain HTTP'}
                    </div>
                    <div class="pop-tile-lbl">Protocol Security</div>
                </div>
            """, unsafe_allow_html=True)

        with ind_col2:
            st.markdown(f"""
                <div class="pop-tile">
                    <div style="font-size: 1.8rem;">{'🌐' if not has_ip else '🚨'}</div>
                    <div class="pop-tile-val" style="color: {'#10B981' if not has_ip else '#EF4444'};">
                        {'Named Domain' if not has_ip else 'Direct IP Host'}
                    </div>
                    <div class="pop-tile-lbl">Host Identity</div>
                </div>
            """, unsafe_allow_html=True)

        with ind_col3:
            st.markdown(f"""
                <div class="pop-tile">
                    <div style="font-size: 1.8rem;">{'🛡️' if not is_risky_tld else '⚠️'}</div>
                    <div class="pop-tile-val" style="color: {'#10B981' if not is_risky_tld else '#FBBF24'};">
                        {'Standard TLD' if not is_risky_tld else 'Risky (.xyz/.tk)'}
                    </div>
                    <div class="pop-tile-lbl">TLD Reputation</div>
                </div>
            """, unsafe_allow_html=True)

        with ind_col4:
            st.markdown(f"""
                <div class="pop-tile">
                    <div style="font-size: 1.8rem;">{'🔑' if keywords_detected > 0 else '✨'}</div>
                    <div class="pop-tile-val" style="color: {'#EF4444' if keywords_detected > 0 else '#10B981'};">
                        {keywords_detected} Keyword{'' if keywords_detected == 1 else 's'}
                    </div>
                    <div class="pop-tile-lbl">Target Keywords</div>
                </div>
            """, unsafe_allow_html=True)

        # feature importance chart and table
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        chart_col, table_col = st.columns([1, 1])

        importance_df = get_importance()
        top_features = (
            importance_df[["feature", model_name]]
            .sort_values(model_name, ascending=False)
            .head(10)
            .rename(columns={model_name: "importance"})
        )
        top_features["value in this URL"] = [features.loc[0, f] for f in top_features["feature"]]
        top_features["friendly_name"] = [
            FEATURE_META.get(f, {}).get("name", f) for f in top_features["feature"]
        ]
        top_features["category"] = [
            FEATURE_META.get(f, {}).get("category", "General") for f in top_features["feature"]
        ]

        with chart_col:
            st.markdown("<h4 class='pop-heading'>📊 Feature Influence Breakdown</h4>", unsafe_allow_html=True)
            st.caption(f"Top 10 most influential features contributing to the {model_name} prediction.")

            fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#FFFDF5")
            ax.set_facecolor("#FFFFFF")

            y_pos = range(len(top_features))
            weights = top_features["importance"].iloc[::-1]
            names = top_features["friendly_name"].iloc[::-1]

            palette = ["#8B5CF6", "#F472B6", "#FBBF24", "#34D399"]
            bar_colors = [palette[i % len(palette)] for i in range(len(top_features))]

            bars = ax.barh(
                y_pos,
                weights,
                color=bar_colors,
                edgecolor="#1E293B",
                linewidth=1.8,
                height=0.68,
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9.5, color="#1E293B", fontweight="600")
            ax.set_xlabel("Relative Feature Weight", fontsize=9.5, color="#475569", fontweight="700")
            ax.tick_params(axis="x", colors="#1E293B", labelsize=8.5)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#1E293B")
            ax.spines["left"].set_linewidth(2)
            ax.spines["bottom"].set_color("#1E293B")
            ax.spines["bottom"].set_linewidth(2)
            ax.grid(axis="x", linestyle=":", alpha=0.3, color="#1E293B")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with table_col:
            st.markdown("<h4 class='pop-heading'>🔬 Observed Feature Values</h4>", unsafe_allow_html=True)
            st.caption("Exact numerical & categorical values extracted from the analyzed URL.")

            display_df = top_features[["friendly_name", "category", "value in this URL", "importance"]].copy()
            display_df.columns = ["Feature", "Category", "Value in URL", "Model Weight"]
            display_df["Model Weight"] = display_df["Model Weight"].apply(lambda w: f"{w:.3f}")

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                height=320,
            )


# batch tab
with tab_batch:
    st.markdown("<h3 class='pop-heading'>📂 Batch URL Inspector</h3>", unsafe_allow_html=True)
    st.markdown("Scan bulk domain logs, proxy files, or security alerts simultaneously to isolate threats with speed:")

    upload_col, sample_col = st.columns([2, 1])

    with upload_col:
        file = st.file_uploader(
            "Upload CSV File containing URLs",
            type=["csv"],
            help="The CSV must contain a column named 'url' (or URLs in the first column)."
        )

    with sample_col:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        st.markdown("<h5 class='pop-heading' style='margin-bottom: 4px;'>Or Load Built-in Test Set</h5>", unsafe_allow_html=True)
        st.caption("Instantly test the batch pipeline with our pre-configured attack vectors.")
        use_sample = st.button("📁 LOAD `test_urls.csv`", key="load_sample_csv", use_container_width=True)

    data = None
    if file is not None:
        data = pd.read_csv(file)
    elif use_sample and os.path.exists("test_urls.csv"):
        data = pd.read_csv("test_urls.csv")
        st.success("Loaded built-in `test_urls.csv` successfully!")

    if data is not None and not data.empty:
        col = next((c for c in data.columns if c.lower() == "url"), data.columns[0])
        url_list = data[col].astype(str).tolist()

        with st.spinner(f"Analyzing {len(url_list)} URLs with {model_name}..."):
            result, _ = predict_urls(url_list, model)

        total_scanned = len(result)
        phishing_count = int((result["Prediction"] == "Phishing").sum())
        legitimate_count = total_scanned - phishing_count
        threat_rate = (phishing_count / total_scanned) if total_scanned > 0 else 0

        # summary tiles
        st.markdown("<hr style='border: none; border-top: 2px solid #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)
        st.markdown("<h4 class='pop-heading'>📊 Batch Scan Executive Summary</h4>", unsafe_allow_html=True)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(f"""
                <div class="pop-tile">
                    <div class="pop-tile-val" style="color: #8B5CF6;">{total_scanned}</div>
                    <div class="pop-tile-lbl">Total URLs Scanned</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
                <div class="pop-tile">
                    <div class="pop-tile-val" style="color: #EF4444;">{phishing_count}</div>
                    <div class="pop-tile-lbl">Phishing Threats Flagged</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
                <div class="pop-tile">
                    <div class="pop-tile-val" style="color: #10B981;">{legitimate_count}</div>
                    <div class="pop-tile-lbl">Legitimate / Safe URLs</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
                <div class="pop-tile">
                    <div class="pop-tile-val" style="color: {'#EF4444' if threat_rate > 0.4 else '#FBBF24' if threat_rate > 0.1 else '#10B981'};">{threat_rate:.1%}</div>
                    <div class="pop-tile-lbl">Batch Threat Rate</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # filter and download
        table_controls, download_col = st.columns([3, 1])
        with table_controls:
            filter_choice = st.radio(
                "Filter View",
                options=["All Records", "🚨 Phishing Only", "🛡️ Legitimate Only"],
                horizontal=True,
                label_visibility="collapsed"
            )

        with download_col:
            csv_data = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 DOWNLOAD CSV REPORT",
                data=csv_data,
                file_name="phishpop_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        display_batch_df = result.copy()
        display_batch_df["Risk Score (%)"] = (display_batch_df["Phishing probability"] * 100).round(1)

        filtered_result = display_batch_df[["URL", "Prediction", "Risk Score (%)"]]
        if filter_choice == "🚨 Phishing Only":
            filtered_result = filtered_result[filtered_result["Prediction"] == "Phishing"]
        elif filter_choice == "🛡️ Legitimate Only":
            filtered_result = filtered_result[filtered_result["Prediction"] == "Legitimate"]

        st.dataframe(
            filtered_result,
            hide_index=True,
            use_container_width=True,
            column_config={
                "URL": st.column_config.TextColumn("Target URL", width="large"),
                "Prediction": st.column_config.TextColumn("Classification"),
                "Risk Score (%)": st.column_config.ProgressColumn(
                    "Threat Probability",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
            }
        )


# model comparison tab
with tab_models:
    st.markdown("<h3 class='pop-heading'>📊 Model Performance & Comparative Benchmarks</h3>", unsafe_allow_html=True)
    st.markdown("Examine the statistical validation metrics across all 4 machine learning architectures evaluated on the test partition.")

    results_df = get_results()

    if results_df is not None:
        best_row = results_df[results_df["Model"] == best_model].iloc[0]

        top_col1, top_col2, top_col3, top_col4 = st.columns(4)
        with top_col1:
            st.markdown(f"""
                <div class="pop-tile">
                    <div class="pop-tile-val" style="font-size: 1.35rem; color: #8B5CF6; padding: 4px 0;">{best_model}</div>
                    <div class="pop-tile-lbl">⭐ Champion Algorithm</div>
                </div>
            """, unsafe_allow_html=True)

        with top_col2:
            st.markdown(f"""
                <div class="pop-tile">
                    <div class="pop-tile-val" style="color: #10B981;">{best_row['Accuracy']:.2%}</div>
                    <div class="pop-tile-lbl">Overall Accuracy</div>
                </div>
            """, unsafe_allow_html=True)

        with top_col3:
            st.markdown(f"""
                <div class="pop-tile">
                    <div class="pop-tile-val" style="color: #F472B6;">{best_row['F1']:.4f}</div>
                    <div class="pop-tile-lbl">Top F1-Score</div>
                </div>
            """, unsafe_allow_html=True)

        with top_col4:
            st.markdown(f"""
                <div class="pop-tile">
                    <div class="pop-tile-val" style="color: #FBBF24;">{best_row['ROC-AUC']:.4f}</div>
                    <div class="pop-tile-lbl">ROC-AUC Discriminator</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        st.markdown("<h4 class='pop-heading'>🏆 Benchmark Leaderboard</h4>", unsafe_allow_html=True)
        formatted_df = results_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")

        st.dataframe(
            formatted_df,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 class='pop-heading'>📈 Visual Performance Evidence</h4>", unsafe_allow_html=True)

    subtab1, subtab2, subtab3 = st.tabs(["📊 Model Comparison", "🎯 Confusion Matrix", "🔍 Global Feature Importance"])

    with subtab1:
        if os.path.exists("models/model_comparison.png"):
            st.image("models/model_comparison.png", use_container_width=True, caption="Multi-Metric Comparison Across Algorithms on Test Partition")
        else:
            st.info("Chart artifact `model_comparison.png` not found in `models/`.")

    with subtab2:
        if os.path.exists("models/confusion_matrix.png"):
            st.image("models/confusion_matrix.png", use_container_width=True, caption=f"Confusion Matrix for Champion Model ({best_model})")
        else:
            st.info("Chart artifact `confusion_matrix.png` not found in `models/`.")

    with subtab3:
        if os.path.exists("models/top_features.png"):
            st.image("models/top_features.png", use_container_width=True, caption=f"Top 10 Global Features Ranked by Importance ({best_model})")
        else:
            st.info("Chart artifact `top_features.png` not found in `models/`.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.expander("🎈 Technical Architecture & Methodology", expanded=False):
        st.markdown("""
        ### Feature Engineering & Modeling Strategy
        - **Lexical Analysis**: Measures domain length, subdomain depth, hyphen occurrences, and numeric digit ratios to identify spoofing attempts without requiring slow active network crawls.
        - **Security Heuristics**: Detects unencrypted HTTP, direct IP addresses (e.g. `192.168.10.5`), high-risk TLDs (`.xyz`, `.tk`, `.top`), URL shorteners, and credential phishing keywords (`login`, `verify`, `bank`).
        - **Ensemble Dominance**: Tree ensembles like **XGBoost** and **Random Forest** consistently outperform standard linear models because they effectively capture non-linear conjunctions (e.g., an unencrypted HTTP URL combined with a sensitive keyword and a high-risk TLD).
        """)