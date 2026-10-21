import os
import re
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import best_model_name, load_model, predict_urls

# ==============================================================================
# PLAYFUL GEOMETRIC DESIGN SYSTEM - STREAMLIT CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="PhishPop! | Playful Phishing URL Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fail-safe check for model artifacts
if not os.path.exists("models/best_model.txt"):
    st.error("🚨 No trained model found. Please run `python -m src.train` first.")
    st.stop()


# ==============================================================================
# CACHED DATA & MODEL LOADERS
# ==============================================================================
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

# Manage active model in session state
if "active_model" not in st.session_state:
    st.session_state.active_model = best_model

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

# ==============================================================================
# CSS: PLAYFUL GEOMETRIC DESIGN TOKEN INJECTION (PLAIN BACKGROUND)
# ==============================================================================
st.markdown("""
<style>
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

    /* 1. Clean Plain Warm Cream Paper (#FFFDF5) */
    .stApp {
        background-color: #FFFDF5 !important;
        background-image: none !important;
        color: #1E293B;
    }

    /* 2. Plain Sidebar with 2.5px solid border */
    section[data-testid="stSidebar"] {
        background-color: #FFFDF5 !important;
        background-image: none !important;
        border-right: 2.5px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.9rem;
    }

    /* 3. Sticker Cards */
    .sticker-card {
        background: #FFFFFF;
        border: 2px solid #1E293B;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 5px 5px 0px 0px #1E293B;
        transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
    }
    .sticker-card:hover {
        transform: translate(-2px, -2px);
        box-shadow: 7px 7px 0px 0px #1E293B;
    }

    /* Hero Banner */
    .hero-banner {
        background: #FFFFFF;
        border: 2.5px solid #1E293B;
        border-radius: 22px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 6px 6px 0px 0px #8B5CF6, 6px 6px 0px 2.5px #1E293B;
        position: relative;
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 900;
        color: #1E293B;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #475569;
        font-weight: 500;
        max-width: 820px;
        line-height: 1.5;
    }

    /* Badges */
    .pop-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        border: 2px solid #1E293B;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 0.82rem;
        box-shadow: 2.5px 2.5px 0px 0px #1E293B;
        letter-spacing: 0.02em;
    }
    .badge-violet { background: #8B5CF6; color: #FFFFFF; }
    .badge-pink   { background: #F472B6; color: #FFFFFF; }
    .badge-yellow { background: #FBBF24; color: #1E293B; }
    .badge-mint   { background: #34D399; color: #1E293B; }
    .badge-white  { background: #FFFFFF; color: #1E293B; }

    /* Rotated Sticker Tape */
    .tape-tag {
        display: inline-block;
        padding: 4px 12px;
        background: #FBBF24;
        color: #1E293B;
        border: 2px solid #1E293B;
        font-family: 'Outfit', sans-serif;
        font-weight: 900;
        font-size: 0.78rem;
        text-transform: uppercase;
        transform: rotate(-2deg);
        box-shadow: 2.5px 2.5px 0px 0px #1E293B;
        border-radius: 6px;
    }

    /* Verdict Cards */
    .verdict-phish {
        background: #FFF1F2;
        border: 2.5px solid #1E293B;
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 6px 6px 0px 0px #F472B6, 6px 6px 0px 2.5px #1E293B;
        position: relative;
    }
    .verdict-safe {
        background: #ECFDF5;
        border: 2.5px solid #1E293B;
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 6px 6px 0px 0px #34D399, 6px 6px 0px 2.5px #1E293B;
        position: relative;
    }

    /* Tactical Mini Tiles */
    .pop-tile {
        background: #FFFFFF;
        border: 2px solid #1E293B;
        border-radius: 16px;
        padding: 16px 14px;
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
        font-size: 1.4rem;
        font-weight: 900;
        line-height: 1.2;
        margin: 4px 0 2px 0;
    }
    .pop-tile-lbl {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }

    /* URL Pill */
    .pop-url-pill {
        font-family: 'JetBrains Mono', monospace;
        background: #F8FAFC;
        border: 2px solid #1E293B;
        border-radius: 10px;
        padding: 10px 14px;
        color: #0F172A;
        font-size: 0.92rem;
        word-break: break-all;
        margin: 10px 0 14px 0;
        box-shadow: 2px 2px 0px 0px #E2E8F0;
    }

    /* Candy Progress Meter */
    .candy-meter-bg {
        width: 100%;
        height: 16px;
        background: #E2E8F0;
        border: 2px solid #1E293B;
        border-radius: 9999px;
        overflow: hidden;
        margin-top: 10px;
    }
    .candy-meter-fill-red {
        height: 100%;
        background: repeating-linear-gradient(45deg, #EF4444, #EF4444 10px, #F87171 10px, #F87171 20px);
        border-radius: 9999px;
        border-right: 2px solid #1E293B;
    }
    .candy-meter-fill-green {
        height: 100%;
        background: repeating-linear-gradient(45deg, #10B981, #10B981 10px, #34D399 10px, #34D399 20px);
        border-radius: 9999px;
        border-right: 2px solid #1E293B;
    }

    /* Tabs Navigation */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        padding: 10px 22px !important;
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

    /* Candy Buttons */
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.92rem !important;
        border: 2px solid #1E293B !important;
        border-radius: 9999px !important;
        padding: 9px 20px !important;
        box-shadow: 3.5px 3.5px 0px 0px #1E293B !important;
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
        box-shadow: 5px 5px 0px 0px #1E293B !important;
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

    /* Primary button: Vivid Violet */
    div.stButton > button[kind="primary"] {
        background-color: #8B5CF6 !important;
        color: #FFFFFF !important;
        border: 2px solid #1E293B !important;
        box-shadow: 3.5px 3.5px 0px 0px #1E293B !important;
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
        box-shadow: 5px 5px 0px 0px #1E293B !important;
    }

    /* High-contrast Input */
    div[data-testid="stTextInputRootElement"],
    div[data-testid="stTextInput"] > div,
    div[data-testid="stTextInput"] > div > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 2px solid #1E293B !important;
        border-radius: 12px !important;
        box-shadow: 3.5px 3.5px 0px 0px #CBD5E1 !important;
    }
    div[data-baseweb="base-input"]:focus-within,
    div[data-baseweb="input"]:focus-within {
        border-color: #8B5CF6 !important;
        box-shadow: 3.5px 3.5px 0px 0px #8B5CF6 !important;
    }
    input[type="text"],
    input[data-testid="stTextInput"],
    div[data-baseweb="input"] input {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        -webkit-text-fill-color: #1E293B !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        border: 2px solid #1E293B !important;
        border-radius: 12px !important;
        background: #FFFFFF !important;
        box-shadow: 3px 3px 0px 0px #1E293B !important;
    }

    /* Compact framed visual chart containers */
    .chart-frame {
        background: #FFFFFF;
        border: 2px solid #1E293B;
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 5px 5px 0px 0px #1E293B;
        margin: 12px auto 20px auto;
        text-align: center;
    }

    /* Dataframe wrapper with hard shadow */
    div[data-testid="stDataFrame"] {
        border: 2px solid #1E293B !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: 5px 5px 0px 0px #1E293B !important;
        background: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# COMPACT PLAYFUL CHART GENERATOR
# ==============================================================================
def render_compact_model_chart(results_df, metric_filter="All Metrics"):
    """Renders a clean, compact, beautifully proportioned Playful Geometric chart."""
    fig, ax = plt.subplots(figsize=(5.6, 2.7), dpi=140, facecolor="#FFFDF5")
    ax.set_facecolor("#FFFFFF")

    if metric_filter != "All Metrics" and metric_filter in results_df.columns:
        models = results_df["Model"].tolist()
        values = (results_df[metric_filter] * 100).tolist()
        colors = ["#8B5CF6" if m == best_model else "#F472B6" for m in models]
        bars = ax.bar(models, values, color=colors, edgecolor="#1E293B", linewidth=1.5, width=0.45)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.45, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1E293B')
        min_val = min(values)
        ax.set_ylim(max(0, min_val - 8), 104)
        ax.set_ylabel(f"{metric_filter} (%)", fontweight='bold', fontsize=8, color='#1E293B')
        ax.set_title(f"Algorithm Benchmark — {metric_filter}", fontsize=9.5, fontweight='bold', pad=8, color='#1E293B')
    else:
        metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        colors = ["#8B5CF6", "#F472B6", "#FBBF24", "#34D399", "#6366F1"]
        x = list(range(len(results_df)))
        width = 0.15
        for i, m in enumerate(metrics):
            ax.bar([p + width * i for p in x], results_df[m] * 100, width=width, label=m, color=colors[i], edgecolor="#1E293B", linewidth=1.1)
        ax.set_xticks([p + width * 2 for p in x])
        ax.set_xticklabels(results_df["Model"], fontweight='bold', fontsize=8, color='#1E293B')
        ax.set_ylim(82, 104)
        ax.set_ylabel("Score (%)", fontweight='bold', fontsize=8, color='#1E293B')
        ax.set_title("Multi-Metric Test Benchmark Comparison", fontsize=9.5, fontweight='bold', pad=8, color='#1E293B')
        ax.legend(loc='lower right', framealpha=0.95, edgecolor='#1E293B', fontsize=6.8, ncol=3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#1E293B")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color("#1E293B")
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(axis='y', linestyle=':', alpha=0.3, color='#1E293B')
    plt.tight_layout()
    return fig


def render_compact_feature_importance(importance_df, target_model="XGBoost"):
    """Renders a compact, attractive horizontal bar chart of top 8 features."""
    fig, ax = plt.subplots(figsize=(5.4, 2.7), dpi=140, facecolor="#FFFDF5")
    ax.set_facecolor("#FFFFFF")

    col_name = target_model if target_model in importance_df.columns else best_model
    sorted_df = importance_df.sort_values(by=col_name, ascending=True).tail(8)

    raw_names = sorted_df["feature"].tolist()
    friendly_names = [FEATURE_META.get(f, {}).get("name", f) for f in raw_names]
    weights = sorted_df[col_name].tolist()

    palette = ["#8B5CF6", "#F472B6", "#FBBF24", "#34D399"]
    bar_colors = [palette[i % len(palette)] for i in range(len(sorted_df))]

    bars = ax.barh(range(len(sorted_df)), weights, color=bar_colors, edgecolor="#1E293B", linewidth=1.4, height=0.62)
    for bar, w in zip(bars, weights):
        ax.text(w + 0.008, bar.get_y() + bar.get_height() / 2.0, f"{w:.3f}", ha='left', va='center', fontsize=7.5, fontweight='bold', color='#1E293B')

    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(friendly_names, fontsize=8, color="#1E293B", fontweight="600")
    ax.set_xlabel("Relative Feature Weight", fontsize=8, color="#475569", fontweight="700")
    ax.set_title(f"Top 8 Critical Signals — {target_model}", fontsize=9.5, fontweight='bold', pad=8, color='#1E293B')

    max_w = max(weights) if weights else 0.5
    ax.set_xlim(0, max_w * 1.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#1E293B")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color("#1E293B")
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(axis='x', linestyle=':', alpha=0.3, color='#1E293B')
    plt.tight_layout()
    return fig


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
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
    sidebar_model = st.selectbox(
        "Active Algorithm",
        MODEL_NAMES,
        index=MODEL_NAMES.index(st.session_state.active_model) if st.session_state.active_model in MODEL_NAMES else 0,
        help="Select which algorithm scores URL threat probabilities."
    )
    if sidebar_model != st.session_state.active_model:
        st.session_state.active_model = sidebar_model
        st.rerun()

    is_champion = (st.session_state.active_model == best_model)
    st.markdown(f"""
        <div style="background: #FFFFFF; border: 2px solid #1E293B; border-radius: 14px; padding: 14px; box-shadow: 4px 4px 0px 0px {'#F472B6' if is_champion else '#E2E8F0'}; margin-top: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.75rem; font-weight: 800; text-transform: uppercase; color: #64748B;">ACTIVE CLASSIFIER</span>
                <span class="pop-badge {'badge-pink' if is_champion else 'badge-yellow'}" style="padding: 2px 10px; font-size: 0.75rem; box-shadow: 2px 2px 0px 0px #1E293B;">
                    {'⭐ BEST PICK' if is_champion else 'TESTED'}
                </span>
            </div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.15rem; font-weight: 800; color: #1E293B; margin-top: 6px;">{st.session_state.active_model}</div>
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

# Active model instance
model = get_model(st.session_state.active_model)


# ==============================================================================
# HERO BANNER
# ==============================================================================
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
                <span class="pop-badge badge-violet">ACTIVE: {st.session_state.active_model.upper()}</span>
                <span class="pop-badge badge-mint">🛡️ 15 GEOMETRIC VECTORS</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


# ==============================================================================
# TOP-LEVEL TABS
# ==============================================================================
tab_single, tab_batch, tab_models = st.tabs([
    "🔍 Single URL Scanner",
    "📂 Batch URL Inspector",
    "📊 Model Performance & Benchmarks"
])


# ==============================================================================
# TAB 1: SINGLE URL SCANNER
# ==============================================================================
with tab_single:
    col_hdr, col_mod_select = st.columns([3, 2])
    with col_hdr:
        st.markdown("<h3 class='pop-heading'>🔍 Single URL Inspection</h3>", unsafe_allow_html=True)
    with col_mod_select:
        st.markdown("<p style='font-size: 0.8rem; font-weight: 800; color: #64748B; text-transform: uppercase; margin-bottom: 2px;'>⚡ Prediction Model:</p>", unsafe_allow_html=True)
        chosen_model = st.selectbox(
            "Change Model",
            MODEL_NAMES,
            index=MODEL_NAMES.index(st.session_state.active_model),
            key="tab1_model_select",
            label_visibility="collapsed"
        )
        if chosen_model != st.session_state.active_model:
            st.session_state.active_model = chosen_model
            st.rerun()

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
                st.rerun()

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
        with st.spinner(f"Parsing vectors using {st.session_state.active_model}..."):
            result, features = predict_urls([active_url], model)
            label = result.loc[0, "Prediction"]
            prob = float(result.loc[0, "Phishing probability"])

        is_phishing = (label == "Phishing")
        verdict_class = "verdict-phish" if is_phishing else "verdict-safe"
        verdict_icon = "🚨" if is_phishing else "🎉"
        verdict_title = "MALICIOUS PHISHING DETECTED!" if is_phishing else "VERIFIED LEGITIMATE & SAFE!"
        badge_style = "badge-pink" if is_phishing else "badge-mint"

        # Verdict Card
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
                        <div style="font-size: 0.78rem; font-weight: 800; text-transform: uppercase; color: #64748B;">Phishing Probability ({st.session_state.active_model})</div>
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

        # ----------------------------------------------------------------------
        # MULTI-MODEL SIDE-BY-SIDE CONSENSUS CHECK
        # ----------------------------------------------------------------------
        with st.expander("🔬 Check Prediction Across All 4 Models (Model Consensus)", expanded=False):
            st.caption("Compare how each trained algorithm evaluates this specific URL side-by-side:")
            all_preds = []
            for m_name in MODEL_NAMES:
                m_obj = get_model(m_name)
                res_m, _ = predict_urls([active_url], m_obj)
                p_label = res_m.loc[0, "Prediction"]
                p_prob = float(res_m.loc[0, "Phishing probability"])
                all_preds.append({
                    "Algorithm": f"⭐ {m_name} (Best)" if m_name == best_model else m_name,
                    "Verdict": "🚨 Phishing" if p_label == "Phishing" else "🟢 Legitimate",
                    "Phishing Risk": f"{p_prob:.1%}",
                    "Raw Prob": p_prob,
                    "Model Key": m_name
                })

            consensus_df = pd.DataFrame(all_preds)
            phish_votes = sum(1 for p in all_preds if "Phishing" in p["Verdict"])
            agreement_pct = (phish_votes / len(MODEL_NAMES)) if phish_votes > 0 else 0

            # Consensus summary card
            st.markdown(f"""
                <div style="background: #FFFFFF; border: 2px solid #1E293B; border-radius: 14px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 3.5px 3.5px 0px 0px #1E293B; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 800; color: #1E293B;">
                            Consensus: {phish_votes} of 4 Algorithms vote Phishing
                        </div>
                        <div style="font-size: 0.82rem; color: #64748B;">
                            {'High model unanimity on threat severity.' if phish_votes in [0, 4] else 'Moderate model disagreement — inspect individual probability scores.'}
                        </div>
                    </div>
                    <span class="pop-badge {'badge-pink' if phish_votes >= 2 else 'badge-mint'}">
                        {'HIGH RISK CONSENSUS' if phish_votes >= 2 else 'LOW RISK CONSENSUS'}
                    </span>
                </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                consensus_df[["Algorithm", "Verdict", "Phishing Risk"]],
                hide_index=True,
                use_container_width=True
            )

        # 4 Signal Tiles
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

        # Feature Importance Chart and Table
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        chart_col, table_col = st.columns([1, 1])

        importance_df = get_importance()
        top_features = (
            importance_df[["feature", st.session_state.active_model]]
            .sort_values(st.session_state.active_model, ascending=False)
            .head(10)
            .rename(columns={st.session_state.active_model: "importance"})
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
            st.caption(f"Top 10 features contributing to the {st.session_state.active_model} prediction.")

            fig, ax = plt.subplots(figsize=(6, 4.0), dpi=130, facecolor="#FFFDF5")
            ax.set_facecolor("#FFFFFF")

            y_pos = range(len(top_features))
            weights = top_features["importance"].iloc[::-1]
            names = top_features["friendly_name"].iloc[::-1]

            palette = ["#8B5CF6", "#F472B6", "#FBBF24", "#34D399"]
            bar_colors = [palette[i % len(palette)] for i in range(len(top_features))]

            ax.barh(
                y_pos,
                weights,
                color=bar_colors,
                edgecolor="#1E293B",
                linewidth=1.8,
                height=0.68,
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9, color="#1E293B", fontweight="600")
            ax.set_xlabel("Relative Feature Weight", fontsize=9, color="#475569", fontweight="700")
            ax.tick_params(axis="x", colors="#1E293B", labelsize=8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#1E293B")
            ax.spines["left"].set_linewidth(1.8)
            ax.spines["bottom"].set_color("#1E293B")
            ax.spines["bottom"].set_linewidth(1.8)
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


# ==============================================================================
# TAB 2: BATCH URL INSPECTOR
# ==============================================================================
with tab_batch:
    st.markdown("<h3 class='pop-heading'>📂 Batch & URL Threat Scanner</h3>", unsafe_allow_html=True)
    st.markdown(f"Scan a single URL on demand or analyze bulk CSV files simultaneously using **{st.session_state.active_model}**:")

    # Mode Selector for Tab 2
    tab2_mode = st.radio(
        "Scanning Mode:",
        ["🔗 Quick Single URL Scan", "📂 Batch CSV File Upload"],
        horizontal=True,
        key="tab2_scan_mode"
    )

    file = None
    use_sample = False

    if tab2_mode == "🔗 Quick Single URL Scan":
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        q_col1, q_col2 = st.columns([4, 1])
        with q_col1:
            tab2_url = st.text_input(
                "Enter URL to analyze:",
                value="http://secure-login.paypal.com.verify-id.xyz/update?id=123",
                placeholder="https://example.com/login",
                key="tab2_single_url_input",
                label_visibility="collapsed"
            )
        with q_col2:
            tab2_scan_btn = st.button("🚀 SCAN URL", type="primary", key="tab2_single_scan_btn", use_container_width=True)

        if tab2_url.strip():
            with st.spinner(f"Evaluating URL with {st.session_state.active_model}..."):
                res_q, feat_q = predict_urls([tab2_url.strip()], model)
                lbl_q = res_q.loc[0, "Prediction"]
                prob_q = float(res_q.loc[0, "Phishing probability"])

            is_phish_q = (lbl_q == "Phishing")
            st.markdown(f"""
                <div class="{'verdict-phish' if is_phish_q else 'verdict-safe'}" style="margin-top: 14px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                        <div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.8rem;">{'🚨' if is_phish_q else '🟢'}</span>
                                <span class="pop-badge {'badge-pink' if is_phish_q else 'badge-mint'}">
                                    {lbl_q.upper()} VERDICT
                                </span>
                            </div>
                            <div class="pop-url-pill" style="margin-top: 8px;">{tab2_url.strip()}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-family: 'Outfit', sans-serif; font-size: 2.4rem; font-weight: 900; color: {'#EF4444' if is_phish_q else '#10B981'}; line-height: 1;">
                                {prob_q:.1%}
                            </div>
                            <div style="font-size: 0.78rem; font-weight: 800; color: #64748B; text-transform: uppercase;">Phishing Risk ({st.session_state.active_model})</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.info("💡 For full lexical anatomy, feature weights, and 4-model consensus comparison, visit the **Single URL Scanner** tab!")

        data = None
    else:
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

        with st.spinner(f"Analyzing {len(url_list)} URLs with {st.session_state.active_model}..."):
            result, _ = predict_urls(url_list, model)

        total_scanned = len(result)
        phishing_count = int((result["Prediction"] == "Phishing").sum())
        legitimate_count = total_scanned - phishing_count
        threat_rate = (phishing_count / total_scanned) if total_scanned > 0 else 0

        # KPI Summary Grid
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

        # Filters and download
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


# ==============================================================================
# TAB 3: MODEL PERFORMANCE & BENCHMARKS (REDESIGNED CHARTS & BEST MODEL ADVISOR)
# ==============================================================================
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

        # ----------------------------------------------------------------------
        # BEST MODEL ADVISOR / RECOMMENDATION WIZARD
        # ----------------------------------------------------------------------
        st.markdown("<h4 class='pop-heading'>🎯 Best Model Recommendation & Selection</h4>", unsafe_allow_html=True)
        st.caption("Select what matters most for your URL security needs to pick or apply the best algorithm:")

        advisor_cols = st.columns(4)
        model_recommendations = [
            ("🏆 Best Overall Balance", "XGBoost", "94.68% Acc, 0.9346 F1", "Best general-purpose classifier across all metrics.", "#8B5CF6"),
            ("🛡️ Zero False Positives", "Random Forest", "98.42% Precision", "Minimizes false alarms on legitimate business links.", "#34D399"),
            ("🚨 Maximum Catch Rate", "XGBoost", "89.01% Recall", "Catches the highest percentage of stealthy phishing links.", "#EF4444"),
            ("⚡ Fastest Inference", "Decision Tree", "72 KB Footprint", "Lightweight single tree with 94.62% test accuracy.", "#FBBF24"),
        ]

        for i, (title, rec_model, metric_val, desc, color) in enumerate(model_recommendations):
            with advisor_cols[i]:
                is_selected = (st.session_state.active_model == rec_model)
                st.markdown(f"""
                    <div style="background: #FFFFFF; border: 2px solid #1E293B; border-radius: 14px; padding: 14px; height: 165px; box-shadow: 4px 4px 0px 0px {'#8B5CF6' if is_selected else '#E2E8F0'}; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <div style="font-size: 0.8rem; font-weight: 800; color: {color}; text-transform: uppercase;">{title}</div>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 1.2rem; font-weight: 900; color: #1E293B; margin-top: 4px;">{rec_model}</div>
                            <div style="font-size: 0.78rem; font-weight: 700; color: #1E293B;">{metric_val}</div>
                            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px; line-height: 1.3;">{desc}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                if st.button(f"{'✓ Active' if is_selected else f'Use {rec_model}'}", key=f"btn_pick_rec_{i}", use_container_width=True):
                    st.session_state.active_model = rec_model
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Active Model Status Banner
        st.markdown(f"""
            <div style="background: #FFFFFF; border: 2px solid #1E293B; border-radius: 14px; padding: 14px 18px; margin-bottom: 14px; box-shadow: 3.5px 3.5px 0px 0px #1E293B; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.05rem; font-weight: 800; color: #1E293B;">
                        Currently Active Prediction Engine: <span style="color: #8B5CF6;">{st.session_state.active_model}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748B;">All URL scans in 'Single URL Scanner' and 'Batch Inspector' use this model.</div>
                </div>
                <span class="pop-badge {'badge-pink' if st.session_state.active_model == best_model else 'badge-yellow'}">
                    {'⭐ BEST MODEL (CHAMPION)' if st.session_state.active_model == best_model else 'ALTERNATIVE MODEL'}
                </span>
            </div>
        """, unsafe_allow_html=True)

        # Tabular Leaderboard
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

    # --------------------------------------------------------------------------
    # VISUAL EVIDENCE: COMPACT & ATTRACTIVE CHARTS
    # --------------------------------------------------------------------------
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 class='pop-heading'>📈 Visual Performance Evidence</h4>", unsafe_allow_html=True)

    subtab1, subtab2, subtab3 = st.tabs(["📊 Model Comparison", "🎯 Confusion Matrix", "🔍 Global Feature Importance"])

    with subtab1:
        st.markdown("<p style='font-size: 0.88rem; color: #475569; margin-bottom: 12px;'>Filter benchmark rankings across all 4 machine learning architectures evaluated on 47,074 test records:</p>", unsafe_allow_html=True)
        metric_col, _ = st.columns([2, 3])
        with metric_col:
            selected_chart_metric = st.selectbox(
                "Filter Metric:",
                options=["All Metrics", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                index=0,
                key="chart_metric_filter"
            )

        if results_df is not None:
            compact_fig = render_compact_model_chart(results_df, selected_chart_metric)
            c_l, c_chart, c_r = st.columns([1, 2.8, 1])
            with c_chart:
                st.pyplot(compact_fig)
                plt.close(compact_fig)

        st.markdown("""
            <div style="background: #FFFFFF; border: 2px solid #1E293B; border-radius: 12px; padding: 12px 16px; margin-top: 14px; box-shadow: 3px 3px 0px 0px #E2E8F0; font-size: 0.82rem; color: #334155;">
                💡 <b>Key Finding:</b> <b>XGBoost</b> achieves the highest overall accuracy (94.68%) and F1-score (0.9346), while <b>Random Forest</b> delivers the lowest false-alarm rate with 98.42% precision.
            </div>
        """, unsafe_allow_html=True)

    with subtab2:
        st.markdown("<p style='font-size: 0.88rem; color: #475569; margin-bottom: 12px;'>Inspect error distribution & misclassification rates on the 47,074 test partition:</p>", unsafe_allow_html=True)
        cm_sel_col, _ = st.columns([2, 3])
        with cm_sel_col:
            chosen_cm_model = st.selectbox(
                "Algorithm to inspect:",
                MODEL_NAMES,
                index=MODEL_NAMES.index(st.session_state.active_model) if st.session_state.active_model in MODEL_NAMES else 0,
                key="cm_model_choice"
            )

        if results_df is not None:
            cm_row = results_df[results_df["Model"] == chosen_cm_model].iloc[0]
            rec_v = float(cm_row["Recall"])
            prec_v = float(cm_row["Precision"])
            acc_v = float(cm_row["Accuracy"])

            tp_v = int(round(rec_v * 20104))
            fn_v = 20104 - tp_v
            tp_fp_v = int(round(tp_v / prec_v))
            fp_v = max(0, tp_fp_v - tp_v)
            tn_v = 26970 - fp_v

            # 2x2 Matrix Cards
            c_l, c_center, c_r = st.columns([1, 3.2, 1])
            with c_center:
                st.markdown(f"""
                    <div style="background: #FFFFFF; border: 2px solid #1E293B; border-radius: 16px; padding: 18px; box-shadow: 4px 4px 0px 0px #1E293B;">
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 800; color: #1E293B; margin-bottom: 12px; text-align: center;">
                            Confusion Matrix — {chosen_cm_model}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                            <div style="background: #F0FDF4; border: 2px solid #10B981; border-radius: 12px; padding: 12px;">
                                <div style="font-size: 0.72rem; font-weight: 800; color: #10B981; text-transform: uppercase;">🟢 True Negatives (Safe)</div>
                                <div style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 900; color: #1E293B; margin-top: 2px;">{tn_v:,}</div>
                                <div style="font-size: 0.75rem; font-weight: 700; color: #475569;">{tn_v/26970:.1%} of Benign URLs</div>
                                <div style="font-size: 0.7rem; color: #64748B;">Correctly passed</div>
                            </div>
                            <div style="background: #FFFBEB; border: 2px solid #FBBF24; border-radius: 12px; padding: 12px;">
                                <div style="font-size: 0.72rem; font-weight: 800; color: #D97706; text-transform: uppercase;">🟡 False Positives (Alarm)</div>
                                <div style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 900; color: #1E293B; margin-top: 2px;">{fp_v:,}</div>
                                <div style="font-size: 0.75rem; font-weight: 700; color: #475569;">{fp_v/26970:.1%} of Benign URLs</div>
                                <div style="font-size: 0.7rem; color: #64748B;">Safe flagged as phish</div>
                            </div>
                            <div style="background: #FEF2F2; border: 2px solid #F87171; border-radius: 12px; padding: 12px;">
                                <div style="font-size: 0.72rem; font-weight: 800; color: #EF4444; text-transform: uppercase;">🟠 False Negatives (Missed)</div>
                                <div style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 900; color: #1E293B; margin-top: 2px;">{fn_v:,}</div>
                                <div style="font-size: 0.75rem; font-weight: 700; color: #475569;">{fn_v/20104:.1%} of Phishing</div>
                                <div style="font-size: 0.7rem; color: #64748B;">Threats not caught</div>
                            </div>
                            <div style="background: #FAF5FF; border: 2px solid #8B5CF6; border-radius: 12px; padding: 12px;">
                                <div style="font-size: 0.72rem; font-weight: 800; color: #8B5CF6; text-transform: uppercase;">🟣 True Positives (Caught)</div>
                                <div style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 900; color: #1E293B; margin-top: 2px;">{tp_v:,}</div>
                                <div style="font-size: 0.75rem; font-weight: 700; color: #475569;">{tp_v/20104:.1%} of Phishing</div>
                                <div style="font-size: 0.7rem; color: #64748B;">Attacks neutralized</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Metric chips
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"<div class='pop-tile'><div class='pop-tile-val' style='color: #8B5CF6;'>{acc_v:.2%}</div><div class='pop-tile-lbl'>Accuracy</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='pop-tile'><div class='pop-tile-val' style='color: #10B981;'>{prec_v:.2%}</div><div class='pop-tile-lbl'>Precision</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='pop-tile'><div class='pop-tile-val' style='color: #EF4444;'>{rec_v:.2%}</div><div class='pop-tile-lbl'>Catch Rate (Recall)</div></div>", unsafe_allow_html=True)
            with m4:
                st.markdown(f"<div class='pop-tile'><div class='pop-tile-val' style='color: #34D399;'>{tn_v/26970:.2%}</div><div class='pop-tile-lbl'>Specificity</div></div>", unsafe_allow_html=True)

    with subtab3:
        st.markdown("<p style='font-size: 0.88rem; color: #475569; margin-bottom: 12px;'>Examine relative feature coefficients and split importance weights across architectures:</p>", unsafe_allow_html=True)
        feat_sel_col, _ = st.columns([2, 3])
        with feat_sel_col:
            chosen_feat_model = st.selectbox(
                "Rank Features For:",
                MODEL_NAMES,
                index=MODEL_NAMES.index(st.session_state.active_model) if st.session_state.active_model in MODEL_NAMES else 0,
                key="feat_model_choice"
            )

        importance_data = get_importance()
        if importance_data is not None:
            compact_feat_fig = render_compact_feature_importance(importance_data, chosen_feat_model)
            c_l, c_chart, c_r = st.columns([1, 2.8, 1])
            with c_chart:
                st.pyplot(compact_feat_fig)
                plt.close(compact_feat_fig)

            # Summary Table
            st.markdown("<h5 class='pop-heading' style='margin-top: 14px;'>📋 Feature Weights Ranking</h5>", unsafe_allow_html=True)
            sorted_feats = importance_data.sort_values(by=chosen_feat_model, ascending=False).copy()
            sorted_feats["Friendly Name"] = sorted_feats["feature"].apply(lambda f: FEATURE_META.get(f, {}).get("name", f))
            sorted_feats["Category"] = sorted_feats["feature"].apply(lambda f: FEATURE_META.get(f, {}).get("category", "General"))
            sorted_feats["Importance Score"] = sorted_feats[chosen_feat_model].apply(lambda w: f"{w:.4f}")

            st.dataframe(
                sorted_feats[["Friendly Name", "Category", "Importance Score"]],
                hide_index=True,
                use_container_width=True,
                height=260
            )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.expander("🎈 Technical Architecture & Methodology", expanded=False):
        st.markdown("""
        ### Feature Engineering & Modeling Strategy
        - **Lexical Analysis**: Measures domain length, subdomain depth, hyphen occurrences, and numeric digit ratios to identify spoofing attempts without requiring slow active network crawls.
        - **Security Heuristics**: Detects unencrypted HTTP, direct IP addresses (e.g. `192.168.10.5`), high-risk TLDs (`.xyz`, `.tk`, `.top`), URL shorteners, and credential phishing keywords (`login`, `verify`, `bank`).
        - **Ensemble Dominance**: Tree ensembles like **XGBoost** and **Random Forest** consistently outperform standard linear models because they effectively capture non-linear conjunctions (e.g., an unencrypted HTTP URL combined with a sensitive keyword and a high-risk TLD).
        """)