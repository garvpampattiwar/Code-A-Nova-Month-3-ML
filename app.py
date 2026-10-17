"""Streamlit Web Application: Email Spam Detection Engine.

Cohesive, high-contrast cybersecurity SaaS interface for message spam classification.
Uses scikit-learn TF-IDF + Multinomial Naive Bayes pipeline trained on benchmark data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from PIL import Image

from src.predict import predict_message

# Configuration & Paths
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
DATASET_SUMMARY_PATH = REPORTS_DIR / "dataset_summary.json"
CM_IMAGE_PATH = REPORTS_DIR / "confusion_matrix.png"

st.set_page_config(
    page_title="SentinelSpam | Email & Message Security Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Refined Cybersecurity SaaS Theme (Restrained Blue, Deep Navy Surface)
st.markdown(
    """
    <style>
    /* Global Reset & Background */
    .stApp {
        background-color: #080d1a !important;
        color: #e2e8f0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    /* Constrain Main Container Width for Centered SaaS Focus */
    .block-container {
        max-width: 940px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        margin: 0 auto !important;
    }

    /* Top Brand & Header Banner */
    .brand-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #38bdf8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }
    .brand-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #38bdf8;
        box-shadow: 0 0 8px #38bdf8;
    }
    .app-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
    }
    .app-subtitle {
        font-size: 14px;
        color: #94a3b8;
        line-height: 1.5;
        margin: 0 0 24px 0;
    }

    /* Section Headings */
    .section-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
        margin-top: 24px;
        margin-bottom: 10px;
    }

    /* Primary Analysis Panel Card */
    .analysis-panel {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #090e1a !important;
        border: 1px solid #26334d !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        font-family: inherit !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    /* Verdict Result Cards */
    .verdict-card {
        border-radius: 8px;
        padding: 18px 20px;
        margin-top: 18px;
        margin-bottom: 18px;
    }
    .verdict-spam {
        background: rgba(244, 63, 94, 0.08);
        border: 1px solid rgba(244, 63, 94, 0.35);
        border-left: 4px solid #f43f5e;
    }
    .verdict-ham {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 4px solid #10b981;
    }
    .verdict-status {
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .status-spam-text { color: #f43f5e; }
    .status-ham-text { color: #10b981; }
    .verdict-detail {
        font-size: 13px;
        color: #cbd5e1;
        margin: 0;
        line-height: 1.5;
    }

    /* Probability Bar & Meter */
    .prob-container {
        background-color: #090e1a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 14px 0;
    }
    .prob-row {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 6px;
    }

    /* Token Chips */
    .token-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
        gap: 10px;
        margin-top: 12px;
    }
    .token-chip {
        background-color: #090e1a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 12px;
    }
    .token-name {
        font-size: 13px;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .token-meta {
        font-size: 11px;
        display: flex;
        justify-content: space-between;
    }

    /* Stat Cards for Metrics */
    .kpi-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #38bdf8;
    }
    .kpi-label {
        font-size: 11px;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Sleek Comparison Table */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin: 14px 0;
    }
    .comparison-table th {
        background-color: #0f172a;
        color: #94a3b8;
        font-weight: 600;
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid #1e293b;
    }
    .comparison-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #131d33;
        color: #e2e8f0;
    }
    .comparison-table tr:hover {
        background-color: #0d1526;
    }

    /* Tab Header Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 6px !important;
        color: #94a3b8 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }

    /* Disclaimer Callout */
    .disclaimer-box {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 12px;
        color: #94a3b8;
        line-height: 1.5;
        margin-top: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics_report() -> Dict[str, Any]:
    """Load actual model evaluation metrics from reports/metrics.json."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_dataset_summary() -> Dict[str, Any]:
    """Load verified dataset summary from reports/dataset_summary.json."""
    if DATASET_SUMMARY_PATH.exists():
        with open(DATASET_SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


metrics_data = load_metrics_report()
dataset_data = load_dataset_summary()

# Pre-set Test Samples for Quick Validation
PRESET_SAMPLES = {
    "Select a pre-loaded message sample...": "",
    "🚨 Phishing / Fake Prize Alert": (
        "URGENT NOTICE! You have been selected as the guaranteed winner of a £1,000 cash prize "
        "or Apple iPhone 15 Pro. Claim immediately at http://award-claim-center.biz/verify before expires. "
        "Call 08718726270 immediately. Text STOP to cancel."
    ),
    "🚨 Corporate Quota Exceeded Scam": (
        "SECURITY ALERT: Your corporate email mailbox has exceeded its storage quota and will be suspended "
        "within 24 hours. Verify your account password and restore access at http://secure-auth-login-portal.com/update "
        "to prevent permanent data termination."
    ),
    "✅ Team Meeting Schedule": (
        "Hi Alex, hope your week is off to a productive start. Are we still confirmed for our sprint "
        "planning sync tomorrow at 10:30 AM in Conference Room B? Let me know if you need to reschedule."
    ),
    "✅ Order Confirmation Receipt": (
        "Hi Sarah, thank you for your recent order #89421. Your payment of $42.50 has been processed successfully. "
        "We are preparing your package and tracking details will be sent once it ships. Thanks for shopping with us!"
    ),
}

# --- HEADER SECTION ---
st.markdown(
    """
    <div class="brand-pill">
        <span class="brand-dot"></span> Sentinel // ML Security Engine
    </div>
    <h1 class="app-title">Email & Message Spam Detection</h1>
    <p class="app-subtitle">
        Natural language processing pipeline trained with TF-IDF n-gram feature extraction and a calibrated 
        Multinomial Naive Bayes classifier to identify unsolicited, fraudulent, and phishing communications.
    </p>
    """,
    unsafe_allow_html=True,
)

# --- QUICK SAMPLE SELECTOR ---
selected_sample_label = st.selectbox(
    "Quick-Test with Pre-loaded Templates:",
    options=list(PRESET_SAMPLES.keys()),
    index=0,
    label_visibility="collapsed",
)

default_text = PRESET_SAMPLES[selected_sample_label]

# --- MAIN ANALYSIS FORM ---
with st.form("spam_analysis_form", clear_on_submit=False):
    input_text = st.text_area(
        "Message or Email Body:",
        value=default_text,
        height=170,
        placeholder="Paste complete email body or message text here for real-time risk assessment...",
        help="Input text is tokenized and evaluated against calibrated n-gram frequencies.",
    )

    col_btn, col_count = st.columns([1, 3])
    with col_btn:
        submitted = st.form_submit_button("Analyze Message", type="primary", use_container_width=True)
    with col_count:
        char_count = len(input_text.strip()) if input_text else 0
        word_count = len(input_text.strip().split()) if char_count > 0 else 0
        st.markdown(
            f"<div style='padding-top: 8px; font-size: 12px; color: #64748b;'>"
            f"Input Length: <strong style='color: #94a3b8;'>{char_count}</strong> characters &nbsp;·&nbsp; "
            f"<strong style='color: #94a3b8;'>{word_count}</strong> words</div>",
            unsafe_allow_html=True,
        )

# --- EXECUTION & VERDICT DISPLAY ---
if submitted:
    if not input_text or not input_text.strip():
        st.warning("Input Required: Please enter or paste message text to analyze.")
    else:
        result = predict_message(input_text)
        if not result["success"]:
            st.error(f"Analysis Failed: {result['error']}")
        else:
            is_spam = result["is_spam"]
            spam_prob = result["spam_probability"]
            ham_prob = result["ham_probability"]
            confidence = result["confidence_pct"]

            # Visual Verdict Card
            if is_spam:
                st.markdown(
                    f"""
                    <div class="verdict-card verdict-spam">
                        <div class="verdict-status status-spam-text">🚨 SPAM / PHISHING DETECTED</div>
                        <p class="verdict-detail">
                            The classifier detected significant spam vocabulary patterns, promotional cues, or suspicious structures.
                            Estimated Spam Posterior Probability: <strong style="color: #f87171;">{spam_prob * 100:.2f}%</strong> 
                            (Confidence: {confidence:.2f}%).
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-card verdict-ham">
                        <div class="verdict-status status-ham-text">✅ LEGITIMATE (NOT SPAM)</div>
                        <p class="verdict-detail">
                            No anomalous spam indicators or high-weight phishing markers were detected in the text.
                            Estimated Legitimate Probability: <strong style="color: #34d399;">{ham_prob * 100:.2f}%</strong>
                            (Confidence: {confidence:.2f}%).
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Calibrated Probability Meter
            st.markdown(
                f"""
                <div class="prob-container">
                    <div class="prob-row">
                        <span><strong>Spam Probability:</strong> {spam_prob * 100:.2f}%</span>
                        <span><strong>Legitimate Probability:</strong> {ham_prob * 100:.2f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(float(spam_prob))

            # Vocabulary Signals
            if result.get("top_signals"):
                st.markdown("<div class='section-title'>Key Detected Lexical Signals</div>", unsafe_allow_html=True)
                chips_html = '<div class="token-grid">'
                for signal in result["top_signals"][:6]:
                    badge_color = "#f43f5e" if signal["spam_tendency"] == "Spam" else "#10b981"
                    chips_html += f"""
                    <div class="token-chip">
                        <div class="token-name">{signal['word']}</div>
                        <div class="token-meta">
                            <span style="color: {badge_color};">{signal['spam_tendency']}</span>
                            <span style="color: #64748b;">TF-IDF: {signal['tfidf_weight']}</span>
                        </div>
                    </div>
                    """
                chips_html += '</div>'
                st.markdown(chips_html, unsafe_allow_html=True)

            # Responsible AI Disclaimer
            st.markdown(
                """
                <div class="disclaimer-box">
                    <strong>Notice:</strong> Predictions are algorithmic estimates computed via lexical n-gram 
                    frequencies. This system does not inspect mail server headers (SPF/DKIM/DMARC) or attachments.
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)

# --- SYSTEM TELEMETRY & DOCUMENTATION TABS ---
st.markdown("<div class='section-title'>System Insights & Documentation</div>", unsafe_allow_html=True)
tab_metrics, tab_compare, tab_pipeline, tab_dataset = st.tabs(
    ["Model Evaluation", "Model Comparison", "How It Works", "Dataset & Limitations"]
)

# TAB 1: MODEL EVALUATION & CONFUSION MATRIX
with tab_metrics:
    st.markdown("#### Held-Out Test Set Performance ($N = 1,032$)")
    st.caption("Evaluated on a strictly held-out 20% stratified test partition unseen during pipeline training.")

    if metrics_data:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-value">{metrics_data.get('accuracy', 0)*100:.2f}%</div>
                    <div class="kpi-label">Accuracy</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-value">{metrics_data.get('precision_spam', 0)*100:.2f}%</div>
                    <div class="kpi-label">Spam Precision</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-value">{metrics_data.get('recall_spam', 0)*100:.2f}%</div>
                    <div class="kpi-label">Spam Recall</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-value">{metrics_data.get('f1_spam', 0)*100:.2f}%</div>
                    <div class="kpi-label">Spam F1-Score</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-value">{metrics_data.get('roc_auc', 0):.4f}</div>
                    <div class="kpi-label">ROC-AUC</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        col_cm_text, col_cm_img = st.columns([1, 1])

        with col_cm_text:
            cm = metrics_data.get("confusion_matrix", {})
            st.markdown("##### Confusion Matrix Breakdown")
            st.markdown(
                f"""
                - **True Negatives (Legitimate correctly identified):** `{cm.get('true_negatives_ham', 0)}`
                - **True Positives (Spam correctly caught):** `{cm.get('true_positives_spam', 0)}`
                - **False Positives (Legitimate flagged as Spam):** `{cm.get('false_positives_ham_as_spam', 0)}` *(Precision-critical)*
                - **False Negatives (Spam missed as Legitimate):** `{cm.get('false_negatives_spam_as_ham', 0)}`
                
                **Precision Rationale:** A high precision rate (98.29%) ensures legitimate business correspondence 
                is almost never discarded into the junk directory, protecting against communication disruption.
                """
            )

        with col_cm_img:
            if CM_IMAGE_PATH.exists():
                st.image(str(CM_IMAGE_PATH), caption="Confusion Matrix (Held-out Test Split)", width=380)
    else:
        st.info("Metrics report pending model evaluation.")

# TAB 2: MODEL COMPARISON (5-FOLD CROSS-VALIDATION)
with tab_compare:
    st.markdown("#### 5-Fold Stratified Cross-Validation on Training Split")
    st.caption("Both candidate pipelines were evaluated across 5 stratified folds before selecting the final architecture:")

    if metrics_data and "cross_validation_comparison" in metrics_data:
        cv_data = metrics_data["cross_validation_comparison"]
        
        table_html = """
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Model Candidate</th>
                    <th>CV Accuracy</th>
                    <th>CV Precision</th>
                    <th>CV Recall</th>
                    <th>CV F1-Score</th>
                </tr>
            </thead>
            <tbody>
        """
        for model_name, stats in cv_data.items():
            table_html += f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td>{stats['cv_accuracy_mean']*100:.2f}%</td>
                    <td>{stats['cv_precision_mean']*100:.2f}%</td>
                    <td>{stats['cv_recall_mean']*100:.2f}%</td>
                    <td>{stats['cv_f1_mean']*100:.2f}%</td>
                </tr>
            """
        table_html += """
            </tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        st.info(
            "**Selection Decision:** Multinomial Naive Bayes was selected over Balanced Logistic Regression "
            "because of its higher spam precision (97.27% vs. 94.03%). In spam filtering, minimizing false positive "
            "alerts on legitimate messages is paramount."
        )

# TAB 3: HOW IT WORKS
with tab_pipeline:
    st.markdown("#### End-to-End Machine Learning Pipeline")
    st.markdown(
        """
        1. **Text Normalization & Token Preservation:**
           - Web URLs $\\rightarrow$ `http_token`
           - Currency symbols (`$`, `£`, `€`, `¥`) $\\rightarrow$ `currency_token`
           - Numeric sequences and phone numbers $\\rightarrow$ `number_token`
           - Consecutive exclamation sequences $\\rightarrow$ `excl_token`
           - Lowercase folding and whitespace contraction
        2. **TF-IDF Feature Vectorization:**
           - Sublinear Term Frequency (`1 + log(tf)`) dampens extreme word repetitions.
           - Unigrams and bigrams (`ngram_range=(1, 2)`) capture short promotional phrases.
           - Frequency bounds (`min_df=2`, `max_df=0.95`) strip rare typos and ubiquitous stopwords.
        3. **Classification & Risk Calibration:**
           - Multinomial Naive Bayes applies Bayes' Theorem with Laplace smoothing (`alpha=0.2`).
           - Raw class log-likelihoods are converted to calibrated posterior probability scores.
        """
    )

# TAB 4: DATASET & LIMITATIONS
with tab_dataset:
    st.markdown("#### Benchmark Dataset Disclosure & System Limitations")
    if dataset_data:
        st.markdown(
            f"""
            - **Benchmark Dataset:** `{dataset_data.get('dataset_name', 'UCI SMS Spam Collection')}`
            - **License:** `{dataset_data.get('dataset_license', 'CC BY 4.0')}`
            - **Cleaned Records:** `{dataset_data.get('deduplicated_total_records', '5,158')}` unique communications
            - **Class Distribution:** `{dataset_data.get('label_distribution', {}).get('ham', {}).get('percentage', 87.55)}%` Legitimate, `{dataset_data.get('label_distribution', {}).get('spam', {}).get('percentage', 12.45)}%` Spam
            """
        )

    st.warning(
        "**Important Dataset & Domain Disclosure:**\n\n"
        "• This model is trained and evaluated on the **UCI SMS Spam Collection** benchmark.\n\n"
        "• **Not a Real Corporate Email Dataset:** The model has **not** been validated on an actual corporate email inbox "
        "corpus containing full RFC mail headers, MIME attachments, or enterprise communication patterns.\n\n"
        "• **Generalization Scope:** While lexical spam signals (monetary lures, urgency, phishing URLs) translate effectively "
        "to email message bodies, production email defense requires multi-layered inspection (SPF/DKIM/DMARC authentication, "
        "sender domain reputation, and attachment sandboxing)."
    )
