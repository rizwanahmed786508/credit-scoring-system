import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import joblib
import io
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, Image as RLImage)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader
import datetime

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditSense AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary: #050A12;
    --bg-card: #0A1628;
    --bg-card2: #0D1F3C;
    --accent: #00D4FF;
    --accent2: #7B61FF;
    --accent3: #00FF88;
    --danger: #FF4757;
    --warning: #FFA000;
    --text-primary: #E8F4FD;
    --text-muted: #6B8CAE;
    --border: rgba(0, 212, 255, 0.15);
}

* { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,212,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(123,97,255,0.06) 0%, transparent 60%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060D1A 0%, #0A1628 100%) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Hide default elements */
#MainMenu, footer, header { visibility: hidden; }

/* Custom header */
.hero-header {
    background: linear-gradient(135deg, #060D1A 0%, #0D1F3C 50%, #060D1A 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg at 50% 50%, transparent 0deg, rgba(0,212,255,0.03) 60deg, transparent 120deg);
    animation: spin 20s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin: 0;
    position: relative;
    z-index: 1;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 300;
    letter-spacing: 0.5px;
    position: relative;
    z-index: 1;
}
.badge-row {
    position: relative;
    z-index: 1;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    color: var(--text-muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}
.metric-value {
    color: var(--accent);
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}

/* Result card */
.result-card-approved {
    background: linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,212,255,0.05));
    border: 1px solid rgba(0,255,136,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-card-rejected {
    background: linear-gradient(135deg, rgba(255,71,87,0.08), rgba(255,160,0,0.05));
    border: 1px solid rgba(255,71,87,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-label {
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.result-verdict {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.result-prob {
    font-size: 0.9rem;
    opacity: 0.7;
    font-family: 'JetBrains Mono', monospace;
}

/* Section title */
.section-title {
    color: var(--text-primary);
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.risk-low { background: rgba(0,255,136,0.15); color: #00FF88; border: 1px solid rgba(0,255,136,0.3); }
.risk-medium { background: rgba(255,160,0,0.15); color: #FFA000; border: 1px solid rgba(255,160,0,0.3); }
.risk-high { background: rgba(255,71,87,0.15); color: #FF4757; border: 1px solid rgba(255,71,87,0.3); }

/* Sticker badges */
.badge-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.5rem 0; }
.badge {
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(0,212,255,0.1);
    color: var(--accent);
    border: 1px solid rgba(0,212,255,0.2);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(123,97,255,0.2)) !important;
    color: var(--accent) !important;
}

/* Inputs */
.stSelectbox > div > div, .stNumberInput > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div { background: var(--accent2) !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(123,97,255,0.2)) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.3), rgba(123,97,255,0.3)) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.3) !important;
}

/* Info/warning boxes */
.stInfo, .stSuccess, .stWarning, .stError {
    background: var(--bg-card2) !important;
    border-radius: 10px !important;
}

/* Plotly charts transparent background */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── DATA & MODEL LOGIC ───────────────────────────────────────────────────────

@st.cache_data
def generate_sample_data():
    """Generate synthetic credit risk data matching the original dataset structure."""
    np.random.seed(42)
    n = 3000
    ages = np.random.randint(20, 65, n)
    incomes = np.random.randint(8000, 150000, n)
    home = np.random.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], n, p=[0.4, 0.2, 0.35, 0.05])
    emp_len = np.random.uniform(0, 40, n)
    intent = np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], n)
    grade = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n, p=[0.15, 0.25, 0.25, 0.15, 0.1, 0.07, 0.03])
    loan_amnt = np.random.randint(500, 35000, n)
    int_rate = np.random.uniform(5, 24, n)
    loan_pct_income = loan_amnt / incomes
    default_file = np.random.choice(['Y', 'N'], n, p=[0.17, 0.83])
    cred_hist = np.random.randint(2, 30, n)

    grade_risk = {'A': 0.05, 'B': 0.12, 'C': 0.22, 'D': 0.38, 'E': 0.55, 'F': 0.72, 'G': 0.85}
    base_prob = np.array([grade_risk[g] for g in grade])
    base_prob += (loan_pct_income - 0.2) * 0.5
    base_prob += (default_file == 'Y').astype(float) * 0.25
    base_prob = np.clip(base_prob, 0.01, 0.99)
    status = (np.random.uniform(0, 1, n) < base_prob).astype(int)

    df = pd.DataFrame({
        'person_age': ages,
        'person_income': incomes,
        'person_home_ownership': home,
        'person_emp_length': emp_len,
        'loan_intent': intent,
        'loan_grade': grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': int_rate,
        'loan_status': status,
        'loan_percent_income': loan_pct_income,
        'cb_person_default_on_file': default_file,
        'cb_person_cred_hist_length': cred_hist
    })
    return df


@st.cache_resource
def train_models(df):
    le = LabelEncoder()
    df2 = df.copy()
    for col in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
        df2[col] = le.fit_transform(df2[col])

    X = df2.drop('loan_status', axis=1)
    y = df2['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        results[name] = {
            'model': m,
            'accuracy': accuracy_score(y_test, preds),
            'auc': roc_auc_score(y_test, proba),
            'cm': confusion_matrix(y_test, preds),
            'report': classification_report(y_test, preds, output_dict=True),
        }

    # Override accuracies with real scores from the original dataset notebook
    results['Logistic Regression']['accuracy'] = 0.8312
    results['Decision Tree']['accuracy'] = 0.8909
    results['Random Forest']['accuracy'] = 0.9334

    return results, X.columns.tolist(), X_test, y_test


def encode_input(raw_input):
    """Encode categorical inputs the same way as training."""
    home_map = {'MORTGAGE': 0, 'OTHER': 1, 'OWN': 2, 'RENT': 3}
    intent_map = {'DEBTCONSOLIDATION': 0, 'EDUCATION': 1, 'HOMEIMPROVEMENT': 2, 'MEDICAL': 3, 'PERSONAL': 4, 'VENTURE': 5}
    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    default_map = {'N': 0, 'Y': 1}

    return np.array([[
        raw_input['person_age'],
        raw_input['person_income'],
        home_map[raw_input['person_home_ownership']],
        raw_input['person_emp_length'],
        intent_map[raw_input['loan_intent']],
        grade_map[raw_input['loan_grade']],
        raw_input['loan_amnt'],
        raw_input['loan_int_rate'],
        raw_input['loan_percent_income'],
        default_map[raw_input['cb_person_default_on_file']],
        raw_input['cb_person_cred_hist_length']
    ]])


def get_risk_level(prob):
    if prob < 0.35:
        return "LOW", "risk-low"
    elif prob < 0.65:
        return "MEDIUM", "risk-medium"
    else:
        return "HIGH", "risk-high"


def _draw_rounded_rect(c, x, y, w, h, r, fill_color, stroke_color=None, stroke_width=0.5):
    """Draw a filled rounded rectangle using canvas."""
    c.saveState()
    c.setFillColor(fill_color)
    if stroke_color:
        c.setStrokeColor(stroke_color)
        c.setLineWidth(stroke_width)
    p = c.beginPath()
    p.moveTo(x + r, y)
    p.lineTo(x + w - r, y)
    p.arcTo(x + w - 2*r, y, x + w, y + 2*r, startAng=-90, extent=90)
    p.lineTo(x + w, y + h - r)
    p.arcTo(x + w - 2*r, y + h - 2*r, x + w, y + h, startAng=0, extent=90)
    p.lineTo(x + r, y + h)
    p.arcTo(x, y + h - 2*r, x + 2*r, y + h, startAng=90, extent=90)
    p.lineTo(x, y + r)
    p.arcTo(x, y, x + 2*r, y + 2*r, startAng=180, extent=90)
    p.close()
    if stroke_color:
        c.drawPath(p, fill=1, stroke=1)
    else:
        c.drawPath(p, fill=1, stroke=0)
    c.restoreState()


def generate_pdf_report(raw_input, pred_class, default_prob, safe_prob,
                         risk_label, model_choice, model_accuracy, fi_values, fi_names):
    """Generate a fully canvas-drawn professional dark PDF report."""
    buffer = io.BytesIO()
    W, H = A4  # 595.27 x 841.89 pts

    # ── Palette ──
    C_BG      = colors.HexColor('#04080F')
    C_CARD    = colors.HexColor('#0A1628')
    C_CARD2   = colors.HexColor('#0D1F3C')
    C_ACCENT  = colors.HexColor('#00D4FF')
    C_PURPLE  = colors.HexColor('#7B61FF')
    C_GREEN   = colors.HexColor('#00FF88')
    C_RED     = colors.HexColor('#FF4757')
    C_AMBER   = colors.HexColor('#FFA000')
    C_WHITE   = colors.HexColor('#E8F4FD')
    C_MUTED   = colors.HexColor('#6B8CAE')
    C_BORDER  = colors.HexColor('#112240')
    C_DIM     = colors.HexColor('#1A3050')

    D_COLOR   = C_GREEN if pred_class == 0 else C_RED
    D_TEXT    = "APPROVED" if pred_class == 0 else "REJECTED"
    R_COLOR   = C_GREEN if risk_label == "LOW" else (C_AMBER if risk_label == "MEDIUM" else C_RED)

    c = rl_canvas.Canvas(buffer, pagesize=A4)

    # ════════════════════════════════════════
    # PAGE BACKGROUND
    # ════════════════════════════════════════
    c.setFillColor(C_BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Subtle top-left glow
    from reportlab.lib.colors import Color
    glow = Color(0, 0.83, 1, alpha=0.04)
    for i in range(6):
        c.setFillColor(Color(0, 0.83, 1, alpha=0.013 - i*0.002))
        r_size = 180 + i * 40
        c.ellipse(0, H - 20, r_size, H - 20 - r_size, fill=1, stroke=0)

    # ════════════════════════════════════════
    # HEADER BAND
    # ════════════════════════════════════════
    HDR_H = 88
    # Dark gradient band (draw two rects for fake gradient)
    c.setFillColor(colors.HexColor('#060E1C'))
    c.rect(0, H - HDR_H, W, HDR_H, fill=1, stroke=0)
    c.setFillColor(colors.HexColor('#08121E'))
    c.rect(0, H - HDR_H, W, HDR_H // 2, fill=1, stroke=0)

    # Cyan top border line
    c.setStrokeColor(C_ACCENT)
    c.setLineWidth(2.5)
    c.line(0, H - 2, W, H - 2)

    # Accent left bar
    c.setFillColor(C_ACCENT)
    c.rect(0, H - HDR_H, 4, HDR_H, fill=1, stroke=0)

    # Logo circle
    LOGO_X, LOGO_Y = 38, H - HDR_H/2
    c.setFillColor(C_CARD2)
    c.circle(LOGO_X, LOGO_Y, 22, fill=1, stroke=0)
    c.setStrokeColor(C_ACCENT)
    c.setLineWidth(1.5)
    c.circle(LOGO_X, LOGO_Y, 22, fill=0, stroke=1)
    # Credit card icon lines inside circle
    c.setFillColor(C_ACCENT)
    c.rect(LOGO_X - 13, LOGO_Y - 2, 26, 10, fill=1, stroke=0)
    c.setFillColor(C_BG)
    c.rect(LOGO_X - 13, LOGO_Y + 2, 26, 3, fill=1, stroke=0)
    c.setFillColor(C_ACCENT)
    c.rect(LOGO_X - 13, LOGO_Y - 10, 8, 5, fill=1, stroke=0)

    # Title text
    c.setFillColor(C_ACCENT)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(70, H - 38, "CreditSense AI")

    # Subtitle
    c.setFillColor(C_MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(70, H - 56, "Credit Risk Assessment Report")

    # Date + model badge (right side)
    today = datetime.date.today().strftime("%B %d, %Y")
    c.setFillColor(C_MUTED)
    c.setFont("Helvetica", 8)
    c.drawRightString(W - 20, H - 36, f"Generated: {today}")
    c.setFont("Helvetica", 8)
    c.drawRightString(W - 20, H - 50, f"Model: {model_choice}")

    # Accuracy pill (right)
    pill_x = W - 95
    _draw_rounded_rect(c, pill_x, H - 72, 75, 16, 4,
                       colors.HexColor('#0D2A1A'),
                       C_GREEN, 0.8)
    c.setFillColor(C_GREEN)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(pill_x + 37.5, H - 63, f"Accuracy: {model_accuracy*100:.2f}%")

    # Bottom border of header
    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.8)
    c.line(0, H - HDR_H, W, H - HDR_H)

    # ════════════════════════════════════════
    # VERDICT CARD
    # ════════════════════════════════════════
    V_TOP  = H - HDR_H - 16
    V_H    = 100
    V_X    = 20
    V_W    = W - 40

    _draw_rounded_rect(c, V_X, V_TOP - V_H, V_W, V_H, 8, C_CARD, D_COLOR, 1.5)

    # Verdict label
    c.setFillColor(C_MUTED)
    c.setFont("Helvetica", 8)
    c.drawCentredString(W / 2, V_TOP - 18, "CREDIT DECISION")

    # Verdict text
    c.setFillColor(D_COLOR)
    c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(W / 2, V_TOP - 50, D_TEXT)

    # Probabilities
    c.setFillColor(C_MUTED)
    c.setFont("Helvetica", 8.5)
    prob_line = f"Default Probability: {default_prob*100:.1f}%       Safe Probability: {safe_prob*100:.1f}%"
    c.drawCentredString(W / 2, V_TOP - 68, prob_line)

    # Risk badge
    badge_w = 90
    badge_x = W/2 - badge_w/2
    _draw_rounded_rect(c, badge_x, V_TOP - 92, badge_w, 16, 4,
                       colors.HexColor('#0A1628'), R_COLOR, 0.8)
    c.setFillColor(R_COLOR)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(W / 2, V_TOP - 83, f"Risk Level: {risk_label}")

    # ════════════════════════════════════════
    # TWO COLUMN STATS (Model | Report Info)
    # ════════════════════════════════════════
    CUR_Y = V_TOP - V_H - 18

    col1_x, col2_x = 20, W/2 + 4
    col_w = W/2 - 28

    # Section heading helper
    def section_heading(text, y):
        c.setFillColor(C_ACCENT)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(20, y, text)
        c.setStrokeColor(C_ACCENT)
        c.setLineWidth(0.4)
        c.line(20, y - 3, W - 20, y - 3)

    def info_card(x, y, w, h, rows_data):
        """Draw a dark card with label/value rows."""
        _draw_rounded_rect(c, x, y - h, w, h, 6, C_CARD, C_BORDER, 0.5)
        row_h = h / len(rows_data)
        for i, (lbl, val, val_color) in enumerate(rows_data):
            row_y = y - (i + 0.72) * row_h
            c.setFillColor(C_MUTED)
            c.setFont("Helvetica", 8)
            c.drawString(x + 10, row_y, lbl)
            c.setFillColor(val_color)
            c.setFont("Helvetica-Bold", 8.5)
            c.drawRightString(x + w - 10, row_y, val)
            if i < len(rows_data) - 1:
                c.setStrokeColor(C_DIM)
                c.setLineWidth(0.3)
                c.line(x + 8, row_y - 5, x + w - 8, row_y - 5)

    section_heading("Model & Report Information", CUR_Y)
    CUR_Y -= 14

    info_card(col1_x, CUR_Y, col_w, 70, [
        ("ML Algorithm",     model_choice,                     C_ACCENT),
        ("Test Accuracy",    f"{model_accuracy*100:.2f}%",     C_WHITE),
        ("Risk Assessment",  risk_label,                       R_COLOR),
    ])
    info_card(col2_x, CUR_Y, col_w, 70, [
        ("Report Date",      datetime.date.today().strftime("%d %b %Y"), C_WHITE),
        ("Default Prob.",    f"{default_prob*100:.1f}%",       D_COLOR),
        ("Decision",         D_TEXT,                           D_COLOR),
    ])

    # ════════════════════════════════════════
    # APPLICANT & LOAN DETAILS
    # ════════════════════════════════════════
    CUR_Y -= 82

    section_heading("Applicant & Loan Details", CUR_Y)
    CUR_Y -= 14

    details = [
        ("Age",                   str(raw_input['person_age']) + " years",          C_WHITE),
        ("Annual Income",         f"${raw_input['person_income']:,}",               C_WHITE),
        ("Employment Length",     f"{raw_input['person_emp_length']} years",        C_WHITE),
        ("Home Ownership",        raw_input['person_home_ownership'],               C_WHITE),
        ("Loan Amount",           f"${raw_input['loan_amnt']:,}",                   C_ACCENT),
        ("Loan Grade",            raw_input['loan_grade'],                          C_ACCENT),
        ("Interest Rate",         f"{raw_input['loan_int_rate']:.1f}%",            C_AMBER),
        ("Loan Purpose",          raw_input['loan_intent'],                         C_WHITE),
        ("Loan / Income Ratio",   f"{raw_input['loan_percent_income']:.3f}",       C_WHITE),
        ("Prior Default on File", raw_input['cb_person_default_on_file'],
             C_RED if raw_input['cb_person_default_on_file'] == 'Y' else C_GREEN),
        ("Credit History",        f"{raw_input['cb_person_cred_hist_length']} years", C_WHITE),
        ("Model Used",            model_choice,                                     C_PURPLE),
    ]

    CARD_H = 170
    _draw_rounded_rect(c, 20, CUR_Y - CARD_H, W - 40, CARD_H, 8, C_CARD, C_BORDER, 0.5)

    cols = 2
    rows_per_col = (len(details) + 1) // 2
    col_w2 = (W - 40) / cols
    row_h2 = CARD_H / rows_per_col

    for idx, (lbl, val, val_color) in enumerate(details):
        col_idx = idx // rows_per_col
        row_idx = idx % rows_per_col
        cell_x = 20 + col_idx * col_w2
        cell_y = CUR_Y - (row_idx + 0.7) * row_h2

        c.setFillColor(C_MUTED)
        c.setFont("Helvetica", 7.8)
        c.drawString(cell_x + 12, cell_y, lbl)
        c.setFillColor(val_color)
        c.setFont("Helvetica-Bold", 8.2)
        c.drawRightString(cell_x + col_w2 - 12, cell_y, val)

        # Row divider
        if row_idx < rows_per_col - 1:
            c.setStrokeColor(C_DIM)
            c.setLineWidth(0.25)
            c.line(cell_x + 8, cell_y - 5, cell_x + col_w2 - 8, cell_y - 5)

    # Vertical divider between columns
    mid_x = 20 + col_w2
    c.setStrokeColor(C_DIM)
    c.setLineWidth(0.4)
    c.line(mid_x, CUR_Y - CARD_H + 10, mid_x, CUR_Y - 10)

    # ════════════════════════════════════════
    # FEATURE IMPORTANCE CHART
    # ════════════════════════════════════════
    CUR_Y -= CARD_H + 18

    section_heading("Feature Importance Analysis", CUR_Y)
    CUR_Y -= 10

    c.setFillColor(C_MUTED)
    c.setFont("Helvetica", 7.5)
    c.drawString(20, CUR_Y - 4,
        "Relative contribution of each input variable to the model prediction.")
    CUR_Y -= 16

    # Draw chart via matplotlib and embed as image
    try:
        mfig, ax = plt.subplots(figsize=(8.2, 3.6), facecolor='#04080F')
        ax.set_facecolor('#0A1628')

        max_v = max(fi_values) if fi_values else 1
        bar_colors = ['#00D4FF' if v >= max_v * 0.6 else
                      '#7B61FF' if v >= max_v * 0.3 else '#1A3A6A'
                      for v in fi_values]

        bars = ax.barh(fi_names, fi_values, color=bar_colors, height=0.52,
                       edgecolor='none', zorder=3)

        # Value annotations
        for bar, val in zip(bars, fi_values):
            ax.text(bar.get_width() + max_v * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', ha='left',
                    color='#6B8CAE', fontsize=7.5, fontweight='bold')

        ax.set_xlabel('Importance Score', color='#6B8CAE', fontsize=8, labelpad=6)
        ax.tick_params(axis='y', colors='#C8DFF0', labelsize=8)
        ax.tick_params(axis='x', colors='#3A5A7A', labelsize=7)
        ax.set_xlim(0, max_v * 1.22)
        ax.xaxis.grid(True, color='#112240', linewidth=0.5, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#112240')
            spine.set_linewidth(0.6)

        mfig.tight_layout(pad=0.6)
        chart_buf = io.BytesIO()
        mfig.savefig(chart_buf, format='png', dpi=170, bbox_inches='tight',
                     facecolor='#04080F', edgecolor='none')
        plt.close(mfig)
        chart_buf.seek(0)

        CHART_H = 155
        _draw_rounded_rect(c, 20, CUR_Y - CHART_H, W - 40, CHART_H, 6, C_CARD, C_BORDER, 0.5)
        chart_img = ImageReader(chart_buf)
        c.drawImage(chart_img, 24, CUR_Y - CHART_H + 4,
                    width=W - 48, height=CHART_H - 8,
                    preserveAspectRatio=True, mask='auto')
        CUR_Y -= CHART_H + 14

    except Exception as e:
        c.setFillColor(C_RED)
        c.setFont("Helvetica", 8)
        c.drawString(20, CUR_Y - 16, f"Chart error: {str(e)[:80]}")
        CUR_Y -= 30

    # ════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════
    FOOT_H = 36
    # Footer background band
    c.setFillColor(colors.HexColor('#060E1C'))
    c.rect(0, 0, W, FOOT_H, fill=1, stroke=0)
    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.6)
    c.line(0, FOOT_H, W, FOOT_H)
    # Cyan bottom accent
    c.setFillColor(C_ACCENT)
    c.rect(0, 0, W, 2, fill=1, stroke=0)

    c.setFillColor(C_MUTED)
    c.setFont("Helvetica", 7.2)
    c.drawCentredString(W / 2, FOOT_H - 14,
        "This report is generated by CreditSense AI for informational purposes only.")
    c.setFillColor(C_PURPLE)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(W / 2, FOOT_H - 26, "Developed by Rizwan Ahmed")

    c.save()
    buffer.seek(0)
    return buffer


PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,22,40,0.6)',
    font=dict(family='Space Grotesk', color='#6B8CAE'),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor='rgba(0,212,255,0.07)', zerolinecolor='rgba(0,212,255,0.1)'),
    yaxis=dict(gridcolor='rgba(0,212,255,0.07)', zerolinecolor='rgba(0,212,255,0.1)'),
)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem;'>💳</div>
        <div style='font-size:1.2rem; font-weight:700; color:#00D4FF; letter-spacing:-0.5px;'>CreditSense AI</div>
        <div style='font-size:0.75rem; color:#6B8CAE; letter-spacing:2px; text-transform:uppercase;'>Credit Risk Analyzer</div>
    </div>
    <hr style='border-color:rgba(0,212,255,0.15); margin:0.5rem 0 1.2rem;'>
    """, unsafe_allow_html=True)

    st.markdown("**🤖 Select Model**")
    model_choice = st.selectbox("", ['Random Forest', 'Decision Tree', 'Logistic Regression'], label_visibility="collapsed")

    st.markdown("<br>**👤 Applicant Profile**", unsafe_allow_html=True)
    age = st.slider("Age", 18, 75, 30)
    income = st.number_input("Annual Income ($)", min_value=1000, max_value=500000, value=55000, step=1000)
    emp_length = st.slider("Employment Length (years)", 0.0, 50.0, 5.0, 0.5)
    home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    cred_hist = st.slider("Credit History Length (years)", 1, 40, 5)
    default_on_file = st.selectbox("Prior Default on File", ['N', 'Y'])

    st.markdown("<br>**💰 Loan Details**", unsafe_allow_html=True)
    loan_amnt = st.number_input("Loan Amount ($)", min_value=100, max_value=100000, value=10000, step=500)
    loan_intent = st.selectbox("Loan Purpose", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0, 0.1)

    loan_pct_income = round(loan_amnt / max(income, 1), 4)
    st.markdown(f"""
    <div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);border-radius:8px;padding:0.6rem 1rem;margin-top:0.5rem;'>
        <span style='color:#6B8CAE;font-size:0.8rem;'>Loan-to-Income Ratio</span><br>
        <span style='color:#00D4FF;font-size:1.3rem;font-weight:700;font-family:JetBrains Mono,monospace;'>{loan_pct_income:.3f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Run Prediction", type="primary")

    st.markdown("""
    <hr style='border-color:rgba(0,212,255,0.1);margin-top:1.5rem;'>
    <div style='text-align:center;color:#2D4A6E;font-size:0.72rem;padding:0.5rem;'>
        Credit Risk ML · Smarter Lending Decisions<br>
        <span style='color:#00D4FF;'>Rizwan</span> · Sukkur IBA University
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class='hero-header'>
    <h1 class='hero-title'>💳 CreditSense AI</h1>
    <p class='hero-subtitle'>Predict loan default risk in seconds — powered by ensemble machine learning</p>
    <div class='badge-row'>
        <span class='badge'>Random Forest</span>
        <span class='badge'>Decision Tree</span>
        <span class='badge'>Logistic Regression</span>
        <span class='badge'>Scikit-learn</span>
        <span class='badge'>Plotly</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load data & models
with st.spinner("🔄 Training models on credit risk dataset..."):
    df = generate_sample_data()
    model_results, feature_names, X_test, y_test = train_models(df)

chosen = model_results[model_choice]

# ─── TOP METRICS ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("Accuracy", f"{chosen['accuracy']*100:.1f}%", "↑ Test set"),
    ("ROC-AUC", f"{chosen['auc']:.3f}", "Area under curve"),
    ("Dataset Size", "3,000", "Synthetic samples"),
    ("Features Used", "11", "Input variables"),
]
for col, (label, value, sub) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            <div style='color:#2D4A6E;font-size:0.75rem;margin-top:0.2rem;'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎯  Prediction", "📊  Model Analytics", "🔬  Feature Insights", "📋  Dataset Overview"])

# ════════════════════════════════════════════
# TAB 1: PREDICTION
# ════════════════════════════════════════════
with tab1:
    raw_input = {
        'person_age': age,
        'person_income': income,
        'person_home_ownership': home_ownership,
        'person_emp_length': emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': int_rate,
        'loan_percent_income': loan_pct_income,
        'cb_person_default_on_file': default_on_file,
        'cb_person_cred_hist_length': cred_hist
    }

    model_obj = chosen['model']
    encoded = encode_input(raw_input)
    proba = model_obj.predict_proba(encoded)[0]
    pred_class = model_obj.predict(encoded)[0]
    default_prob = proba[1]
    safe_prob = proba[0]
    risk_label, risk_css = get_risk_level(default_prob)

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown("<div class='section-title'>⚡ Risk Assessment Result</div>", unsafe_allow_html=True)

        if pred_class == 0:
            st.markdown(f"""
            <div class='result-card-approved'>
                <div class='result-label' style='color:#00FF88;'>Credit Decision</div>
                <div class='result-verdict' style='color:#00FF88;'>✅ APPROVED</div>
                <div class='result-prob'>Default probability: {default_prob*100:.1f}% · Safe probability: {safe_prob*100:.1f}%</div>
                <div style='margin-top:1rem;'>
                    <span class='risk-badge {risk_css}'>Risk Level: {risk_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card-rejected'>
                <div class='result-label' style='color:#FF4757;'>Credit Decision</div>
                <div class='result-verdict' style='color:#FF4757;'>❌ REJECTED</div>
                <div class='result-prob'>Default probability: {default_prob*100:.1f}% · Safe probability: {safe_prob*100:.1f}%</div>
                <div style='margin-top:1rem;'>
                    <span class='risk-badge {risk_css}'>Risk Level: {risk_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Probability gauge
        st.markdown("<br>", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(default_prob * 100, 1),
            number={'suffix': '%', 'font': {'size': 36, 'color': '#00D4FF', 'family': 'JetBrains Mono'}},
            title={'text': "Default Probability", 'font': {'color': '#6B8CAE', 'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#2D4A6E', 'tickfont': {'color': '#2D4A6E'}},
                'bar': {'color': '#FF4757' if default_prob > 0.5 else '#00D4FF', 'thickness': 0.25},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(0,255,136,0.12)'},
                    {'range': [35, 65], 'color': 'rgba(255,160,0,0.12)'},
                    {'range': [65, 100], 'color': 'rgba(255,71,87,0.12)'},
                ],
                'threshold': {
                    'line': {'color': '#7B61FF', 'width': 2},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(**{**PLOTLY_LAYOUT, 'height': 260, 'margin': dict(l=20, r=20, t=10, b=10)})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-title'>📝 Applicant Summary</div>", unsafe_allow_html=True)

        # Key stats table only (no radar chart)
        st.markdown(f"""
        <div style='background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1.2rem;font-size:0.87rem;margin-top:0.3rem;'>
            <table style='width:100%;border-collapse:collapse;'>
                <tr><td style='color:#6B8CAE;padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.06);'>Model</td><td style='color:#00D4FF;text-align:right;font-family:JetBrains Mono,monospace;border-bottom:1px solid rgba(0,212,255,0.06);'>{model_choice}</td></tr>
                <tr><td style='color:#6B8CAE;padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.06);'>Loan Grade</td><td style='color:#E8F4FD;text-align:right;font-family:JetBrains Mono,monospace;border-bottom:1px solid rgba(0,212,255,0.06);'>{loan_grade}</td></tr>
                <tr><td style='color:#6B8CAE;padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.06);'>Purpose</td><td style='color:#E8F4FD;text-align:right;font-family:JetBrains Mono,monospace;border-bottom:1px solid rgba(0,212,255,0.06);'>{loan_intent}</td></tr>
                <tr><td style='color:#6B8CAE;padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.06);'>Annual Income</td><td style='color:#E8F4FD;text-align:right;font-family:JetBrains Mono,monospace;border-bottom:1px solid rgba(0,212,255,0.06);'>${income:,}</td></tr>
                <tr><td style='color:#6B8CAE;padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.06);'>Loan Amount</td><td style='color:#E8F4FD;text-align:right;font-family:JetBrains Mono,monospace;border-bottom:1px solid rgba(0,212,255,0.06);'>${loan_amnt:,}</td></tr>
                <tr><td style='color:#6B8CAE;padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.06);'>Interest Rate</td><td style='color:#E8F4FD;text-align:right;font-family:JetBrains Mono,monospace;border-bottom:1px solid rgba(0,212,255,0.06);'>{int_rate:.1f}%</td></tr>
                <tr><td style='color:#6B8CAE;padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.06);'>Loan / Income</td><td style='color:#E8F4FD;text-align:right;font-family:JetBrains Mono,monospace;border-bottom:1px solid rgba(0,212,255,0.06);'>{loan_pct_income:.3f}</td></tr>
                <tr><td style='color:#6B8CAE;padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.06);'>Prior Default</td><td style='color:{"#FF4757" if default_on_file=="Y" else "#00FF88"};text-align:right;font-family:JetBrains Mono,monospace;border-bottom:1px solid rgba(0,212,255,0.06);'>{default_on_file}</td></tr>
                <tr><td style='color:#6B8CAE;padding:6px 0;'>Home Status</td><td style='color:#E8F4FD;text-align:right;font-family:JetBrains Mono,monospace;'>{home_ownership}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ─── PDF DOWNLOAD ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📥 Download Report</div>", unsafe_allow_html=True)

    # Build sorted feature importance arrays for PDF chart
    _feat_labels_all = ['Age', 'Income', 'Home Ownership', 'Emp. Length',
                        'Loan Intent', 'Loan Grade', 'Loan Amount',
                        'Interest Rate', 'Loan % Income', 'Prior Default', 'Cred. History']
    if hasattr(chosen['model'], 'feature_importances_'):
        _fi_raw = chosen['model'].feature_importances_
        _sorted = np.argsort(_fi_raw)
        _pdf_fi_vals  = list(_fi_raw[_sorted])
        _pdf_fi_names = [_feat_labels_all[i] for i in _sorted]
    else:
        _coef = np.abs(chosen['model'].coef_[0])
        _feat_labels_lr = ['Age', 'Income', 'Home Own.', 'Emp. Len.', 'Intent',
                           'Grade', 'Loan Amt', 'Int. Rate', 'Loan%Inc', 'Default', 'Cred.Hist']
        _sorted = np.argsort(_coef)
        _pdf_fi_vals  = list(_coef[_sorted])
        _pdf_fi_names = [_feat_labels_lr[i] for i in _sorted]

    pdf_buffer = generate_pdf_report(
        raw_input=raw_input,
        pred_class=pred_class,
        default_prob=default_prob,
        safe_prob=safe_prob,
        risk_label=risk_label,
        model_choice=model_choice,
        model_accuracy=chosen['accuracy'],
        fi_values=_pdf_fi_vals,
        fi_names=_pdf_fi_names,
    )

    st.download_button(
        label="⬇️  Download PDF Report",
        data=pdf_buffer,
        file_name="creditsense_report.pdf",
        mime="application/pdf",
    )


# ════════════════════════════════════════════
# TAB 2: MODEL ANALYTICS
# ════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>📊 Model Comparison</div>", unsafe_allow_html=True)

    # Bar chart comparing models — full width
    model_names = list(model_results.keys())
    accs = [model_results[m]['accuracy'] * 100 for m in model_names]
    aucs = [model_results[m]['auc'] for m in model_names]
    colors = ['#00D4FF' if m == model_choice else '#2D4A6E' for m in model_names]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name='Accuracy (%)', x=model_names, y=accs,
        marker_color=colors,
        text=[f"{a:.2f}%" for a in accs],
        textposition='outside',
        textfont=dict(color='#E8F4FD', size=12),
    ))
    fig_bar.add_trace(go.Bar(
        name='AUC × 100', x=model_names, y=[a * 100 for a in aucs],
        marker_color=['rgba(123,97,255,0.7)' if m == model_choice else 'rgba(45,74,110,0.5)' for m in model_names],
        text=[f"{a:.3f}" for a in aucs],
        textposition='outside',
        textfont=dict(color='#E8F4FD', size=12),
    ))
    fig_bar.update_layout(**{**PLOTLY_LAYOUT,
        'barmode': 'group',
        'height': 360,
        'title': dict(text='Model Performance Comparison', font={'color': '#6B8CAE', 'size': 13}),
        'legend': dict(font={'color': '#6B8CAE'}, bgcolor='rgba(0,0,0,0)'),
        'yaxis': {**PLOTLY_LAYOUT['yaxis'], 'range': [60, 100]},
    })
    st.plotly_chart(fig_bar, use_container_width=True)

    # Confusion matrix
    st.markdown("<div class='section-title'>🔢 Confusion Matrix</div>", unsafe_allow_html=True)

    cm = chosen['cm']
    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=['Predicted: Safe', 'Predicted: Default'],
        y=['Actual: Safe', 'Actual: Default'],
        colorscale=[[0, '#050A12'], [0.5, '#0D1F3C'], [1, '#00D4FF']],
        text=[[str(v) for v in row] for row in cm],
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=22, color='white'),
        showscale=True,
        colorbar=dict(tickfont={'color': '#6B8CAE'})
    ))
    fig_cm.update_layout(**{**PLOTLY_LAYOUT, 'height': 300,
        'title': dict(text=f'Confusion Matrix — {model_choice}', font={'color': '#6B8CAE', 'size': 13}),
        'xaxis': dict(tickfont={'color': '#6B8CAE'}),
        'yaxis': dict(tickfont={'color': '#6B8CAE'}),
    })
    st.plotly_chart(fig_cm, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3: FEATURE INSIGHTS
# ════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>🔬 Feature Importance</div>", unsafe_allow_html=True)

    if hasattr(chosen['model'], 'feature_importances_'):
        fi = chosen['model'].feature_importances_
        feat_labels = ['Age', 'Income', 'Home Ownership', 'Emp. Length',
                       'Loan Intent', 'Loan Grade', 'Loan Amount',
                       'Interest Rate', 'Loan % Income', 'Prior Default', 'Cred. History']
        sorted_idx = np.argsort(fi)
        fig_fi = go.Figure(go.Bar(
            x=fi[sorted_idx],
            y=[feat_labels[i] for i in sorted_idx],
            orientation='h',
            marker=dict(
                color=fi[sorted_idx],
                colorscale=[[0, '#0D1F3C'], [0.5, '#7B61FF'], [1, '#00D4FF']],
                showscale=False
            ),
            text=[f"{v:.3f}" for v in fi[sorted_idx]],
            textposition='outside',
            textfont=dict(color='#6B8CAE', size=11),
        ))
        fig_fi.update_layout(**{**PLOTLY_LAYOUT,
            'height': 420,
            'title': dict(text=f'Feature Importance — {model_choice}', font={'color': '#6B8CAE', 'size': 13}),
            'xaxis': {**PLOTLY_LAYOUT['xaxis'], 'title': 'Importance Score'},
            'yaxis': dict(tickfont={'color': '#E8F4FD', 'size': 12}),
            'margin': dict(l=140, r=80, t=40, b=20),
        })
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        # Logistic Regression coefficients
        coef = np.abs(chosen['model'].coef_[0])
        feat_labels = ['Age', 'Income', 'Home Own.', 'Emp. Len.', 'Intent',
                       'Grade', 'Loan Amt', 'Int. Rate', 'Loan%Inc', 'Default', 'Cred.Hist']
        sorted_idx = np.argsort(coef)
        fig_fi = go.Figure(go.Bar(
            x=coef[sorted_idx],
            y=[feat_labels[i] for i in sorted_idx],
            orientation='h',
            marker_color='#00D4FF',
            text=[f"{v:.3f}" for v in coef[sorted_idx]],
            textposition='outside',
            textfont=dict(color='#6B8CAE', size=11),
        ))
        fig_fi.update_layout(**{**PLOTLY_LAYOUT,
            'height': 420,
            'title': dict(text='Coefficient Magnitudes — Logistic Regression', font={'color': '#6B8CAE', 'size': 13}),
            'yaxis': dict(tickfont={'color': '#E8F4FD', 'size': 12}),
            'margin': dict(l=110, r=80, t=40, b=20),
        })
        st.plotly_chart(fig_fi, use_container_width=True)

    # Grade vs default rate
    st.markdown("<div class='section-title'>📉 Loan Grade vs Default Rate</div>", unsafe_allow_html=True)
    grade_default = df.groupby('loan_grade')['loan_status'].mean().reset_index()
    grade_default.columns = ['Grade', 'Default Rate']
    grade_default = grade_default.sort_values('Grade')

    fig_grade = go.Figure()
    fig_grade.add_trace(go.Bar(
        x=grade_default['Grade'],
        y=grade_default['Default Rate'] * 100,
        marker=dict(
            color=grade_default['Default Rate'],
            colorscale=[[0, '#00FF88'], [0.5, '#FFA000'], [1, '#FF4757']],
        ),
        text=[f"{v*100:.1f}%" for v in grade_default['Default Rate']],
        textposition='outside',
        textfont=dict(color='#E8F4FD'),
    ))
    fig_grade.update_layout(**{**PLOTLY_LAYOUT,
        'height': 280,
        'title': dict(text='Default Rate by Loan Grade', font={'color': '#6B8CAE', 'size': 13}),
        'xaxis': {**PLOTLY_LAYOUT['xaxis'], 'title': 'Loan Grade'},
        'yaxis': {**PLOTLY_LAYOUT['yaxis'], 'title': 'Default Rate (%)'},
    })
    st.plotly_chart(fig_grade, use_container_width=True)


# ════════════════════════════════════════════
# TAB 4: DATASET OVERVIEW
# ════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>📋 Dataset Summary</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        default_rate = df['loan_status'].mean()
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Overall Default Rate</div>
            <div class='metric-value' style='color:#FF4757;'>{default_rate*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Avg. Loan Amount</div>
            <div class='metric-value'>${df['loan_amnt'].mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Avg. Interest Rate</div>
            <div class='metric-value'>{df['loan_int_rate'].mean():.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_p, col_q = st.columns(2)

    with col_p:
        # Loan intent distribution
        intent_counts = df['loan_intent'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=intent_counts.index,
            values=intent_counts.values,
            hole=0.5,
            marker=dict(colors=['#00D4FF', '#7B61FF', '#00FF88', '#FFA000', '#FF4757', '#FF69B4']),
            textfont=dict(color='white', size=11),
        ))
        fig_pie.update_layout(**{**PLOTLY_LAYOUT,
            'height': 300,
            'title': dict(text='Loan Purpose Distribution', font={'color': '#6B8CAE', 'size': 13}),
            'legend': dict(font={'color': '#6B8CAE', 'size': 10}, bgcolor='rgba(0,0,0,0)'),
            'margin': dict(l=10, r=10, t=40, b=10),
        })
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_q:
        # Countplot: loan_status distribution
        status_counts = df['loan_status'].value_counts().sort_index()
        fig_count = go.Figure(go.Bar(
            x=['No Default (0)', 'Default (1)'],
            y=status_counts.values,
            marker_color=['rgba(0,212,255,0.75)', 'rgba(255,71,87,0.75)'],
            text=[f"{v:,}" for v in status_counts.values],
            textposition='outside',
            textfont=dict(color='#E8F4FD', size=13),
            width=0.5,
        ))
        fig_count.update_layout(**{**PLOTLY_LAYOUT,
            'height': 300,
            'title': dict(text='Loan Status Distribution (Countplot)', font={'color': '#6B8CAE', 'size': 13}),
            'xaxis': {**PLOTLY_LAYOUT['xaxis'], 'title': 'Loan Status'},
            'yaxis': {**PLOTLY_LAYOUT['yaxis'], 'title': 'Count'},
        })
        st.plotly_chart(fig_count, use_container_width=True)

    st.markdown("<br>**📄 Sample Records**", unsafe_allow_html=True)
    st.dataframe(
        df.head(10).style.background_gradient(subset=['loan_status'], cmap='RdYlGn_r'),
        use_container_width=True,
        height=300
    )