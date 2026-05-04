import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from deep_translator import GoogleTranslator
import time

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="FakeGuard",
    page_icon="🛡️",
    layout="centered",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #f1f5f9;
}

.stApp {
    background: #0a0a0f;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 3rem !important; max-width: 680px !important; }

/* ─── Typography ─── */
h1, h2, h3 { color: #f1f5f9 !important; }

/* ─── Hero ─── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.35);
    color: #a78bfa;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 0.3rem 0.9rem;
    border-radius: 99px;
    margin-bottom: 1.2rem;
    text-transform: uppercase;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.15;
    color: #f8fafc;
    margin-bottom: 0.75rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-desc {
    color: #64748b;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 0;
}

/* ─── Stats Row ─── */
.stats-row {
    display: flex;
    gap: 1px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    overflow: hidden;
    margin: 2rem 0;
}
.stat-item {
    flex: 1;
    text-align: center;
    padding: 1rem 0.5rem;
    background: #0a0a0f;
}
.stat-item:not(:last-child) {
    border-right: 1px solid rgba(255,255,255,0.06);
}
.stat-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #a78bfa;
    display: block;
}
.stat-lbl {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ─── Input Card ─── */
.input-card {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
}
.input-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ─── Override Streamlit inputs ─── */
.stTextArea textarea {
    background: #0d0d14 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    resize: none !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: rgba(139,92,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
}
.stTextArea label { display: none !important; }
.stSelectbox label { 
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
div[data-baseweb="select"] > div {
    background: #0d0d14 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* ─── Analyze Button ─── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.2s, transform 0.15s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ─── Verdict Cards ─── */
.verdict {
    border-radius: 16px;
    padding: 2rem 1.75rem;
    text-align: center;
    margin: 1.5rem 0;
}
.verdict-fake {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
}
.verdict-real {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
}
.verdict-icon { font-size: 2.5rem; margin-bottom: 0.75rem; display: block; }
.verdict-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.verdict-fake .verdict-title { color: #f87171; }
.verdict-real .verdict-title { color: #34d399; }
.verdict-body {
    font-size: 0.88rem;
    color: #64748b;
    line-height: 1.6;
}

/* ─── Pipeline Steps ─── */
.pipeline-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #475569;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem;
}
.step {
    display: flex;
    gap: 0.85rem;
    align-items: flex-start;
    padding: 0.65rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.step:last-child { border-bottom: none; }
.step-badge {
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    border-radius: 6px;
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 2px;
}
.step-value {
    font-size: 0.8rem;
    color: #475569;
    word-break: break-word;
    line-height: 1.5;
}

/* ─── Expander ─── */
.streamlit-expanderHeader {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #64748b !important;
    font-size: 0.82rem !important;
}
.streamlit-expanderContent {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    background: #0d0d14 !important;
    padding: 0 !important;
}

/* ─── Divider ─── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }

/* ─── Footer ─── */
.footer {
    text-align: center;
    color: #1e293b;
    font-size: 0.75rem;
    padding: 2rem 0 1rem;
}

/* Misc overrides */
label, p, span, div { color: inherit; }
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── NLTK Setup ────────────────────────────────────────────────
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
_stemmer   = PorterStemmer()
_stopwords = set(stopwords.words('english'))

# ── Helpers ───────────────────────────────────────────────────
def _clean(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def run_pipeline(text, translate=True):
    steps = [("0", "Teks Asli", text)]
    if translate:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        steps.append(("T", "Translasi EN", translated))
    else:
        translated = text
    cleaned = _clean(translated)
    steps.append(("1", "Cleaning", cleaned))
    folded = cleaned.lower()
    steps.append(("2", "Case Folding", folded))
    tokens = word_tokenize(folded)
    steps.append(("3", "Tokenization", str(tokens[:12]) + ("..." if len(tokens) > 12 else "")))
    no_stop = [w for w in tokens if w not in _stopwords]
    steps.append(("4", "Stopword Removal", str(no_stop[:12]) + ("..." if len(no_stop) > 12 else "")))
    stemmed = [_stemmer.stem(w) for w in no_stop]
    steps.append(("5", "Stemming", str(stemmed[:12]) + ("..." if len(stemmed) > 12 else "")))
    final = " ".join(stemmed)
    steps.append(("F", "Final Text", final))
    return final, steps

@st.cache_resource(show_spinner=False)
def load_models():
    return joblib.load('tfidf_model.pkl'), joblib.load('svm_model.pkl')

# ── Load ──────────────────────────────────────────────────────
try:
    tfidf_model, svm_model = load_models()
    model_ok = True
except Exception as e:
    model_ok = False; model_err = str(e)

# ══════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI Powered Detection</div>
    <div class="hero-title">Deteksi <span>Ulasan Palsu</span><br>dalam Hitungan Detik</div>
    <div class="hero-desc">Menggunakan algoritma TF-IDF &amp; Support Vector Machine<br>yang dilatih pada 40.000+ ulasan produk</div>
</div>
""", unsafe_allow_html=True)

# Stats
st.markdown("""
<div class="stats-row">
    <div class="stat-item"><span class="stat-val">87%</span><span class="stat-lbl">Akurasi</span></div>
    <div class="stat-item"><span class="stat-val">0.87</span><span class="stat-lbl">Precision</span></div>
    <div class="stat-item"><span class="stat-val">0.87</span><span class="stat-lbl">F1-Score</span></div>
    <div class="stat-item"><span class="stat-val">40K+</span><span class="stat-lbl">Data Latih</span></div>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.error(f"Model tidak ditemukan. Pastikan `svm_model.pkl` & `tfidf_model.pkl` ada.\n\n`{model_err}`")
    st.stop()

# Input Card
lang_opt = st.selectbox(
    "BAHASA ULASAN",
    ["Indonesia 🇮🇩 (Auto-Translate)", "English 🇬🇧"],
    index=0,
    key="lang"
)
needs_translate = "Indonesia" in lang_opt

review_text = st.text_area(
    "review",
    height=150,
    placeholder="Tempel teks ulasan produk di sini…",
    key="review_input",
    label_visibility="collapsed"
)

btn = st.button("Analisis Sekarang →", key="analyze")

# Analysis
if btn:
    if not review_text.strip():
        st.warning("Masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("Menganalisis…"):
            try:
                final_text, steps = run_pipeline(review_text.strip(), translate=needs_translate)
                vec   = tfidf_model.transform([final_text])
                label = svm_model.predict(vec)[0]
                is_fake = (label == "CG")
                time.sleep(0.3)

                # Verdict
                if is_fake:
                    st.markdown("""
                    <div class="verdict verdict-fake">
                        <span class="verdict-icon">🚨</span>
                        <div class="verdict-title">Ulasan Palsu Terdeteksi</div>
                        <div class="verdict-body">Model mendeteksi pola bahasa yang tidak natural.<br>
                        Ulasan ini kemungkinan dibuat secara otomatis (Computer Generated).</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="verdict verdict-real">
                        <span class="verdict-icon">✅</span>
                        <div class="verdict-title">Ulasan Asli</div>
                        <div class="verdict-body">Model mendeteksi pola bahasa yang natural dan autentik.<br>
                        Ulasan ini kemungkinan besar ditulis oleh manusia sungguhan.</div>
                    </div>""", unsafe_allow_html=True)

                # Pipeline Steps
                with st.expander("Lihat proses NLP step-by-step"):
                    html_steps = '<div style="padding:1rem 1.25rem;">'
                    html_steps += '<div class="pipeline-title">Alur Pemrosesan Teks</div>'
                    for badge, name, val in steps:
                        html_steps += f"""
                        <div class="step">
                            <div class="step-badge">{badge}</div>
                            <div>
                                <div class="step-name">{name}</div>
                                <div class="step-value">{str(val)[:200]}</div>
                            </div>
                        </div>"""
                    html_steps += '</div>'
                    st.markdown(html_steps, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

# Footer
st.markdown("""
<div class="footer">
    FakeGuard &nbsp;·&nbsp; TF-IDF + SVM + NLTK &nbsp;·&nbsp; Progress Report APM 2025
</div>
""", unsafe_allow_html=True)
