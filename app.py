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
    color: #1e293b;
}

.stApp {
    background: #f8fafc;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 3rem !important; max-width: 720px !important; }

/* ─── Typography ─── */
h1, h2, h3 { color: #0f172a !important; font-weight: 700 !important; }
p { color: #475569 !important; }

/* ─── Hero ─── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    color: #4f46e5;
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
    font-weight: 800;
    line-height: 1.15;
    color: #0f172a;
    margin-bottom: 0.75rem;
    letter-spacing: -0.03em;
}
.hero-title span {
    background: linear-gradient(135deg, #6366f1, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-desc {
    color: #64748b;
    font-size: 1.05rem;
    line-height: 1.6;
    margin-bottom: 0;
}

/* ─── Stats Row ─── */
.stats-row {
    display: flex;
    gap: 1px;
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
    border-radius: 14px;
    overflow: hidden;
    margin: 2rem 0;
}
.stat-item {
    flex: 1;
    text-align: center;
    padding: 1rem 0.5rem;
    background: #ffffff;
}
.stat-item:not(:last-child) {
    border-right: 1px solid rgba(0,0,0,0.04);
}
.stat-val {
    font-size: 1.6rem;
    font-weight: 800;
    color: #4f46e5;
    display: block;
}
.stat-lbl {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 2px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ─── Input Area ─── */
.stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    color: #0f172a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    resize: none !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    transition: all 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stTextArea label { display: none !important; }
.stSelectbox label { 
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
}

/* ─── Analyze Button ─── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 6px -1px rgba(99,102,241,0.2) !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    opacity: 0.95 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 8px -1px rgba(99,102,241,0.3) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ─── Verdict Cards ─── */
.verdict {
    border-radius: 16px;
    padding: 2rem 1.75rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
}
.verdict-fake {
    background: #fef2f2;
    border: 1px solid #fecaca;
}
.verdict-real {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
}
.verdict-icon { font-size: 3rem; margin-bottom: 0.75rem; display: block; }
.verdict-title {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.verdict-fake .verdict-title { color: #dc2626; }
.verdict-real .verdict-title { color: #059669; }
.verdict-body {
    font-size: 0.95rem;
    color: #475569;
    line-height: 1.6;
}

/* ─── Pipeline Steps ─── */
.pipeline-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem;
}
.step {
    display: flex;
    gap: 0.85rem;
    align-items: flex-start;
    padding: 0.75rem 0;
    border-bottom: 1px solid rgba(0,0,0,0.05);
}
.step:last-child { border-bottom: none; }
.step-badge {
    background: #e0e7ff;
    color: #4f46e5;
    border-radius: 6px;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-name {
    font-size: 0.85rem;
    font-weight: 700;
    color: #334155;
    margin-bottom: 3px;
}
.step-value {
    font-size: 0.85rem;
    color: #475569;
    word-break: break-word;
    line-height: 1.5;
}

/* ─── Expander ─── */
.streamlit-expanderHeader {
    background: transparent !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.streamlit-expanderContent {
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    background: #ffffff !important;
    padding: 0 !important;
}

/* ─── Info/Alert boxes ─── */
.stAlert {
    background-color: #f1f5f9 !important;
    border-radius: 12px !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0 !important;
}

/* ─── Divider ─── */
hr { border-color: #e2e8f0 !important; margin: 2rem 0 !important; }

/* ─── Footer ─── */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.8rem;
    padding: 2.5rem 0 1.5rem;
}
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
    <div class="hero-title">Deteksi <span>Ulasan Palsu</span><br>dalam Hitungan Detik</div>
    <div class="hero-desc">Menggunakan algoritma TF-IDF &amp; Support Vector Machine<br>yang dilatih pada 40.000+ ulasan produk</div>
</div>
""", unsafe_allow_html=True)

# Stats
st.markdown("""
<div class="stats-row">
    <div class="stat-item"><span class="stat-val">92%</span><span class="stat-lbl">Akurasi</span></div>
    <div class="stat-item"><span class="stat-val">0.92</span><span class="stat-lbl">Precision</span></div>
    <div class="stat-item"><span class="stat-val">0.92</span><span class="stat-lbl">F1-Score</span></div>
    <div class="stat-item"><span class="stat-val">40K+</span><span class="stat-lbl">Data Latih</span></div>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.error(f"Model tidak ditemukan. Pastikan `svm_model.pkl` & `tfidf_model.pkl` ada.\n\n`{model_err}`")
    st.stop()

# Information Section
st.markdown("### 📖 Cara Penggunaan & Tentang Model")
st.info("""
1. Pilih bahasa ulasan di bawah ini.
2. Tempel ulasan produk yang dicurigai (misalnya dari e-commerce) ke dalam kotak teks.
3. Klik tombol **Analisis Sekarang →**
""")

with st.expander("🤖 Arsitektur Super-Optimized Hybrid Model"):
    st.markdown("""
    **Bagaimana cara kerjanya?**
    - **Algoritma Utama:** Ridge Classifier (Model Linear Cepat berakurasi tinggi).
    - **Ekstraksi Fitur:** TF-IDF (Term Frequency-Inverse Document Frequency).
    - **Analisis Ganda:** Model menganalisis susunan **Kata** (Word N-Grams) sekaligus struktur **Karakter** (Character N-Grams).
    - **Skala Analisis:** >200.000 kombinasi linguistik unik diuji dalam sepersekian detik.
    
    Model dilatih mengenali karakteristik ulasan bot (seperti minim sentimen spesifik, pengulangan frasa, dan huruf kapital/tanda baca spam) untuk membedakannya dengan ulasan tulisan manusia asli.
    """)

st.markdown("<hr>", unsafe_allow_html=True)

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
