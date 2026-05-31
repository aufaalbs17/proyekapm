from flask import Flask, request, jsonify, render_template
import joblib
import os
import re
import math
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# NLTK resources
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

# Setup path to import scripts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, '../scripts')
MODEL_DIR = os.path.join(BASE_DIR, '../models')
sys.path.append(SCRIPTS_DIR)

# Load text-only model
tfidf_model = joblib.load(os.path.join(MODEL_DIR, 'tfidf_model.pkl'))
svm_model   = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))

# ── Preprocessing Pipeline ────────────────────────────────────
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text) # Merampingkan huruf berulang (contoh: mantaaappp -> mantaapp)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess(text):
    steps = []
    steps.append({"step": "Teks Asli", "value": text})

    cleaned = clean_text(text)
    steps.append({"step": "Cleaning (Hapus Simbol)", "value": cleaned})

    folded = cleaned.lower()
    steps.append({"step": "Case Folding (Huruf Kecil)", "value": folded})

    tokens = word_tokenize(folded)
    steps.append({"step": "Tokenization", "value": str(tokens[:15]) + ("..." if len(tokens) > 15 else "")})

    # Impor dinamis fungsi dari text_preprocessing
    from text_preprocessing import remove_stopwords, stem_text
    
    no_stop = remove_stopwords(tokens, 'indonesian')
    steps.append({"step": "Stopword Removal (Indonesian)", "value": str(no_stop[:15]) + ("..." if len(no_stop) > 15 else "")})

    stemmed = stem_text(no_stop, 'indonesian')
    steps.append({"step": "Stemming (Sastrawi)", "value": str(stemmed[:15]) + ("..." if len(stemmed) > 15 else "")})

    final = " ".join(stemmed)
    steps.append({"step": "Final Text", "value": final})

    return final, steps

def get_explanation(text, tfidf_model, svm_model, top_n=5):
    try:
        vec = tfidf_model.transform([text])
        coef = svm_model.coef_[0] if len(svm_model.coef_.shape) > 1 else svm_model.coef_
        feature_names = tfidf_model.get_feature_names_out()
        
        # Ekstraksi indices dan nilai
        indices = vec.indices
        values = vec.data
            
        contributions = []
        for idx, val in zip(indices, values):
            name = feature_names[idx]
            
            # Abaikan char n-grams agar penjelasan kata lebih bersih
            if 'char__' in name:
                continue
            name = name.split('__')[-1] if '__' in name else name
                
            score = float(val * coef[idx])
            contributions.append((name, score))
            
        contributions.sort(key=lambda x: x[1])
        
        # Koefisien positif memperkuat label 1 (Palsu), negatif memperkuat 0 (Asli)
        fake_reasons = [{"feature": n, "score": s} for n, s in sorted(contributions, key=lambda x: x[1], reverse=True)[:top_n] if s > 0]
        real_reasons = [{"feature": n, "score": s} for n, s in sorted(contributions, key=lambda x: x[1])[:top_n] if s < 0]
                
        return {"fake": fake_reasons, "real": real_reasons}
    except Exception as e:
        print(f"Error in get_explanation: {e}")
        return {"fake": [], "real": []}

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong'}), 400

    # Heuristic Filter untuk teks asal-asalan / ketikan acak ekstrim
    words = text.split()
    if len(text) < 2 or any(len(w) > 25 for w in words):
        steps = [{"step": "Heuristic Filter", "value": "Teks diblokir karena terlalu pendek (kurang dari 2 karakter) atau terindikasi ketikan acak."}]
        explanation = {"fake": [{"feature": "Keyboard Smash / Spam", "score": 99.9}], "real": []}
        return jsonify({'result': 'fake', 'label': '1', 'confidence': 99.9, 'steps': steps, 'explanation': explanation})

    try:
        # Menjalankan Preprocessing Bahasa Indonesia Murni secara lokal
        final_text, steps = preprocess(text)
        
        # Prediksi berbasis teks murni
        vec = tfidf_model.transform([final_text])
        decision = svm_model.decision_function(vec)[0]
        prob = 1 / (1 + math.exp(-decision)) # Sigmoid mapping
        label  = int(svm_model.predict(vec)[0])
        label_str = str(label)
        
        result = 'fake' if label == 1 else 'real'
        confidence = prob * 100 if label == 1 else (1 - prob) * 100
        
        explanation = get_explanation(final_text, tfidf_model, svm_model)
        return jsonify({'result': result, 'label': label_str, 'confidence': round(confidence, 1), 'steps': steps, 'explanation': explanation})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
