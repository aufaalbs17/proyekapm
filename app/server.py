from flask import Flask, request, jsonify, render_template
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from deep_translator import GoogleTranslator

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')

tfidf_model = joblib.load(os.path.join(MODEL_DIR, 'tfidf_model.pkl'))
svm_model   = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))

stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ── Preprocessing Pipeline ────────────────────────────────────
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess(text, translate=True):
    steps = []
    steps.append({"step": "Teks Asli", "value": text})

    if translate:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        steps.append({"step": "Translasi (ID→EN)", "value": translated})
    else:
        translated = text

    cleaned = clean_text(translated)
    steps.append({"step": "Cleaning", "value": cleaned})

    folded = cleaned.lower()
    steps.append({"step": "Case Folding", "value": folded})

    tokens = word_tokenize(folded)
    steps.append({"step": "Tokenization", "value": str(tokens[:15]) + ("..." if len(tokens) > 15 else "")})

    no_stop = [w for w in tokens if w not in stop_words]
    steps.append({"step": "Stopword Removal", "value": str(no_stop[:15]) + ("..." if len(no_stop) > 15 else "")})

    stemmed = [stemmer.stem(w) for w in no_stop]
    steps.append({"step": "Stemming", "value": str(stemmed[:15]) + ("..." if len(stemmed) > 15 else "")})

    final = " ".join(stemmed)
    steps.append({"step": "Final Text", "value": final})

    return final, steps

def get_explanation(text, tfidf_model, svm_model, top_n=5):
    vec = tfidf_model.transform([text])
    indices = vec.indices
    values = vec.data
    coef = svm_model.coef_
    feature_names = tfidf_model.get_feature_names_out()
    
    contributions = []
    for idx, val in zip(indices, values):
        if 'char__' in feature_names[idx]:
            continue
        score = float(val * coef[idx])
        name = feature_names[idx].split('__')[-1] if '__' in feature_names[idx] else feature_names[idx]
        contributions.append((name, score))
    
    contributions.sort(key=lambda x: x[1])
    
    fake_reasons = [{"feature": n, "score": s} for n, s in contributions[:top_n] if s < 0]
    real_reasons = [{"feature": n, "score": s} for n, s in sorted(contributions, key=lambda x: x[1], reverse=True)[:top_n] if s > 0]
            
    return {"fake": fake_reasons, "real": real_reasons}

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data      = request.get_json()
    text      = data.get('text', '').strip()
    translate = data.get('translate', True)

    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong'}), 400

    try:
        final_text, steps = preprocess(text, translate=translate)
        vec    = tfidf_model.transform([final_text])
        label  = svm_model.predict(vec)[0]
        result = 'fake' if label == 'CG' else 'real'
        explanation = get_explanation(final_text, tfidf_model, svm_model)
        return jsonify({'result': result, 'label': label, 'steps': steps, 'explanation': explanation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
