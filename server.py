from flask import Flask, request, jsonify, render_template
import joblib
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
tfidf_model = joblib.load('tfidf_model.pkl')
svm_model   = joblib.load('svm_model.pkl')

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
        return jsonify({'result': result, 'label': label, 'steps': steps})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
