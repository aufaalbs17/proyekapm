import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def run_text_svm_training():
    print("="*60)
    print("PURE TEXT MODEL TRAINING & EVALUATION (TF-IDF Only)")
    print("="*60)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(BASE_DIR, '..')
    PREPROCESSED_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'tokopedia_preprocessed.csv')
    
    if not os.path.exists(PREPROCESSED_PATH):
        print(f"Error: tokopedia_preprocessed.csv tidak ditemukan di {PREPROCESSED_PATH}")
        return
        
    print("\n[1] Memuat dataset tokopedia_preprocessed.csv...")
    df = pd.read_csv(PREPROCESSED_PATH)
    df = df.dropna(subset=['review_text_clean'])
    
    X = df['review_text_clean']
    y = df['label']
    
    print("[2] Membagi dataset menjadi 80% Data Latih dan 20% Data Uji...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[3] Mengekstrak fitur TF-IDF (Word & Char N-Grams)...")
    word_vectorizer = TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, min_df=2, analyzer='word')
    char_vectorizer = TfidfVectorizer(ngram_range=(2,5), sublinear_tf=True, min_df=3, analyzer='char_wb', max_features=50000)
    
    tfidf_vectorizer = FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer)
    ])
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"    - Dimensi fitur teks akhir: {X_train_tfidf.shape[1]}")
    
    print("\n[4] Memulai pelatihan Model RidgeClassifier (Teks Murni)...")
    svm_model = RidgeClassifier(random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    print("    Pelatihan Selesai!")
    
    print("\n[5] Melakukan Prediksi dan Pengujian Performa...")
    y_pred = svm_model.predict(X_test_tfidf)
    
    # Metrik
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*60)
    print("HASIL EVALUASI PERFORMA MODEL TEKS MURNI:")
    print("="*60)
    print(f"AKURASI MODEL (Accuracy) : {acc * 100:.2f}%\n")
    
    print("--- Laporan Klasifikasi Rinci (Precision, Recall, F1-Score) ---")
    print(classification_report(y_test, y_pred, target_names=['Asli (0)', 'Palsu (1)']))
    
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Benar menebak Asli (0)      : {cm[0][0]}")
    print(f"Salah tebak Palsu pdhl Asli : {cm[0][1]}")
    print(f"Salah tebak Asli pdhl Palsu : {cm[1][0]}")
    print(f"Benar menebak Palsu (1)     : {cm[1][1]}\n")
    
    # Menyimpan model
    MODEL_DIR = os.path.join(ROOT_DIR, 'models')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print("[6] Menyimpan model optimal dan TF-IDF Vectorizer...")
    joblib.dump(tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_model.pkl'))
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'svm_model.pkl'))
    print("    Model teks murni berhasil disimpan di folder 'models/'!")

if __name__ == "__main__":
    run_text_svm_training()
