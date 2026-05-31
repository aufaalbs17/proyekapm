import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def run_hybrid_svm_training():
    print("="*60)
    print("HYBRID MODEL TRAINING & EVALUATION (Text + Metadata)")
    print("="*60)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(BASE_DIR, '..')
    PREPROCESSED_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'tokopedia_preprocessed.csv')
    
    if not os.path.exists(PREPROCESSED_PATH):
        print(f"Error: tokopedia_preprocessed.csv tidak ditemukan di {PREPROCESSED_PATH}")
        return
        
    print("\n[1] Memuat dataset tokopedia_preprocessed.csv...")
    df = pd.read_csv(PREPROCESSED_PATH)
    
    # Isi baris kosong
    df['review_text'] = df['review_text'].fillna("")
    df['review_text_clean'] = df['review_text_clean'].fillna("")
    
    print("[2] Melakukan Feature Engineering untuk Metadata Numerik...")
    df['review_len'] = df['review_text'].apply(len)
    df['uppercase_ratio'] = df['review_text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    df['exclamation_count'] = df['review_text'].apply(lambda x: x.count('!'))
    df['rating'] = df['rating'].fillna(5.0)
    df['product_price'] = df['product_price'].fillna(0.0)
    
    # Menentukan fitur dan target
    feature_cols = ['review_text_clean', 'rating', 'review_len', 'uppercase_ratio', 'exclamation_count', 'product_price']
    X = df[feature_cols]
    y = df['label']
    
    print("[3] Membagi dataset menjadi 80% Data Latih dan 20% Data Uji...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[4] Membangun Pipeline Hybrid ColumnTransformer...")
    # TF-IDF Feature Extraction
    word_vectorizer = TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, min_df=2, analyzer='word')
    char_vectorizer = TfidfVectorizer(ngram_range=(2,5), sublinear_tf=True, min_df=3, analyzer='char_wb', max_features=50000)
    
    text_union = FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer)
    ])
    
    # Menggabungkan Fitur Teks dan Numerik
    preprocessor = ColumnTransformer([
        ('text', text_union, 'review_text_clean'),
        ('num', StandardScaler(), ['rating', 'review_len', 'uppercase_ratio', 'exclamation_count', 'product_price'])
    ])
    
    print("    Fitting ColumnTransformer pada Data Latih...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    print(f"    - Dimensi fitur akhir setelah transformasi: {X_train_transformed.shape[1]}")
    
    print("\n[5] Memulai pelatihan Model Super-Optimized RidgeClassifier...")
    svm_model = RidgeClassifier(random_state=42)
    svm_model.fit(X_train_transformed, y_train)
    print("    Pelatihan Selesai!")
    
    print("\n[6] Melakukan Prediksi dan Pengujian Performa...")
    y_pred = svm_model.predict(X_test_transformed)
    
    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*60)
    print("HASIL EVALUASI PERFORMA MODEL HYBRID:")
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
    
    # 7. Menyimpan model gabungan
    MODEL_DIR = os.path.join(ROOT_DIR, 'models')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print("[7] Menyimpan model optimal dan ColumnTransformer...")
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'tfidf_model.pkl'))
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'svm_model.pkl'))
    print("    Model berhasil disimpan di folder 'models/'!")

if __name__ == "__main__":
    run_hybrid_svm_training()
