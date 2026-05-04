import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def run_svm_training():
    print("="*60)
    print("PROGRESS REPORT 3: SVM MODEL TRAINING & EVALUATION")
    print("="*60)
    
    # 1. Memuat dataset bersih
    print("\n[1] Memuat dataset 'fake_reviews_preprocessed.csv'...")
    df = pd.read_csv('fake_reviews_preprocessed.csv')
    df = df.dropna(subset=['text_clean'])
    
    # X adalah data teks, y adalah label (OR = Original, CG = Computer Generated/Fake)
    X = df['text_clean']
    y = df['label']
    
    # 2. Split Data (Pembagian Data Latih dan Data Uji)
    # 80% untuk Train (Belajar), 20% untuk Test (Ujian)
    print("[2] Membagi dataset menjadi 80% Data Latih dan 20% Data Uji...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    from sklearn.pipeline import FeatureUnion
    from sklearn.linear_model import RidgeClassifier
    
    # 3. TF-IDF Feature Extraction (Feature Union: Word + Character N-Grams)
    print("[3] Mengekstrak fitur TF-IDF (Word & Char N-Grams) dari Data Latih...")
    word_vectorizer = TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, min_df=2, analyzer='word')
    char_vectorizer = TfidfVectorizer(ngram_range=(2,5), sublinear_tf=True, min_df=3, analyzer='char_wb', max_features=50000)
    
    tfidf_vectorizer = FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer)
    ])
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # 4. Inisialisasi dan Pelatihan Model (RidgeClassifier)
    print("\n[4] Memulai pelatihan Model Super-Optimized (RidgeClassifier)...")
    print("    Mencari hyperplane terbaik dengan ~200k+ fitur (Kombinasi Kata & Karakter)...")
    svm_model = RidgeClassifier(random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    print("    Pelatihan Selesai!")
    
    # 5. Prediksi dan Pengujian Performa
    print("\n[5] Melakukan Prediksi pada Data Uji (Testing)...")
    y_pred = svm_model.predict(X_test_tfidf)
    
    # Menghitung Metrik
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("HASIL EVALUASI PERFORMA MODEL SVM:")
    print("="*60)
    print(f"AKURASI MODEL (Accuracy) : {acc * 100:.2f}%\n")
    
    print("--- Laporan Klasifikasi Rinci (Precision, Recall, F1-Score) ---")
    print(classification_report(y_test, y_pred, target_names=['Fake (CG)', 'Original (OR)']))
    
    print("--- Confusion Matrix (Matriks Kebingungan) ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Benar menebak Fake (CG)     : {cm[0][0]}")
    print(f"Salah tebak Asli padahal Fake: {cm[0][1]}")
    print(f"Salah tebak Fake padahal Asli: {cm[1][0]}")
    print(f"Benar menebak Asli (OR)     : {cm[1][1]}\n")
    
    # 6. Menyimpan Model untuk di-deploy ke Web
    joblib.dump(tfidf_vectorizer, 'tfidf_model.pkl')
    joblib.dump(svm_model, 'svm_model.pkl')
    print("[6] Model SVM dan TF-IDF Vectorizer telah berhasil disimpan (.pkl).")

if __name__ == "__main__":
    run_svm_training()
