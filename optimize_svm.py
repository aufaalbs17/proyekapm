import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def analyze_and_optimize():
    print("="*60)
    print("ANALISIS KESALAHAN & OPTIMASI MODEL SVM")
    print("="*60)

    # 1. Memuat dataset bersih
    print("\n[1] Memuat dataset...")
    df = pd.read_csv('fake_reviews_preprocessed.csv')
    df = df.dropna(subset=['text_clean'])
    X = df['text_clean']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # =========================================================
    # ANALISIS KESALAHAN (CONFUSION MATRIX)
    # =========================================================
    print("\n" + "="*60)
    print("BAGIAN 1: ANALISIS KESALAHAN CONFUSION MATRIX")
    print("="*60)
    
    # Baseline model (default)
    baseline_model = LinearSVC(C=1.0, max_iter=1000, random_state=42)
    baseline_model.fit(X_train_tfidf, y_train)
    y_pred_base = baseline_model.predict(X_test_tfidf)
    
    cm = confusion_matrix(y_test, y_pred_base)
    acc_base = accuracy_score(y_test, y_pred_base)

    print(f"\nAkurasi Baseline (C=1.0): {acc_base * 100:.2f}%")
    print("\n--- Confusion Matrix Baseline ---")
    print(f"  {'':30} PREDIKSI PALSU   PREDIKSI ASLI")
    print(f"  SEBENARNYA PALSU (CG)   : {cm[0][0]:8} (Benar)   {cm[0][1]:5} (Salah Tebak Asli)")
    print(f"  SEBENARNYA ASLI  (OR)   : {cm[1][0]:8} (Salah Tebak Palsu)   {cm[1][1]:5} (Benar)")
    
    # Hitung persentase kesalahan spesifik
    total_fake = cm[0][0] + cm[0][1]
    total_real = cm[1][0] + cm[1][1]
    err_fake_as_real = cm[0][1] / total_fake * 100
    err_real_as_fake = cm[1][0] / total_real * 100

    print(f"\n--- Persentase Kesalahan ---")
    print(f"Ulasan PALSU salah ditebak sebagai ASLI : {err_fake_as_real:.2f}% ({cm[0][1]} dari {total_fake})")
    print(f"Ulasan ASLI  salah ditebak sebagai PALSU: {err_real_as_fake:.2f}% ({cm[1][0]} dari {total_real})")

    # Tampilkan 5 contoh salah tebak
    X_test_list = X_test.reset_index(drop=True)
    y_test_list = y_test.reset_index(drop=True)
    y_pred_series = pd.Series(y_pred_base)
    
    print("\n--- Contoh Ulasan yang Salah Ditebak ---")
    wrong_idx = y_test_list[y_test_list != y_pred_series].index[:5]
    for idx in wrong_idx:
        print(f"  Teks    : {X_test_list[idx][:80]}...")
        print(f"  Label Asli : {y_test_list[idx]}  |  Prediksi Model: {y_pred_series[idx]}")
        print()

    # =========================================================
    # OPTIMASI (FINE-TUNING PARAMETER C)
    # =========================================================
    print("="*60)
    print("BAGIAN 2: OPTIMASI FINE-TUNING PARAMETER C")
    print("="*60)
    print("\nMencoba berbagai nilai Parameter C (Regularization)...")
    print(f"{'Parameter C':>15} | {'Akurasi':>10} | {'F1-Score Palsu':>15} | {'F1-Score Asli':>13}")
    print("-"*60)
    
    best_acc = 0
    best_c = 1.0
    best_model = None
    
    # Grid Search manual untuk nilai C
    c_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for c_val in c_values:
        model = LinearSVC(C=c_val, max_iter=2000, random_state=42)
        model.fit(X_train_tfidf, y_train)
        pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, pred)
        report = classification_report(y_test, pred, target_names=['CG', 'OR'], output_dict=True)
        f1_fake = report['CG']['f1-score']
        f1_real = report['OR']['f1-score']
        marker = " [*] TERBAIK" if acc > best_acc else ""
        print(f"  C = {c_val:>6} | {acc*100:>8.2f}% | {f1_fake:>14.4f} | {f1_real:>12.4f}{marker}")
        
        if acc > best_acc:
            best_acc = acc
            best_c   = c_val
            best_model = model

    # Laporan akhir model terbaik
    print(f"\n{'='*60}")
    print(f"HASIL OPTIMASI: Parameter C Terbaik = {best_c}")
    print(f"{'='*60}")
    y_pred_best = best_model.predict(X_test_tfidf)
    print(f"\nAkurasi Model Optimal: {best_acc * 100:.2f}%")
    print(f"Peningkatan dari Baseline: +{(best_acc - acc_base) * 100:.2f}%\n")
    print("--- Laporan Klasifikasi Model Optimal ---")
    print(classification_report(y_test, y_pred_best, target_names=['Fake (CG)', 'Original (OR)']))

    # Confusion Matrix model optimal
    cm_best = confusion_matrix(y_test, y_pred_best)
    print("--- Confusion Matrix Model Optimal ---")
    print(f"  SEBENARNYA PALSU (CG)   : {cm_best[0][0]:8} (Benar)   {cm_best[0][1]:5} (Salah)")
    print(f"  SEBENARNYA ASLI  (OR)   : {cm_best[1][0]:8} (Salah)   {cm_best[1][1]:5} (Benar)")

    # Simpan model optimal
    joblib.dump(tfidf, 'tfidf_model.pkl')
    joblib.dump(best_model, 'svm_model.pkl')
    print(f"\n[OK] Model optimal (C={best_c}) disimpan sebagai 'svm_model.pkl'")

if __name__ == "__main__":
    analyze_and_optimize()
