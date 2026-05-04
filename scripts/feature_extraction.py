import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def run_tfidf_extraction():
    print("="*50)
    print("PROGRESS REPORT 3: FEATURE EXTRACTION (TF-IDF)")
    print("="*50)
    
    # 1. Memuat dataset yang sudah bersih
    print("\n[1] Memuat dataset bersih: 'fake_reviews_preprocessed.csv'...")
    df = pd.read_csv('fake_reviews_preprocessed.csv')
    
    # Memastikan tidak ada nilai kosong (NaN) setelah preprocessing
    df = df.dropna(subset=['text_clean'])
    
    # 2. Inisialisasi TfidfVectorizer
    print("\n[2] Membangun Model TF-IDF...")
    # max_features membatasi jumlah kata (kolom) maksimal yang diambil,
    # misalnya mengambil 10.000 kata yang paling sering muncul di seluruh dataset
    # agar memori tidak meledak, namun tetap mempertahankan kata-kata paling penting.
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    
    # 3. Proses Fit & Transform (Mengubah Teks menjadi Matriks Angka)
    print("[3] Mengonversi teks menjadi representasi numerik (vektor)...")
    # tfidf_matrix akan berisi ribuan baris ulasan x ribuan kolom fitur kata
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_clean'])
    
    # 4. Menyimpan model TF-IDF ke file agar bisa dipakai di aplikasi Web nanti
    joblib.dump(tfidf_vectorizer, 'tfidf_model.pkl')
    
    # Menampilkan Hasil Pembobotan
    print("\n" + "="*50)
    print("HASIL EKSTRAKSI TF-IDF:")
    print("="*50)
    print(f"Total Ulasan yang Diproses : {tfidf_matrix.shape[0]:,} baris")
    print(f"Total Fitur/Kata Ditemukan : {tfidf_matrix.shape[1]:,} kolom kata (Vektor)")
    
    # Menampilkan beberapa contoh kata (Vocabulary) yang dijadikan fitur/kolom
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print("\nContoh 10 Kata (Fitur) Acak di dalam Vektor:")
    print(list(feature_names[500:510]))
    
    # Mendemonstrasikan hasil angka dari ulasan pertama
    print("\nContoh Pembobotan Angka pada Ulasan Pertama:")
    print(f"Teks Asli Bersih: '{df['text_clean'].iloc[0]}'")
    
    # Mengambil nilai TF-IDF bukan nol untuk ulasan pertama
    first_review_vector = tfidf_matrix[0].tocoo()
    print("Nilai TF-IDF (Hanya kata yang ada di kalimat tersebut yang nilainya > 0):")
    for col, data in zip(first_review_vector.col, first_review_vector.data):
        kata = feature_names[col]
        print(f"- Kata '{kata:10}': {data:.4f}")
        
    print("\nModel TF-IDF berhasil disimpan sebagai 'tfidf_model.pkl'")

if __name__ == "__main__":
    run_tfidf_extraction()
