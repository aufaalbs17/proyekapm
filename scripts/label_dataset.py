import pandas as pd
import os

def generate_labeled_dataset():
    print("="*60)
    print("WEAK SUPERVISION: PELABELAN OTOMATIS DATASET TOKOPEDIA")
    print("="*60)
    
    # Menentukan path yang tepat
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(BASE_DIR, '..')
    RAW_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'tokopedia_product_reviews.csv')
    PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'tokopedia_labeled.csv')
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Dataset mentah tidak ditemukan di {RAW_DATA_PATH}")
        print("Silakan masukkan file 'tokopedia_product_reviews.csv' ke folder 'data/raw/'.")
        return
        
    print("[1] Memuat dataset mentah Tokopedia...")
    df = pd.read_csv(RAW_DATA_PATH)
    total_awal = len(df)
    
    print("[2] Inisialisasi Label...")
    # Inisialisasi semua label sebagai 'Asli' (0) terlebih dahulu
    df['label'] = 0
    
    print("[3] Menerapkan Heuristic Rules (Weak Supervision)...")
    
    # --- RULE 1: Deteksi Duplikasi Teks (Spamming) ---
    text_counts = df['review_text'].value_counts()
    duplicates = text_counts[text_counts > 2].index
    df.loc[df['review_text'].isin(duplicates), 'label'] = 1
    
    # --- RULE 2: Pola Manipulatif (Rating Tinggi + Teks Terlalu Pendek) ---
    df.loc[(df['rating'] == 5) & (df['review_text'].str.len() < 15), 'label'] = 1
    
    # --- RULE 3: Analisis Lonjakan (Review Burst) ---
    burst_check = df.groupby(['product_id', 'review_date']).size().reset_index(name='daily_count')
    anomalous_products = burst_check[burst_check['daily_count'] > 10]
    
    for _, row in anomalous_products.iterrows():
        df.loc[(df['product_id'] == row['product_id']) & 
               (df['review_date'] == row['review_date']), 'label'] = 1

    print("[4] Evaluasi Hasil Pelabelan:")
    distribusi = df['label'].value_counts()
    print(f"    - Ulasan Asli (0) : {distribusi.get(0, 0)}")
    print(f"    - Ulasan Palsu (1): {distribusi.get(1, 0)}")
    
    # 3. Simpan dataset baru yang sudah berlabel
    print("\n[5] Menyimpan dataset berlabel...")
    if not os.path.exists(os.path.dirname(PROCESSED_DATA_PATH)):
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH))
        
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"    Selesai! File disimpan di: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    generate_labeled_dataset()
