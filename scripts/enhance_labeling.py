import pandas as pd
import os
import re

def generate_enhanced_labeled_dataset():
    print("="*60)
    print("ENHANCED WEAK SUPERVISION: PELABELAN DATASET TOKOPEDIA v2.0")
    print("="*60)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(BASE_DIR, '..')
    RAW_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'tokopedia_product_reviews.csv')
    PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'tokopedia_labeled.csv')
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Dataset mentah tidak ditemukan di {RAW_DATA_PATH}")
        return
        
    print("[1] Memuat dataset mentah Tokopedia...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    print("[2] Inisialisasi Label...")
    # Inisialisasi ulasan asli sebagai 0 (Asli)
    df['label'] = 0
    
    print("[3] Menerapkan Enhanced Heuristic Rules...")
    
    # --- RULE 1: Deteksi Duplikasi Teks (Spamming) ---
    text_counts = df['review_text'].value_counts()
    duplicates = text_counts[text_counts > 2].index
    df.loc[df['review_text'].isin(duplicates), 'label'] = 1
    rule1_count = df['label'].sum()
    print(f"    -> Rule 1 (Spamming Duplikasi) mendeteksi: {rule1_count} ulasan palsu")
    
    # --- RULE 2: Pola Manipulatif (Rating Tinggi + Teks Terlalu Pendek) ---
    # Ulasan rating 5 tapi sangat pendek biasanya berupa bot pengisi ulasan kosong
    df.loc[(df['rating'] == 5) & (df['review_text'].str.len() < 15), 'label'] = 1
    rule2_count = df['label'].sum() - rule1_count
    print(f"    -> Rule 2 (Rating 5 + Teks Pendek) mendeteksi: {rule2_count} ulasan palsu tambahan")
    
    # --- RULE 3: Analisis Lonjakan (Review Burst) ---
    burst_check = df.groupby(['product_id', 'review_date']).size().reset_index(name='daily_count')
    anomalous_products = burst_check[burst_check['daily_count'] > 10]
    
    for _, row in anomalous_products.iterrows():
        df.loc[(df['product_id'] == row['product_id']) & 
               (df['review_date'] == row['review_date']), 'label'] = 1
    rule3_count = df['label'].sum() - rule1_count - rule2_count
    print(f"    -> Rule 3 (Review Burst / Lonjakan Harian) mendeteksi: {rule3_count} ulasan palsu tambahan")

    # --- RULE 4: Mismatch Sentimen & Rating (Sarcasm/Rating Anomaly) ---
    # Rating 5 tapi sentimen negatif, atau rating 1-2 tapi sentimen positif
    mismatch_cond = (
        ((df['rating'] == 5) & (df['sentiment_label'] == 'negative')) |
        ((df['rating'].isin([1, 2])) & (df['sentiment_label'] == 'positive'))
    )
    df.loc[mismatch_cond, 'label'] = 1
    rule4_count = df['label'].sum() - rule1_count - rule2_count - rule3_count
    print(f"    -> Rule 4 (Mismatch Sentimen & Rating) mendeteksi: {rule4_count} ulasan palsu tambahan")

    # --- RULE 5: Tanda Baca Berlebihan & Rasio Kapitalisasi Anomali ---
    def check_text_anomaly(text):
        if not isinstance(text, str) or len(text) == 0:
            return False
        # Tanda seru/tanya berulang (misal: "!!!", "???")
        exclamations = len(re.findall(r'[!?]{3,}', text)) > 0
        # Kapitalisasi huruf besar semua untuk ulasan cukup panjang
        uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        capital_anomaly = len(text) > 20 and uppercase_ratio > 0.8
        return exclamations or capital_anomaly

    anomaly_mask = df['review_text'].apply(check_text_anomaly)
    df.loc[anomaly_mask, 'label'] = 1
    rule5_count = df['label'].sum() - rule1_count - rule2_count - rule3_count - rule4_count
    print(f"    -> Rule 5 (Tanda Baca/Kapitalisasi Anomali) mendeteksi: {rule5_count} ulasan palsu tambahan")
    
    print("\n[4] Evaluasi Hasil Pelabelan Akhir:")
    distribusi = df['label'].value_counts()
    print(f"    - Ulasan Asli (0) : {distribusi.get(0, 0)}")
    print(f"    - Ulasan Palsu (1): {distribusi.get(1, 0)}")
    
    print("\n[5] Menyimpan dataset berlabel baru...")
    if not os.path.exists(os.path.dirname(PROCESSED_DATA_PATH)):
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH))
        
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"    Selesai! File berhasil diperbarui di: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    generate_enhanced_labeled_dataset()
