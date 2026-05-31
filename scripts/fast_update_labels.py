import pandas as pd
import os

def fast_update_labels():
    print("="*60)
    print("FAST UPDATE LABELS: PENGGABUNGAN LABEL DATASET PREPROCESSED")
    print("="*60)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(BASE_DIR, '..')
    LABELED_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'tokopedia_labeled.csv')
    PREPROCESSED_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'tokopedia_preprocessed.csv')
    
    if not os.path.exists(LABELED_PATH):
        print(f"Error: tokopedia_labeled.csv tidak ditemukan di {LABELED_PATH}")
        return
    if not os.path.exists(PREPROCESSED_PATH):
        print(f"Error: tokopedia_preprocessed.csv tidak ditemukan di {PREPROCESSED_PATH}")
        return
        
    print("[1] Memuat dataset labeled dan preprocessed...")
    df_labeled = pd.read_csv(LABELED_PATH)
    df_prep = pd.read_csv(PREPROCESSED_PATH)
    
    print(f"    - Baris Labeled: {len(df_labeled)}")
    print(f"    - Baris Preprocessed: {len(df_prep)}")
    
    print("[2] Melakukan merging label berdasarkan 'review_id'...")
    # Menghapus kolom label lama di df_prep jika ada, untuk menghindari kolom ganda
    if 'label' in df_prep.columns:
        df_prep = df_prep.drop(columns=['label'])
        
    # Mengambil review_id dan label baru saja
    df_new_labels = df_labeled[['review_id', 'label']]
    
    # Gabungkan label baru berdasarkan review_id
    df_prep_updated = pd.merge(df_prep, df_new_labels, on='review_id', how='inner')
    
    print(f"    - Jumlah baris setelah penggabungan: {len(df_prep_updated)}")
    dist = df_prep_updated['label'].value_counts()
    print(f"    - Distribusi Label Baru:")
    print(f"      * Asli (0) : {dist.get(0, 0)}")
    print(f"      * Palsu (1): {dist.get(1, 0)}")
    
    print("[3] Menyimpan data terupdate...")
    df_prep_updated.to_csv(PREPROCESSED_PATH, index=False)
    print(f"    Selesai! tokopedia_preprocessed.csv berhasil diperbarui di: {PREPROCESSED_PATH}")

if __name__ == "__main__":
    fast_update_labels()
