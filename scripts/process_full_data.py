import pandas as pd
import time
from tqdm import tqdm
from text_preprocessing import preprocess_text

# Aktifkan progress bar untuk pandas apply
tqdm.pandas()

def process_fake_reviews():
    print("="*50)
    print("MEMPROSES DATASET TOKOPEDIA Labeled (Bahasa Indonesia)")
    print("="*50)
    
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(BASE_DIR, '..')
    INPUT_FILE = os.path.join(ROOT_DIR, 'data', 'processed', 'tokopedia_labeled.csv')
    
    if not os.path.exists(INPUT_FILE):
        print(f"File tidak ditemukan: {INPUT_FILE}")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    # Hapus baris yang teksnya kosong
    df = df.dropna(subset=['review_text'])
    
    print("Memulai preprocessing NLP (Cleaning, Fold, Token, Stopword, Stemming)...")
    # Terapkan fungsi preprocess bahasa Indonesia ke seluruh kolom 'review_text'
    start_time = time.time()
    df['review_text_clean'] = df['review_text'].progress_apply(lambda x: preprocess_text(str(x), 'indonesian'))
    print(f"Selesai dalam {time.time() - start_time:.2f} detik.")
    
    # Simpan ke CSV baru
    output_name = os.path.join(ROOT_DIR, 'data', 'processed', 'tokopedia_preprocessed.csv')
    df.to_csv(output_name, index=False)
    print(f"Data berhasil disimpan ke: {output_name}\n")

if __name__ == "__main__":
    # Eksekusi penuh untuk training data
    process_fake_reviews()
