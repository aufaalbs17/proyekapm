import pandas as pd
import time
from tqdm import tqdm
from text_preprocessing import preprocess_text

# Aktifkan progress bar untuk pandas apply
tqdm.pandas()

def process_fake_reviews():
    print("="*50)
    print("MEMPROSES FAKE REVIEWS DATASET (40.432 Baris)")
    print("="*50)
    
    df = pd.read_csv('fake reviews dataset.csv')
    
    # Hapus baris yang teksnya kosong
    df = df.dropna(subset=['text_'])
    
    print("Memulai preprocessing NLP (Cleaning, Fold, Token, Stopword, Stemming)...")
    # Terapkan fungsi preprocess bahasa inggris ke seluruh kolom 'text_'
    start_time = time.time()
    df['text_clean'] = df['text_'].progress_apply(lambda x: preprocess_text(x, 'english'))
    print(f"Selesai dalam {time.time() - start_time:.2f} detik.")
    
    # Simpan ke CSV baru
    output_name = 'fake_reviews_preprocessed.csv'
    df.to_csv(output_name, index=False)
    print(f"Data berhasil disimpan ke: {output_name}\n")

if __name__ == "__main__":
    # Eksekusi penuh untuk training data
    process_fake_reviews()
