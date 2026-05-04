import pandas as pd
from text_preprocessing import preprocess_text

def run_progress_report_2():
    print("="*50)
    print("PROGRESS REPORT 2: TEXT PREPROCESSING (NLTK)")
    print("="*50)
    
    # 1. Dataset Fake Reviews (Bahasa Inggris)
    print("\n[1] Memproses Dataset: Fake Reviews (Bahasa Inggris)")
    try:
        df_fake = pd.read_csv('fake reviews dataset.csv')
        sample_fake = df_fake['text_'].dropna().head(3)
        
        for i, text in enumerate(sample_fake, 1):
            print(f"\n--- Data ke-{i} ---")
            print(f"Text Mentah   : {text}")
            
            # Step-by-step just for demonstration
            from text_preprocessing import clean_text, case_folding, tokenize_text, remove_stopwords, stem_text
            
            cleaned = clean_text(text)
            print(f"1. Cleaning   : {cleaned}")
            
            folded = case_folding(cleaned)
            print(f"2. Case Fold  : {folded}")
            
            tokens = tokenize_text(folded)
            print(f"3. Tokenize   : {tokens}")
            
            no_stop = remove_stopwords(tokens, 'english')
            print(f"4. Stopwords  : {no_stop}")
            
            stemmed = stem_text(no_stop, 'english')
            print(f"5. Stemming   : {stemmed}")
            
            final_text = " ".join(stemmed)
            print(f"-> FINAL TEXT : {final_text}")
            
    except Exception as e:
        print("Gagal memproses fake reviews:", e)


if __name__ == "__main__":
    run_progress_report_2()
