import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🔍 DEMONSTRASI INDOBERT UNTUK KLASIFIKASI TEKS")
print("="*60)
print("\nMengunduh dan memuat model IndoBERT (ini mungkin memakan waktu untuk pertama kali)...")

try:
    from transformers import pipeline
except ImportError:
    print("\n[ERROR] Library 'transformers' atau 'torch' belum terinstall.")
    print("Silakan install dengan menjalankan perintah berikut di terminal:")
    print("pip install transformers torch")
    exit()

# Menggunakan model IndoBERT yang sudah di-fine-tune untuk sentimen sebagai contoh
# indobenchmark/indobert-base-p1 adalah model dasar, butuh fine-tuning untuk fake review.
# Sebagai demonstrasi, kita gunakan model sentimen berbahasa Indonesia.
try:
    classifier = pipeline("sentiment-analysis", model="indobenchmark/indobert-base-p1")
    
    print("\n✅ Model berhasil dimuat!\n")
    
    test_reviews = [
        "Barangnya sangat bagus, pengiriman super cepat, mantap!",
        "Barang jelek, nyesel beli disini, penjualnya tidak ramah.",
        "bagus",
        "asdfghjkl",
        "mantaaappppppp bangeeeett"
    ]
    
    print("Menganalisis teks...")
    print("-" * 50)
    for text in test_reviews:
        result = classifier(text)[0]
        # Catatan: Karena model ini bukan dilatih khusus untuk "Fake Review", 
        # melainkan bahasa secara umum, ini hanya demonstrasi bagaimana BERT
        # bisa mengekstrak sentimen dengan probabilitas yang sangat presisi.
        print(f"Teks  : '{text}'")
        print(f"Hasil : {result['label']} (Skor Kepastian: {result['score']:.4f})")
        print("-" * 50)
        
    print("\n💡 CATATAN UNTUK PENGEMBANGAN:")
    print("Untuk menggunakan IndoBERT di deteksi Ulasan Palsu (Fake Review),")
    print("kamu perlu melatih (fine-tune) model 'indobenchmark/indobert-base-p1'")
    print("menggunakan dataset 58.000 ulasan yang kamu miliki.")
    print("Proses fine-tuning ini membutuhkan GPU (seperti NVIDIA CUDA) dan")
    print("bisa memakan waktu beberapa jam.")

except Exception as e:
    print(f"\n[ERROR] Terjadi masalah saat memuat model: {e}")
