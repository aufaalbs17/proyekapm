import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from deep_translator import GoogleTranslator

# Unduh resource NLTK yang dibutuhkan
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Inisialisasi Stemmer untuk bahasa Inggris (NLTK)
english_stemmer = PorterStemmer()

# Inisialisasi Stemmer untuk bahasa Indonesia (Sastrawi)
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    indonesian_stemmer = factory.create_stemmer()
except ImportError:
    indonesian_stemmer = None

def translate_to_english(text):
    """Menerjemahkan teks bahasa Indonesia ke bahasa Inggris."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    try:
        # Gunakan Google Translate via deep-translator
        translator = GoogleTranslator(source='id', target='en')
        return translator.translate(text)
    except Exception as e:
        print(f"Error translation: {e}")
        return text

def clean_text(text):
    """Menghilangkan karakter spesial, angka, atau simbol yang tidak relevan."""
    if not isinstance(text, str):
        return ""
    # Menghapus semua karakter kecuali huruf alfabet dan spasi
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def case_folding(text):
    """Mengubah seluruh karakter teks menjadi huruf kecil (lowercase)."""
    return text.lower()

def tokenize_text(text):
    """Memecah kalimat menjadi daftar kata-kata (Tokenization)."""
    return word_tokenize(text)

def remove_stopwords(tokens, language='english'):
    """Membuang kata hubung/umum yang tidak memiliki makna (Stopword Removal)."""
    if language == 'english':
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set(stopwords.words('indonesian'))
    
    return [word for word in tokens if word not in stop_words]

def stem_text(tokens, language='english'):
    """Mencari kata dasar dari setiap kata (Stemming)."""
    if language == 'english':
        # Menggunakan NLTK Porter Stemmer untuk bahasa Inggris
        return [english_stemmer.stem(word) for word in tokens]
    else:
        # Menggunakan Sastrawi untuk bahasa Indonesia
        if indonesian_stemmer:
            # Sastrawi lebih baik menerima bentuk string kalimat
            stemmed_sentence = indonesian_stemmer.stem(" ".join(tokens))
            return stemmed_sentence.split()
        else:
            return tokens

def preprocess_text(text, language='english'):
    """
    Menjalankan seluruh tahapan pipeline preprocessing.
    Return berupa list of tokens atau string tergabung (tergantung kebutuhan).
    Di sini dikembalikan sebagai string utuh agar mudah dimasukkan ke TF-IDF.
    """
    text = clean_text(text)
    text = case_folding(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens, language)
    tokens = stem_text(tokens, language)
    return " ".join(tokens)

# Contoh penggunaan jika dijalankan langsung
if __name__ == "__main__":
    contoh_en = "This is a REALLY bad product! 100% don't buy it... I am totally disappointed."
    print("Mentah EN:", contoh_en)
    print("Hasil EN :", preprocess_text(contoh_en, 'english'))
    
    contoh_id = "Barang ini jelek banget!! 100% nyesel beli di toko ini, pengirimannya lambat."
    print("\nMentah ID:", contoh_id)
    print("Hasil ID :", preprocess_text(contoh_id, 'indonesian'))
