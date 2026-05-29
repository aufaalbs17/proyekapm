# Sistem Deteksi Ulasan Palsu (Fake Review Detection)

Proyek ini bertujuan untuk mendeteksi ulasan palsu (_fake reviews_) pada produk e-commerce (Tokopedia) menggunakan pendekatan Machine Learning (Support Vector Machine) dan teknik Natural Language Processing (NLP).

## Struktur Proyek

- `data/raw/`: Tempat menyimpan dataset mentah (`tokopedia_product_reviews.csv`).
- `data/processed/`: Tempat menyimpan dataset yang telah dilabeli dan diproses (`tokopedia_labeled.csv` dan `tokopedia_preprocessed.csv`).
- `scripts/`: Kode sumber Python untuk pelabelan, preprocessing, dan pelatihan model.
- `models/`: Tempat menyimpan model Machine Learning (`svm_model.pkl`) dan Vectorizer (`tfidf_model.pkl`) yang sudah dilatih dan siap digunakan untuk prediksi.
- `app/`: Aplikasi web Flask (backend di `server.py`, UI di `templates/index.html`).

## Cara Menjalankan Proyek

Pastikan Anda berada di dalam folder utama `proyekapm` dan mengaktifkan _virtual environment_ jika ada.

### 1. Install Dependencies

Pastikan semua pustaka yang dibutuhkan sudah terinstal:

```bash
pip install pandas numpy scikit-learn nltk Sastrawi tqdm flask joblib deep-translator
```

### 2. Pelabelan Dataset (Weak Supervision)

Karena dataset awal tidak memiliki label "Palsu/Asli", kita melabelinya berdasarkan aturan heuristik (rating, panjang teks, dan duplikasi):

```bash
python scripts/label_dataset.py
```

_Output: `data/processed/tokopedia_labeled.csv`_

### 3. Pemrosesan Bahasa Alami (NLP Preprocessing)

Tahap ini membersihkan teks, menghapus karakter aneh, mengubah huruf kecil, dan membuang stopword bahasa Indonesia.

```bash
python scripts/process_full_data.py
```

_Output: `data/processed/tokopedia_preprocessed.csv`_

### 4. Pelatihan Model (Training)

Melatih model Support Vector Machine (SVM) menggunakan algoritma TF-IDF.

```bash
python scripts/train_svm.py
```

_Output: Menampilkan akurasi di terminal dan menyimpan file model di folder `models/`._

---

## Integrasi Web App (Flask)

Untuk menjalankan aplikasi web Flask, yang dibutuhkan sebagai **input** ke dalam web hanyalah:

1. **Teks Ulasan (Review Text)**.

Anda cukup mem-paste teks ulasan dari pembeli, lalu sistem web akan:

1. Menjalankan fungsi `process_text` (menghapus tanda baca, membersihkan teks).
2. Mengubah teks ke dalam angka menggunakan `tfidf_model.pkl`.
3. Memprediksi hasil "Asli" atau "Palsu" menggunakan `svm_model.pkl`.

### Menjalankan Web App (Flask)

```bash
python app/server.py
```

Lalu buka `http://localhost:5000` di browser.
