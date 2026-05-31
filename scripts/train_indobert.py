import os
import pandas as pd
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:
    print("="*60)
    print("[ERROR] Pustaka PyTorch atau Transformers tidak ditemukan.")
    print("Skrip ini dirancang untuk dijalankan di lingkungan dengan GPU/CUDA.")
    print("Silakan jalankan perintah berikut untuk menginstal pustaka yang dibutuhkan:")
    print("pip install torch transformers scikit-learn pandas accelerate")
    print("="*60)
    exit()

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_indobert():
    print("="*60)
    print("TRAINING DEEP LEARNING MODEL: INDOBERT FOR FAKE REVIEWS")
    print("="*60)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(BASE_DIR, '..')
    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'tokopedia_preprocessed.csv')
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset preprocessed tidak ditemukan di {DATA_PATH}")
        return
        
    print("[1] Memuat dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['review_text', 'label'])
    
    # Ambil sampel subset jika ingin melatih lebih cepat (misal: 10,000 baris)
    # Untuk pelatihan penuh, gunakan seluruh baris: df
    print("    Mengambil 10,000 ulasan acak untuk optimasi waktu (jika melatih di CPU/Colab terbatas)...")
    df_sampled = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)
    
    X = df_sampled['review_text'].tolist()
    y = df_sampled['label'].tolist()
    
    print("[2] Membagi data latih dan data uji (80% Train, 20% Val)...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[3] Memuat IndoBERT Tokenizer...")
    model_name = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("    Tokenizing teks...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = ReviewDataset(train_encodings, train_labels)
    val_dataset = ReviewDataset(val_encodings, val_labels)
    
    print("[4] Menginisialisasi Model IndoBERT untuk Klasifikasi Sekuensial...")
    # Model memiliki 2 label: 0 (Asli), 1 (Palsu)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    print("[5] Mengatur Parameter Pelatihan (Training Arguments)...")
    output_dir = os.path.join(ROOT_DIR, 'models', 'indobert_output')
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\n[6] Memulai Pelatihan Model IndoBERT (Deep Learning)...")
    print("    CATATAN: Proses ini membutuhkan waktu beberapa menit s.d. jam tergantung GPU Anda.")
    trainer.train()
    
    print("\n[7] Menyimpan Model IndoBERT Terbaik...")
    model_save_path = os.path.join(ROOT_DIR, 'models', 'indobert_final')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"    Selesai! Model disimpan di: {model_save_path}")

if __name__ == "__main__":
    train_indobert()
