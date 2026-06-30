# 💬 Tokopedia Product Sentiment Analysis: Indonesian Product Reviews

Klasifikasi sentimen **3 kelas** (positif, netral, negatif) pada ulasan produk e-commerce berbahasa Indonesia, menggunakan **TF-IDF + Complement Naive Bayes**. Inti project: **preprocessing teks bahasa Indonesia yang disesuaikan dengan karakter ulasan e-commerce** (normalisasi slang/typo dan penjagaan kata negasi).

## 📊 Dataset

- **File:** `review_product.csv`, berisi 40.597 ulasan produk (kolom utama: `text`, `rating`, plus metadata produk)
- **Sumber:** `⟨ISI SUMBER DATA, mis. link Kaggle / hasil scraping / dataset mata kuliah⟩`
- **Label** diturunkan dari `rating`:
  - rating >= 4 dilabeli `positif` (1)
  - rating = 3 dilabeli `netral` (0)
  - rating <= 2 dilabeli `negatif` (-1)
- **Distribusi (imbalanced berat):** positif 37.851, netral 1.825, negatif 921

## 🔍 Workflow

1. **Labeling:** Konversi `rating` bintang menjadi label sentimen 3 kelas.
2. **Text Preprocessing:** Pipeline khusus Bahasa Indonesia (lihat di bawah).
3. **Feature Extraction:** `TfidfVectorizer`.
4. **Imbalance Handling:** `RandomOverSampler` (kelas minoritas dinaikkan).
5. **Modeling:** Complement Naive Bayes (`alpha=0.1`).
6. **Evaluation & Inference:** Classification report dan prediksi pada data baru (`question.csv`).

## ⭐ Text Preprocessing (Bahasa Indonesia)

Bagian paling krusial: ulasan e-commerce penuh typo, singkatan, dan elongation. Pipeline `preprocessing()` melakukan:

- **Lowercasing** dan **normalisasi pengulangan huruf** (mis. `baguuuus` menjadi `bagus`)
- **Word elongation & slang** via `indoNLP` (`replace_word_elongation`, `replace_slang`)
- **Kamus normalisasi custom (~600+ entri):** memetakan ratusan variasi typo/slang khas ulasan ke bentuk baku (mis. `mantul`, `barangnyaa`, dan ratusan salah ketik `pengiriman` menjadi bentuk standar)
- **Tokenisasi** (`nltk`) dan pembuangan tanda baca
- **Stopword removal** dengan **penjagaan kata sentimen:** kata seperti `tidak`, `sangat`, `kurang`, `jangan`, `belum` **sengaja dipertahankan** (tidak dibuang seperti pada stopword list default), agar **makna negasi dan intensitas tidak hilang**.

> Penjagaan kata negasi ini penting: tanpa itu, "barang **tidak** bagus" akan tereduksi jadi "barang bagus", yang justru terbalik maknanya.

## 🤖 Model & Hasil

**Model:** Complement Naive Bayes (`alpha=0.1`) + RandomOverSampler + TF-IDF
**Accuracy:** **0.80**

| Kelas | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| negatif (-1) | 0.14 | 0.67 | 0.23 | 184 |
| netral (0) | 0.09 | 0.19 | 0.12 | 365 |
| positif (1) | 0.97 | 0.83 | 0.90 | 7.571 |
| **Macro avg** | 0.40 | 0.56 | 0.42 | 8.120 |
| **Weighted avg** | 0.91 | 0.80 | 0.85 | 8.120 |

## 📝 Catatan & Limitasi

Akurasi 0.80 **menyesatkan jika berdiri sendiri:** angka itu didominasi kelas positif yang sangat besar. **Macro-F1 hanya 0.42** karena model lemah pada kelas netral dan negatif, konsekuensi langsung dari *class imbalance* yang ekstrem (positif sekitar 40 kali lipat negatif). Perbaikan ke depan: resampling lebih agresif, pengumpulan lebih banyak ulasan negatif/netral, atau model yang lebih kuat menangani minoritas.

## ▶️ How to Run

Project dijalankan di **Google Colab**:
1. Upload `review_product.csv` (training) dan `question.csv` (data uji inference).
2. Klik `Runtime`, lalu `Run all`.

## 🧰 Tech Stack

Python, pandas, scikit-learn, imbalanced-learn, **Sastrawi** & **indoNLP** (NLP Bahasa Indonesia), nltk

## 👤 Author

**Muhammad Faishal Ardiansyah**, [@Ishalllll](https://github.com/Ishalllll)
