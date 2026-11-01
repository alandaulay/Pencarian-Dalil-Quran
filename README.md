#  Sistem Pencarian Dalil Al-Qur'an (Information Retrieval)

Aplikasi Temu Kembali Informasi (*Information Retrieval*) untuk mencari dalil ayat Al-Qur'an yang paling bersesuaian dengan dokumen input (seperti artikel berita, esai, atau pertanyaan tematik) menggunakan pendekatan **Vector Space Model (VSM)** dan pembobotan **TF-IDF (Term Frequency-Inverse Document Frequency)** dengan **Cosine Similarity**.

---

## Fitur Utama
1. **Dataset Al-Qur'an Lengkap:**
   - 114 Surah, 6.236 Ayat lengkap dengan teks Arab dan terjemahan resmi Kementerian Agama RI (Kemenag).
2. **Text Preprocessing Lengkap:**
   - *Case Folding & Normalisasi Ejaan Serapan Islami* (misal: *shalat* $\rightarrow$ *salat*, *ramadhan* $\rightarrow$ *ramadan*).
   - *Pembersihan Teks (Cleaning):* Menghilangkan tanda baca, karakter khusus, dan angka.
   - *Tokenizing:* Pemecahan kata.
   - *Stopword Removal:* Gabungan daftar stopword Sastrawi dan NLTK Bahasa Indonesia (770+ kata stopword).
   - *Stemming:* Algoritma Nazief-Adriani melalui pustaka **Sastrawi** dengan optimasi *dictionary caching* untuk kecepatan komputasi tinggi.
3. **Pencarian VSM & TF-IDF (Cosine Similarity):**
   - Menghitung kemiripan sudut kosinus antara vektor dokumen input dan seluruh 6.236 vektor ayat.
   - Output ranking terurut dengan tabel Top-5 sesuai kriteria: `Rank`, `Surah & Ayat`, `Score`, `Teks Ayat & Terjemahan`.
   - Kartu visual ayat dengan tipografi kaligrafi Arab dan penjelasan kata kunci yang cocok (*explainability*).
4. **Evaluasi Kinerja Model (Benchmark Kuantitatif):**
   - Menghitung **Precision@5**, **Recall@5**, **F1-Score**, dan **Mean Reciprocal Rank (MRR)** pada 8 set query tematik kanonikal.
   - Memberikan penjelasan akademik dan alasan pemilihan masing-masing metrik.
5. **Laporan Analisis 6 Pertanyaan Evaluasi:**
   - Analisis mendalam mengenai relevansi ranking, pendorong skor tinggi, missing relevance (*false negative*), *false positive*, kelebihan/kekurangan TF-IDF, serta pengaruh karakteristik Bahasa Indonesia.
6. **Web App Interaktif (Streamlit):**
   - Antarmuka web modern, responsif, dan mudah dijalankan secara lokal.

---

## Struktur Direktori Proyek

```text
pencarian-dalil-quran/
│
├── data/
│   ├── quran_raw.json           # Raw JSON Al-Qur'an (24 MB)
│   ├── quran_dataset.json       # Dataset bersih 6.236 ayat terindeks
│   └── stem_cache.json          # Cache hasil stemming Sastrawi (5.600+ lema)
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py          # Modul pembersihan, normalisasi serapan, stopwords & stemming
│   ├── vsm_tfidf.py             # Engine VSM, TF-IDF, Cosine Similarity, & Top-K ranking
│   ├── evaluation.py            # Modul benchmarking & perhitungan metrik (P@5, R@5, F1, MRR)
│   └── analysis_report.py       # Laporan analisis akademik 6 pertanyaan evaluasi tugas
│
├── app.py                       # Aplikasi web utama berbasis Streamlit
├── pencarian_dalil_quran.ipynb  # Jupyter Notebook lengkap dengan kode & cell markdown analisis
├── run_preprocessing.py         # Skrip pembuatan/pembaruan dataset terproses
├── test_search.py               # Skrip pengujian mandiri di terminal
└── README.md                    # Dokumentasi lengkap proyek
```

---

## Panduan Menjalankan Aplikasi

### 1. Prasyarat Lingkungan
Pastikan Python 3.9+ sudah terpasang. Dependensi yang dibutuhkan:
```bash
pip install streamlit scikit-learn Sastrawi nltk pandas numpy
```

### 2. Menjalankan Web App Streamlit
Jalankan perintah berikut di terminal pada folder proyek:
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

### 3. Menjalankan Jupyter Notebook
Jika ingin menjalankan dalam bentuk notebook:
```bash
jupyter notebook pencarian_dalil_quran.ipynb
```

### 4. Menjalankan Pengujian di Terminal
Untuk memeriksa hasil pencarian dan evaluasi secara cepat lewat terminal:
```bash
python test_search.py
```

---

## Hasil Evaluasi Model

| Metrik Evaluasi | Nilai Rata-rata | Penjelasan Singkat |
| :--- | :---: | :--- |
| **MRR (Mean Reciprocal Rank)** | **1.0000** | Ayat primer yang paling relevan selalu berada di **Rank 1** untuk seluruh query benchmark. |
| **Precision@5** | **45.00%** | Hampir setengah dari Top-5 ayat yang ditampilkan merupakan ayat kanonikal relevan. |
| **Recall@5** | **46.13%** | Mampu menjaring 46% dari seluruh ayat terkait yang tersebar dalam 6.236 ayat Al-Qur'an. |
| **F1-Score** | **44.79%** | Keseimbangan harmonik yang kuat antara presisi dan kelengkapan. |

---

## Ringkasan 6 Pertanyaan Evaluasi Tugas

1. **Relevansi Ranking:**  
   Metode VSM TF-IDF berhasil menempatkan ayat paling primer (seperti QS Al-Baqarah [2:183] untuk puasa) tepat di Rank 1 dengan skor Cosine Similarity tertinggi (0.6159), menghasilkan skor MRR sempurna 1.0.
2. **Penyebab Skor Tinggi:**  
   Skor tinggi didorong oleh tingginya bobot IDF pada kata kunci langka (*ramadan*, *riba*), kepadatan interseksi token (*lexical overlap*), dan normalisasi panjang dokumen L2-norm yang memberi bobot optimal pada ayat pendek yang padat.
3. **Missing Relevance (False Negatives):**  
   Terjadi akibat *Vocabulary Mismatch Problem*, di mana terjemahan Al-Qur'an menggunakan ungkapan metaforis atau sinonim deskriptif (misal: *"jangan membentak ibu bapak"* pada QS 17:23 alih-alih memakai kata eksplisit *"berbakti"*).
4. **False Positive (Noise):**  
   Ayat pendek yang memuat kata umum (misal QS Yasin [36:17] memuat kata *"wajib"*) dapat lolos ke peringkat bawah Top-5 karena norma Euclidean kecil dan hilangnya konteks sintaksis pada model *Bag-of-Words*.
5. **Kelebihan & Kekurangan TF-IDF:**  
   *Kelebihan:* Komputasi sangat cepat (<20 ms), transparan (*interpretable*), tanpa kebutuhan data latih beranotasi.  
   *Kekurangan:* Buta semantik (*semantic blindness*), rentan terhadap variasi sinonim, dan mengabaikan tata bahasa kalimat.
6. **Pengaruh Karakteristik Bahasa Indonesia:**  
   Sifat aglutinatif Bahasa Indonesia membutuhkan stemming akurat (Sastrawi) agar kata berimbuhan (*berpuasa*, *kewajiban*) kembali ke kata dasar (*puasa*, *wajib*). Selain itu, variasi ejaan serapan Arab (*shalat* vs *salat*, *ramadhan* vs *ramadan*) harus ditangani melalui normalisasi ejaan sebelum pembobotan TF-IDF.
