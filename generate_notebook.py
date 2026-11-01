import json
import os
from src.analysis_report import ANALYSIS_MARKDOWN

def make_cell(cell_type, source, outputs=None):
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        }
    elif cell_type == "code":
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": outputs or [],
            "source": [line + "\n" for line in source.split("\n")]
        }

def main():
    cells = []
    
    # 1. Judul & Identitas
    cells.append(make_cell("markdown", """# 📖 Tugas Information Retrieval: Pencarian Dalil Al-Qur'an
**Metode:** Vector Space Model (VSM) & Term Frequency-Inverse Document Frequency (TF-IDF)  
**Tujuan:** Menghitung kemiripan (Cosine Similarity) antara sebuah dokumen input (artikel/esai/berita) dengan korpus terjemahan Al-Qur'an Bahasa Indonesia (6.236 ayat, 114 surah)."""))

    # 2. Section 1: Library & Data Loading
    cells.append(make_cell("markdown", """## 1. Import Library & Persiapan Corpus Data
Menggunakan dataset Al-Qur'an lengkap (6.236 ayat) dengan teks Arab dan terjemahan resmi Bahasa Indonesia (Kemenag)."""))

    cells.append(make_cell("code", """import os
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords as nltk_stopwords

# Load corpus ayat Al-Qur'an yang telah diproses
data_path = 'data/quran_dataset.json'
with open(data_path, 'r', encoding='utf-8') as f:
    quran_data = json.load(f)

df_quran = pd.DataFrame(quran_data)
print(f"Total Ayat Berhasil Dimuat: {len(df_quran)} ayat dari {df_quran['surah_number'].nunique()} surah.")
df_quran[['surah_ayah', 'arabic_text', 'translation_id']].head(3)"""))

    # 3. Section 2: Preprocessing
    cells.append(make_cell("markdown", """## 2. Text Preprocessing Pipeline
Tahapan preprocessing teks terjemahan dan dokumen query:
1. **Case Folding & Normalisasi:** Mengubah teks ke huruf kecil dan menyelaraskan ejaan kata serapan Islami (misal: *shalat* -> *salat*, *ramadhan* -> *ramadan*).
2. **Cleaning:** Menghilangkan tanda baca, karakter khusus, dan angka.
3. **Tokenizing:** Memecah teks menjadi daftar kata.
4. **Stopword Removal:** Menghapus kata umum non-substantif (gabungan Sastrawi dan NLTK Bahasa Indonesia).
5. **Stemming:** Mengembalikan kata berimbuhan ke kata dasar menggunakan algoritma Nazief-Adriani (Sastrawi)."""))

    cells.append(make_cell("code", """from src.preprocessor import QuranPreprocessor

preprocessor = QuranPreprocessor()

# Uji coba preprocessing pada contoh teks artikel
sample_text = "Kewajiban ibadah puasa di bulan Ramadhan bagi orang-orang yang beriman agar bertakwa."
steps = preprocessor.preprocess_steps(sample_text)

print("Original Text   :", steps['original'])
print("Normalized      :", steps['normalized_loanwords'])
print("Cleaned         :", steps['cleaned'])
print("Tokenized       :", steps['tokenized'])
print("Stopwords Out   :", steps['filtered_stopwords'])
print("Stemmed Words   :", steps['stemmed'])
print("Final Clean Text:", steps['final_text'])"""))

    # 4. Section 3: VSM & TF-IDF Model
    cells.append(make_cell("markdown", """## 3. Pembobotan TF-IDF dan Vector Space Model (VSM)
- Setiap dokumen ayat $D_i$ dan dokumen query $Q$ direpresentasikan sebagai vektor bobot:
  $$w_{t,d} = \text{TF}(t, d) \times \text{IDF}(t)$$
  $$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
- Kemiripan antara query dan ayat dihitung menggunakan **Cosine Similarity**:
  $$\text{Cosine Similarity}(Q, D) = \frac{Q \cdot D}{\|Q\|_2 \times \|D\|_2}$$"""))

    cells.append(make_cell("code", """from src.vsm_tfidf import QuranSearchEngine

# Inisialisasi Search Engine VSM & TF-IDF
engine = QuranSearchEngine(preprocessor=preprocessor)
print(f"Dimensi Vektor Fitur (Vocabulary Size): {len(engine.feature_names)} term unik.")"""))

    # 5. Section 4: Pencarian Dalil & Tabel Top-5
    cells.append(make_cell("markdown", """## 4. Pencarian Dalil & Hasil Ranking Top-5
Sesuai kriteria tugas, hasil disajikan dalam tabel dengan kolom:
- **Rank**
- **Surah & Ayat**
- **Score** (Cosine Similarity)
- **Teks Ayat & Terjemahan**"""))

    cells.append(make_cell("code", """# Dokumen input berupa esai / pertanyaan pencarian dalil
dokumen_input = "Kewajiban ibadah puasa di bulan Ramadhan bagi orang-orang yang beriman"

# Eksekusi pencarian Top-5
hasil_top5 = engine.search(dokumen_input, top_k=5)

# Tampilkan tabel Top-5 sesuai kriteria tugas
tabel_tugas = hasil_top5[['Rank', 'Surah & Ayat', 'Score', 'Teks Ayat & Terjemahan']]
tabel_tugas"""))

    # 6. Section 5: Evaluasi Model
    cells.append(make_cell("markdown", """## 5. Evaluasi Kinerja Model Information Retrieval
Metrik yang dievaluasi:
1. **Precision@5**
2. **Recall@5**
3. **F1-Score**
4. **MRR (Mean Reciprocal Rank)**

Evaluasi diuji pada 8 set benchmark query tematik kanonikal yang memiliki daftar ayat ground-truth."""))

    cells.append(make_cell("code", """from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator(engine)
df_eval, summary = evaluator.evaluate_benchmark(top_k=5)

print("=" * 50)
print("RINGKASAN METRIK EVALUASI MODEL:")
print("=" * 50)
print(f"Mean Precision@5 : {summary['mean_precision_at_5']:.4f} ({summary['mean_precision_at_5'] * 100:.1f}%)")
print(f"Mean Recall@5    : {summary['mean_recall_at_5']:.4f} ({summary['mean_recall_at_5'] * 100:.1f}%)")
print(f"Mean F1-Score    : {summary['mean_f1_score']:.4f} ({summary['mean_f1_score'] * 100:.1f}%)")
print(f"MRR              : {summary['mrr']:.4f}")
print("=" * 50)

# Tampilkan rincian tabel per query
df_eval[['ID', 'Tema', 'First Hit Rank', 'Precision@5', 'Recall@5', 'F1-Score', 'Reciprocal Rank', 'Relevant Hits']]"""))

    # 7. Penjelasan Alasan Pemilihan Metrik
    cells.append(make_cell("markdown", """### Penjelasan & Alasan Pemilihan Metrik Evaluasi:
1. **Precision@5:**
   - *Rumus:* $\\text{Precision@5} = \\frac{|\\text{Top 5 Retrieved} \\cap \\text{Relevant}|}{5}$
   - *Alasan:* Dalam antarmuka pencarian ayat Al-Qur'an, pengguna manusia umumnya hanya membaca 5 rekomendasi teratas. Precision@5 mengukur seberapa bersih atau bebas dari noise (false positive) ayat yang disajikan pada layar utama.
2. **Recall@5:**
   - *Rumus:* $\\text{Recall@5} = \\frac{|\\text{Top 5 Retrieved} \\cap \\text{Relevant}|}{|\\text{Total Relevant Ground Truth}|}$
   - *Alasan:* Mengukur kelengkapan sistem. Suatu hukum/tema dalam Al-Qur'an sering kali tersebar di beberapa ayat. Recall@5 menguji apakah Top-5 mampu merangkum sebanyak mungkin dalil penting yang ada.
3. **F1-Score:**
   - *Rumus:* $\\text{F1} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$
   - *Alasan:* Rata-rata harmonik penyeimbang antara presisi dan kelengkapan. Menghindari sistem yang hanya mengejar presisi tinggi namun kehilangan banyak ayat penting, atau sebaliknya.
4. **Mean Reciprocal Rank (MRR):**
   - *Rumus:* $\\text{MRR} = \\frac{1}{|Q|} \\sum_{i=1}^{|Q|} \\frac{1}{\\text{rank}_i}$
   - *Alasan:* Sangat krusial untuk search engine dalil. Pengguna menginginkan ayat yang paling primer (seperti QS 2:183 untuk puasa) langsung muncul di posisi Rank 1. MRR mengukur efisiensi sistem menempatkan dalil sahih di urutan pertama."""))

    # 8. Section 6: Analisis 6 Pertanyaan
    cells.append(make_cell("markdown", ANALYSIS_MARKDOWN))

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    nb_path = "pencarian_dalil_quran.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    print(f"Jupyter Notebook berhasil dibuat: {nb_path}")

if __name__ == '__main__':
    main()
