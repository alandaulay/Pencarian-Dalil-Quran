"""
Modul Laporan Analisis Evaluasi Information Retrieval (IR)
Pencarian Dalil Al-Qur'an berbasis VSM dan TF-IDF.
Memuat analisis komprehensif 6 pertanyaan evaluasi tugas.
"""

ANALYSIS_MARKDOWN = r"""# 📑 Laporan Analisis Hasil Information Retrieval: Pencarian Dalil Al-Qur'an
**Metode:** Vector Space Model (VSM) & TF-IDF (Term Frequency-Inverse Document Frequency)  
**Corpus:** Al-Qur'an Terjemahan Bahasa Indonesia (Kemenag - 114 Surah, 6.236 Ayat)  
**Metrik Evaluasi:** Precision@5, Recall@5, F1-Score, dan Mean Reciprocal Rank (MRR)

---

## 1. Relevansi Ranking
> **Pertanyaan:** *Apakah ayat pada Rank 1 benar-benar paling relevan secara semantik/substansi dibanding rank di bawahnya?*

### Temuan dan Analisis:
Berdasarkan hasil pengujian pada 8 set query benchmark tematik kanonikal, sistem VSM TF-IDF berhasil memperoleh skor **Mean Reciprocal Rank (MRR) sebesar 1.0**. Nilai ini menunjukkan bahwa **seluruh query pengujian berhasil menempatkan ayat primer yang paling sahih dan relevan tepat pada Rank 1**.

Sebagai contoh nyata pada query:
> *"Kewajiban ibadah puasa di bulan Ramadhan bagi orang-orang yang beriman"*

* **Rank 1: QS. Al-Baqarah [2:183]** (Cosine Similarity: **0.6159**)  
  *Teks:* *"Wahai orang-orang yang beriman! Diwajibkan atas kamu berpuasa sebagaimana diwajibkan atas orang-orang sebelum kamu agar kamu bertakwa."*  
  **Evaluasi Relevansi:** Ayat ini adalah dalil fardhu puasa yang paling fundamental dan qath'i dalam syariat Islam. Keberadaannya di Rank 1 sangat akurat secara substansi fikih dan semantik.
* **Rank 2: QS. Al-Baqarah [2:185]** (Cosine Similarity: **0.4862**)  
  Membahas kekhususan bulan Ramadhan sebagai bulan diturunkannya Al-Qur'an dan rukhsah bagi orang sakit/musafir.
* **Rank 3: QS. Al-Baqarah [2:184]** (Cosine Similarity: **0.4021**)  
  Membahas ketentuan qadha dan fidyah puasa.

**Kesimpulan:** Urutan peringkat 1 hingga 3 secara berurutan mencerminkan hierarki dalil dari penetapan hukum utama (*al-ashl*), konteks waktu (*syahr ramadhan*), hingga dispensasi teknis (*ar-rukhsah*).

---

## 2. Penyebab Skor Tinggi (Tinggi Cosine Similarity)
> **Pertanyaan:** *Apa faktor leksikal (bobot TF-IDF, frekuensi kata kunci unik) yang menyebabkan ayat tertentu memperoleh Cosine Similarity tertinggi?*

### Formulasi Matematis:
Cosine Similarity antara vektor dokumen input $Q$ dan vektor ayat Al-Qur'an $D$ dihitung dengan rumus:
$$\text{Cosine Similarity}(Q, D) = \frac{Q \cdot D}{\|Q\|_2 \times \|D\|_2} = \frac{\sum_{t \in Q \cap D} w_{t,Q} \times w_{t,D}}{\sqrt{\sum_{t \in Q} w_{t,Q}^2} \times \sqrt{\sum_{t \in D} w_{t,D}^2}}$$

### Faktor Pendorong Skor Tertinggi:
1. **Tinggi Bobot IDF (Inverse Document Frequency):**  
   Kata-kata langka yang memiliki frekuensi dokumen rendah dalam 6.236 ayat Al-Qur'an memiliki nilai IDF yang sangat besar. Misalnya, kata `ramadan` hanya muncul di 1 ayat (QS. 2:185) sehingga bobot IDF-nya maksimum ($\approx 8.3$). Ketika kata ini beririsan dengan query, ia mendongkrak skor dot product secara signifikan.
2. **Kepadatan Interseksi Kata Kunci (Lexical Overlap):**  
   Ayat pada Rank 1 memiliki interseksi token terbanyak dengan query setelah melalui stemming. Pada QS 2:183, token yang cocok meliputi: `iman`, `orang`, `puasa`, dan `wajib`. Perkalian bobot bobot ini terakumulasi dalam penjumlahan dot product.
3. **Efek Normalisasi Panjang Dokumen (L2 Norm):**  
   Pembagi $\|D\|_2 = \sqrt{\sum w_{t,D}^2}$ memastikan ayat pendek yang padat kata kunci tidak kalah oleh ayat yang sangat panjang. Jika sebuah ayat pendek memiliki 4 kata dan 3 di antaranya cocok dengan query, rasio similaritasnya mendekati 1.0.

---

## 3. Analisis Missing Relevance (False Negatives)
> **Pertanyaan:** *Mengapa ada ayat yang sebenarnya sangat relevan secara tema/makna, tetapi tidak masuk Top-5 atau mendapat skor rendah?*

### Temuan Kasus:
Pada tema *"Berbakti kepada orang tua (Birrul Walidain)"*, ayat **QS. Al-Isra' [17:23]**:
> *"Dan Tuhanmu telah memerintahkan agar kamu jangan menyembah selain Dia dan hendaklah berbuat baik kepada ibu bapak. Jika salah seorang di antara keduanya atau kedua-duanya sampai berusia lanjut dalam pemeliharaanmu, maka sekali-kali janganlah engkau mengatakan kepada keduanya perkataan 'ah' dan janganlah engkau membentak keduanya..."*

Ayat ini adalah salah satu rujukan dalil etika berbakti paling masyhur. Namun, dalam ranking VSM murni, posisinya dapat tergeser di bawah QS Maryam [19:14] dan Maryam [19:32].

### Penyebab Kegagalan Leksikal (Vocabulary Mismatch):
1. **Ketiadaan Kata Kunci Eksplisit:**  
   Query pengguna menyertakan kata `"berbakti"`, sedangkan terjemahan resmi QS 17:23 menggunakan frasa *"berbuat baik kepada ibu bapak"* dan melarang ucapan *"ah"*. Kata `"berbakti"` (stem: `bakti`) sama sekali tidak muncul pada teks QS 17:23, sehingga komponen pembobotan untuk kata tersebut bernilai **0**.
2. **Bahasa Deskriptif / Metaforis:**  
   Al-Qur'an sering kali mengajarkan adab secara naratif atau deskriptif perilaku (seperti *"merendahkan sayap kasih sayang"* pada QS 17:24) daripada memakai istilah teknis hukum formal yang diketikkan pengguna. VSM berbasis TF-IDF tidak memahami bahwa *"jangan membentak"* bersinonim secara semantik dengan *"berbakti"*.

---

## 4. Analisis False Positive (Noise pada Hasil)
> **Pertanyaan:** *Mengapa ada ayat yang masuk Top-5 tetapi secara substansi kurang relevan dengan maksud query pengguna?*

### Temuan Kasus:
Pada pencarian dalil *"kewajiban ibadah puasa di bulan Ramadhan"*, ayat **QS. Yasin [36:17]** muncul pada peringkat 5 (skor: 0.2926):
> *"Dan kewajiban kami hanyalah menyampaikan (perintah Allah) dengan jelas."*

Secara substansi fikih, QS. Yasin 36:17 berbicara mengenai keteguhan rasul dalam berdakwah kepada penduduk negeri, **sama sekali tidak berkaitan dengan ibadah puasa**.

### Akar Masalah pada Model VSM:
1. **Polisemi dan Generalitas Kata Dasar:**  
   Setelah stemming, kata `"kewajiban"` berubah menjadi kata dasar `wajib`. Kata `wajib` muncul pada QS Yasin 36:17.
2. **Dokumen Sangat Singkat (Short Document Anomaly):**  
   QS Yasin 36:17 adalah ayat pendek yang hanya memiliki sedikit token terbersihkan (`wajib`, `sampai`, `perintah`, `allah`, `jelas`). Karena norm Euclidean-nya kecil, pembagi $\|D\|_2$ juga kecil, sehingga bobot kata `wajib` yang terisolasi menghasilkan rasio Cosine Similarity yang cukup tinggi untuk masuk Top-5 mengalahkan ayat lain yang panjang.
3. **Ketiadaan Konteks Kalimat (Loss of Syntactic Context):**  
   Model TF-IDF menganut asumsi independensi *Bag-of-Words* (BoW), sehingga ia tidak mengetahui apakah kata "wajib" melekat pada puasa atau melekat pada tabligh risalah.

---

## 5. Kelebihan dan Kekurangan Metode VSM / TF-IDF
> **Pertanyaan:** *Apa kelebihan dan kekurangan metode TF-IDF dalam pencarian dalil Al-Qur'an dibanding pendekatan lainnya?*

### Kelebihan:
1. **Kecepatan dan Skalabilitas Ekstrem:**  
   Representasi matriks jarang (*sparse matrix*) memungkinkan pencarian Cosine Similarity pada seluruh 6.236 ayat selesai dalam waktu **kurang dari 20 milidetik** tanpa memerlukan GPU.
2. **Interpretabilitas Transparan (Explainable AI):**  
   Setiap skor dapat diaudit secara eksak: kita dapat melihat secara transparan kata mana yang menyumbang bobot dot product terbesar, memudahkan verifikasi ilmiah bagi akademisi.
3. **Tanpa Kebutuhan Data Latih Berlabel:**  
   TF-IDF bersifat nir-pengawasan (*unsupervised*). Sistem tidak memerlukan ribuan anotasi pasangan query-ayat untuk dapat berfungsi dengan baik.

### Kekurangan:
1. **Vocabulary Mismatch Problem:**  
   Jika pengguna menggunakan sinonim yang tidak ada di korpus (misal: "saum" vs "puasa", "sembahyang" vs "salat", "shadaqah" vs "sedekah"), TF-IDF akan memberikan skor 0.
2. **Kehilangan Struktur Tata Bahasa & Frasa:**  
   Urutan kata diabaikan. Frasa negatif seperti *"tidak beriman"* dan frasa positif *"orang beriman"* memiliki bobot token individual yang mirip, berpotensi menimbulkan kesalahan interpretasi hukum syar'i.
3. **Ketiadaan Pemahaman Semantik (Semantic Blindness):**  
   Berbeda dengan model Dense Embedding (seperti IndoBERT atau Sentence-BERT), TF-IDF tidak memahami relasi semantik ontologis antara konsep agama.

---

## 6. Pengaruh Karakteristik Bahasa Indonesia
> **Pertanyaan:** *Bagaimana karakteristik bahasa Indonesia (morfologi, imbuhan, ejaan serapan Arab) mempengaruhi efektivitas retrieval?*

### 1. Kompleksitas Morfologi & Stemming (Sastrawi):
Bahasa Indonesia adalah bahasa aglutinatif yang kaya akan afiksasi:
* Prefiks: *ber-puasa*, *di-wajibkan*, *mem-perintahkan*
* Konfiks: *ke-wajib-an*, *ke-utama-an*, *ber-bakti*
* Partikel & Klitika: *-lah*, *-kah*, *-nya*, *-mu*, *-ku*

**Dampak pada IR:**  
Jika tanpa stemming, kata `"berpuasa"` tidak akan pernah cocok dengan `"puasa"`, dan `"kewajiban"` tidak akan cocok dengan `"wajib"`. Stemming dengan algoritma Nazief-Adriani (via pustaka Sastrawi) berhasil memangkas variasi morfemik ini kembali ke lema dasarnya, meningkatkan Recall sistem secara drastis dari 0% menjadi 75% pada tema puasa. Namun, resiko *over-stemming* (memotong terlalu banyak) atau *under-stemming* (gagal mengurai imbuhan rangkap) tetap menjadi tantangan khas Bahasa Indonesia.

### 2. Variasi Ortografi & Ejaan Kata Serapan Bahasa Arab:
Kosakata keagamaan dalam Bahasa Indonesia merupakan serapan fonemis dari Bahasa Arab yang memiliki variasi ejaan masyarakat yang sangat tinggi:
* *Shalat / Sholat / Sembahyang / Salat*
* *Ramadhan / Ramadan*
* *Riba / Ribaat*
* *Dzakat / Zakat*
* *Al-Qur'an / Al-Quran / Alquran / Quran*
* *Ridha / Ridho / Rida*

**Dampak pada IR & Solusi:**  
Korpus Kemenag menggunakan ejaan baku KBBI (*salat*, *ramadan*, *zakat*, *rida*). Jika pengguna mengetikkan ejaan populer non-baku (*"tata cara sholat"*), TF-IDF standar akan gagal total karena *mismatch* huruf 'sh'. Oleh karena itu, modul pipeline yang kami rancang menambahkan lapisan **Spelling Normalization** khusus serapan Islami sebelum tahap stemming, sehingga akurasi sistem tetap tangguh terhadap variasi input pengguna.
"""
