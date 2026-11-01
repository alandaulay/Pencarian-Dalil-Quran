import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessor import QuranPreprocessor

class QuranSearchEngine:
    def __init__(self, data_path=None, preprocessor=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if data_path is None:
            data_path = os.path.join(base_dir, 'data', 'quran_dataset.json')
        self.data_path = data_path
        
        self.preprocessor = preprocessor or QuranPreprocessor()
        self.verses_data = []
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        
        self._load_and_index()

    def _load_and_index(self):
        """Memuat dataset ayat Al-Qur'an dan membangun matriks TF-IDF."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset belum siap di: {self.data_path}. Jalankan run_preprocessing.py terlebih dahulu.")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.verses_data = json.load(f)

        self.df = pd.DataFrame(self.verses_data)
        
        # Inisialisasi TfidfVectorizer
        # Menggunakan sublinear_tf=True untuk meredam dominasi frekuensi kata ekstrem (1 + log(tf))
        self.vectorizer = TfidfVectorizer(
            lowercase=False,     # Teks sudah di-clean dan di-stem sebelumnya
            norm='l2',          # Normalisasi Euclidean L2 untuk Cosine Similarity
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            min_df=1
        )
        
        corpus = self.df['clean_text'].fillna('').tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Mencari top_k ayat paling relevan berdasarkan Cosine Similarity VSM-TFIDF.
        
        Parameters:
            query (str): Teks dokumen input (esai, artikel, pertanyaan, kata kunci).
            top_k (int): Jumlah ayat teratas yang dikembalikan (default: 5).
            
        Returns:
            pd.DataFrame: Tabel berisikan Rank, Surah & Ayat, Score, Teks Ayat & Terjemahan.
        """
        if not query or not query.strip():
            return pd.DataFrame()

        # 1. Preprocessing teks query input
        processed_query = self.preprocessor.preprocess(query)
        if not processed_query.strip():
            return pd.DataFrame()

        # 2. Vektorisasi query ke dalam ruang VSM (TF-IDF)
        query_vector = self.vectorizer.transform([processed_query])

        # 3. Hitung Cosine Similarity antara Query dan seluruh dokumen Ayat
        # Cosine Similarity = (Q . D) / (||Q|| * ||D||)
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # 4. Ambil indeks teratas dengan skor tertinggi
        # Filter ayat yang memiliki skor > 0
        non_zero_indices = np.where(similarities > 0)[0]
        if len(non_zero_indices) == 0:
            return pd.DataFrame()

        # Urutkan berdasarkan similarity terbesar
        sorted_indices = non_zero_indices[np.argsort(similarities[non_zero_indices])[::-1]]
        top_indices = sorted_indices[:top_k]

        results = []
        query_tokens = set(processed_query.split())

        for rank, idx in enumerate(top_indices, start=1):
            row = self.df.iloc[idx]
            score = float(similarities[idx])
            
            # Cari kata kunci yang cocok antara query dan ayat
            verse_tokens = set(str(row['clean_text']).split())
            matched_words = sorted(list(query_tokens.intersection(verse_tokens)))
            
            # Format kolom wajib: Rank, Surah & Ayat, Score, Teks Ayat & Terjemahan
            teks_dan_terjemahan = f"{row['arabic_text']}\n\nTerjemahan:\n\"{row['translation_id']}\""

            results.append({
                'Rank': rank,
                'Surah & Ayat': row['surah_ayah'],
                'Score': round(score, 4),
                'Teks Ayat & Terjemahan': teks_dan_terjemahan,
                # Kolom pelengkap untuk visualisasi di UI
                'Surah': row['surah_name'],
                'Nomor Surah': row['surah_number'],
                'Nomor Ayat': row['ayah_number'],
                'Teks Arab': row['arabic_text'],
                'Terjemahan': row['translation_id'],
                'Matched Keywords': ", ".join(matched_words) if matched_words else "-",
                'Raw Score': score
            })

        return pd.DataFrame(results)

    def explain_query_weights(self, query: str):
        """Menganalisis bobot TF-IDF dari kata-kata pada query pengguna."""
        processed_query = self.preprocessor.preprocess(query)
        if not processed_query:
            return []
        
        query_vec = self.vectorizer.transform([processed_query])
        non_zeros = query_vec.nonzero()[1]
        
        term_weights = []
        for idx in non_zeros:
            term = self.feature_names[idx]
            weight = query_vec[0, idx]
            idf = self.vectorizer.idf_[idx]
            term_weights.append({
                'term': term,
                'tf_idf_weight': round(float(weight), 4),
                'idf': round(float(idf), 4)
            })
            
        term_weights.sort(key=lambda x: x['tf_idf_weight'], reverse=True)
        return term_weights
