import os
import json
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords as nltk_stopwords

class QuranPreprocessor:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        self.data_dir = data_dir
        self.cache_path = os.path.join(data_dir, 'stem_cache.json')
        
        # Inisialisasi Stemmer Sastrawi
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        # Inisialisasi Stopwords gabungan Sastrawi + NLTK
        sastrawi_stops = set(StopWordRemoverFactory().get_stop_words())
        try:
            nltk_stops = set(nltk_stopwords.words('indonesian'))
        except Exception:
            import nltk
            nltk.download('stopwords')
            nltk_stops = set(nltk_stopwords.words('indonesian'))
            
        self.stopwords = sastrawi_stops.union(nltk_stops)
        # Partikel tambahan
        self.stopwords.update({'nya', 'lah', 'kah', 'tah', 'pun', 'terhadap', 'kepada', 'dan', 'yang', 'di', 'ke', 'dari', 'itu', 'ini'})
        
        # Kamus normalisasi variasi kata serapan Islami dalam Bahasa Indonesia
        self.loanword_norm = {
            r'\bshalat\b': 'salat',
            r'\bsholat\b': 'salat',
            r'\bsembahyang\b': 'salat',
            r'\bramadhan\b': 'ramadan',
            r'\bdzakat\b': 'zakat',
            r'\bal[\-\s]?qur[\'a]an\b': 'quran',
            r'\bridho\b': 'rida',
            r'\bridha\b': 'rida',
            r'\bdzikir\b': 'zikir',
            r'\bdikr\b': 'zikir',
            r'\bdzalim\b': 'zalim',
            r'\bkaafir\b': 'kafir',
            r'\btaqwa\b': 'takwa',
            r'\bmu[\'a]?min\b': 'mukmin',
            r'\bshadaqah\b': 'sedekah',
            r'\bshodaqoh\b': 'sedekah',
            r'\brijal\b': 'laki',
            r'\bghaib\b': 'gaib',
            r'\bakhirat\b': 'akhirat'
        }
        
        # Load cache stemming jika ada
        self.stem_cache = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.stem_cache = json.load(f)
            except Exception:
                self.stem_cache = {}

    def normalize_loanwords(self, text: str) -> str:
        """Menyelaraskan variasi ejaan serapan Arab-Indonesia (misal shalat -> salat)."""
        res = text
        for pattern, repl in self.loanword_norm.items():
            res = re.sub(pattern, repl, res)
        return res

    def clean_text(self, text: str) -> str:
        """Case folding, normalisasi serapan, dan pembersihan tanda baca/angka."""
        text_lower = text.lower()
        # Normalisasi variasi fonologis / ejaan serapan
        text_norm = self.normalize_loanwords(text_lower)
        # Ganti karakter tanda baca dan angka dengan spasi
        text_clean = re.sub(r'[^a-zA-Z\s]', ' ', text_norm)
        # Hapus spasi berlebih
        return re.sub(r'\s+', ' ', text_clean).strip()

    def tokenize(self, text: str) -> list:
        """Memecah teks menjadi token/kata."""
        return [w for w in text.split() if len(w) > 1]

    def remove_stopwords(self, tokens: list) -> list:
        """Menghapus stopword bahasa Indonesia."""
        return [w for w in tokens if w not in self.stopwords]

    def stem_token(self, word: str) -> str:
        """Stemming kata tunggal dengan memoization cache."""
        if word in self.stem_cache:
            return self.stem_cache[word]
        stemmed = self.stemmer.stem(word)
        self.stem_cache[word] = stemmed
        return stemmed

    def stem_tokens(self, tokens: list) -> list:
        """Melakukan stemming pada daftar token."""
        return [self.stem_token(w) for w in tokens]

    def preprocess(self, text: str) -> str:
        """Pipeline lengkap: Case folding -> Cleaning & Normalisasi -> Tokenizing -> Stopwords -> Stemming."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        stemmed = self.stem_tokens(filtered)
        # Saring kembali token pendek setelah di-stem
        final_tokens = [w for w in stemmed if len(w) > 1 and w not in self.stopwords]
        return " ".join(final_tokens)

    def preprocess_steps(self, text: str) -> dict:
        """Mengembalikan langkah-langkah preprocessing untuk inspeksi edukasi/analisis."""
        lower_text = text.lower()
        normalized = self.normalize_loanwords(lower_text)
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', normalized)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        stemmed = self.stem_tokens(filtered)
        final_tokens = [w for w in stemmed if len(w) > 1 and w not in self.stopwords]
        return {
            'original': text,
            'lowercased': lower_text,
            'normalized_loanwords': normalized,
            'cleaned': cleaned,
            'tokenized': tokens,
            'filtered_stopwords': filtered,
            'stemmed': stemmed,
            'final_text': " ".join(final_tokens)
        }
