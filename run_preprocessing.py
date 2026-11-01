import json
import time
import os
from src.preprocessor import QuranPreprocessor

def main():
    print("=" * 60)
    print("MEMULAI PROSES PREPROCESSING CORPUS AL-QUR'AN LENGKAP")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, 'data', 'quran_raw.json')
    output_path = os.path.join(base_dir, 'data', 'quran_dataset.json')

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File {raw_path} tidak ditemukan.")

    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 1. Ekstraksi data mentah
    print("1. Mengekstraksi 6.236 ayat...")
    surahs = raw_data['data']
    verses_list = []
    
    for surah in surahs:
        s_num = surah['number']
        s_name = surah['name']['transliteration']['id']
        s_name_ar = surah['name']['short']
        for v in surah['verses']:
            a_num = v['number']['inSurah']
            in_quran = v['number']['inQuran']
            ar_text = v['text']['arab']
            trans_id = v['translation']['id']
            verses_list.append({
                'id': in_quran,
                'surah_number': s_num,
                'surah_name': s_name,
                'surah_name_ar': s_name_ar,
                'ayah_number': a_num,
                'surah_ayah': f"{s_name} [{s_num}:{a_num}]",
                'arabic_text': ar_text,
                'translation_id': trans_id
            })

    print(f"   Total ayat terekstraksi: {len(verses_list)}")

    # 2. Inisialisasi Preprocessor
    print("2. Menginisialisasi Preprocessor (Case folding, Cleaning, Loanwords, Tokenizing, Stopwords, Stemming)...")
    preprocessor = QuranPreprocessor(data_dir=os.path.join(base_dir, 'data'))

    # 3. Menjalankan Preprocessing pada seluruh ayat
    print("3. Menjalankan Preprocessing pada seluruh 6.236 ayat...")
    t0 = time.time()
    for i, item in enumerate(verses_list):
        item['clean_text'] = preprocessor.preprocess(item['translation_id'])
        if (i + 1) % 1500 == 0 or (i + 1) == len(verses_list):
            print(f"   Diproses: {i + 1}/{len(verses_list)} ayat...")

    print(f"   Preprocessing selesai dalam {time.time() - t0:.2f} detik.")

    # 4. Simpan kembali stem cache yang terupdate
    cache_path = os.path.join(base_dir, 'data', 'stem_cache.json')
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessor.stem_cache, f, ensure_ascii=False, indent=2)

    # 5. Simpan dataset lengkap
    print(f"4. Menyimpan dataset bersih ke {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(verses_list, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("SUKSES: Preprocessing selesai sempurna!")
    print(f"Contoh Ayat 1 ({verses_list[0]['surah_ayah']}):")
    print(f" - Terjemahan Asli : {verses_list[0]['translation_id']}")
    print(f" - Clean & Stemmed : {verses_list[0]['clean_text']}")
    print("=" * 60)

if __name__ == '__main__':
    main()
