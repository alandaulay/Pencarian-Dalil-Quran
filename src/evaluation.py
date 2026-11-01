import os
import json
import pandas as pd
import numpy as np

# Benchmark query tematik beserta himpunan ground-truth ayat relevan (canonical verses)
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "theme": "Kewajiban Puasa Ramadhan",
        "query": "Kewajiban ibadah puasa di bulan Ramadhan bagi orang-orang yang beriman",
        "relevant_verses": [
            "Al-Baqarah [2:183]",
            "Al-Baqarah [2:184]",
            "Al-Baqarah [2:185]",
            "Al-Baqarah [2:187]"
        ],
        "description": "Ayat fardhu puasa Ramadhan, rukhsah sakit/musafir, dan turunnya Al-Qur'an."
    },
    {
        "id": 2,
        "theme": "Larangan Riba",
        "query": "Larangan memakan dan mengambil harta riba serta ancaman bagi pelaku riba",
        "relevant_verses": [
            "Al-Baqarah [2:275]",
            "Al-Baqarah [2:276]",
            "Al-Baqarah [2:278]",
            "Ali 'Imran [3:130]",
            "An-Nisa' [4:161]"
        ],
        "description": "Haramnya riba berlipat ganda, pemusnahan berkah riba, dan laknat pemakan riba."
    },
    {
        "id": 3,
        "theme": "Berbakti Kepada Orang Tua (Birrul Walidain)",
        "query": "Perintah berbuat baik dan berbakti kepada kedua orang tua terutama ibu bapak",
        "relevant_verses": [
            "Maryam [19:14]",
            "Maryam [19:32]",
            "Luqman [31:14]",
            "Al-Isra' [17:23]",
            "Al-Isra' [17:24]",
            "Al-Ahqaf [46:15]",
            "Al-'Ankabut [29:8]"
        ],
        "description": "Perintah berbakti kepada orang tua, larangan membentak, dan keteladanan Nabi Yahya serta Nabi Isa."
    },
    {
        "id": 4,
        "theme": "Perintah Shalat dan Zakat",
        "query": "Kewajiban mendirikan shalat dan menunaikan zakat bagi kaum muslimin",
        "relevant_verses": [
            "Al-Baqarah [2:43]",
            "Al-Baqarah [2:110]",
            "Luqman [31:4]",
            "An-Naml [27:3]",
            "Al-Mu'minun [23:4]",
            "An-Nur [24:56]",
            "Al-Bayyinah [98:5]"
        ],
        "description": "Perintah kembar mendirikan/melaksanakan salat dan menunaikan zakat dalam berbagai surah."
    },
    {
        "id": 5,
        "theme": "Sabar Menghadapi Ujian dan Cobaan",
        "query": "Perintah bersabar dan shalat dalam menghadapi ujian musibah kelaparan ketakutan",
        "relevant_verses": [
            "Al-Baqarah [2:153]",
            "Al-Baqarah [2:155]",
            "Al-Baqarah [2:156]",
            "Ali 'Imran [3:200]"
        ],
        "description": "Jadikan sabar dan shalat penolong, ujian ketakutan dan kelaparan, serta kalimat tarji'."
    },
    {
        "id": 6,
        "theme": "Toleransi Beragama",
        "query": "Toleransi beragama lakum dinukum waliyadin tidak ada paksaan dalam agama",
        "relevant_verses": [
            "Al-Kafirun [109:6]",
            "Al-Baqarah [2:256]",
            "Yunus [10:99]",
            "Al-Mumtahanah [60:8]"
        ],
        "description": "Prinsip tiada paksaan dalam memeluk Islam dan toleransi beragama."
    },
    {
        "id": 7,
        "theme": "Penciptaan Manusia",
        "query": "Proses penciptaan manusia dari saripati tanah segumpal darah air mani",
        "relevant_verses": [
            "Al-'Alaq [96:2]",
            "Al-Mu'minun [23:12]",
            "Al-Mu'minun [23:13]",
            "Al-Mu'minun [23:14]",
            "At-Tin [95:4]"
        ],
        "description": "Fase embriologis penciptaan manusia dalam Al-Qur'an."
    },
    {
        "id": 8,
        "theme": "Larangan Ghibah dan Menggunjing",
        "query": "Larangan menggunjing ghibah mencari kesalahan orang lain seperti memakan daging bangkai saudara",
        "relevant_verses": [
            "Al-Hujurat [49:12]",
            "Al-Humazah [104:1]",
            "Al-Hujurat [49:11]"
        ],
        "description": "Perumpamaan menggunjing seperti memakan bangkai saudara sendiri dan celaan mengolok-olok."
    }
]

class ModelEvaluator:
    def __init__(self, search_engine):
        self.search_engine = search_engine

    def evaluate_benchmark(self, top_k: int = 5):
        """
        Menjalankan evaluasi komprehensif pada kumpulan benchmark query.
        Menghitung Precision@k, Recall@k, F1-Score, dan Reciprocal Rank (RR).
        """
        results_list = []
        reciprocal_ranks = []
        precisions = []
        recalls = []
        f1_scores = []

        for b in BENCHMARK_QUERIES:
            q_id = b["id"]
            theme = b["theme"]
            query_text = b["query"]
            ground_truth = set(b["relevant_verses"])

            # Jalankan pencarian model VSM TF-IDF
            df_search = self.search_engine.search(query_text, top_k=top_k)
            retrieved_verses = df_search['Surah & Ayat'].tolist() if not df_search.empty else []

            # 1. Hitung interseksi ayat relevan yang ditemukan
            hits = [v for v in retrieved_verses if v in ground_truth]
            hit_count = len(hits)

            # 2. Precision@k = |Retrieved ∩ Relevant| / k
            precision_at_k = hit_count / top_k

            # 3. Recall@k = |Retrieved ∩ Relevant| / |Relevant|
            recall_at_k = hit_count / len(ground_truth) if len(ground_truth) > 0 else 0.0

            # 4. F1-Score = 2 * (P * R) / (P + R)
            if (precision_at_k + recall_at_k) > 0:
                f1 = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
            else:
                f1 = 0.0

            # 5. Reciprocal Rank (RR) = 1 / first_rank_of_relevant_item
            rr = 0.0
            first_hit_rank = None
            for rank_idx, v in enumerate(retrieved_verses, start=1):
                if v in ground_truth:
                    rr = 1.0 / rank_idx
                    first_hit_rank = rank_idx
                    break

            precisions.append(precision_at_k)
            recalls.append(recall_at_k)
            f1_scores.append(f1)
            reciprocal_ranks.append(rr)

            results_list.append({
                "ID": q_id,
                "Tema": theme,
                "Query": query_text,
                "Ground Truth (Ayat)": ", ".join(b["relevant_verses"]),
                "Top-5 Ditemukan": ", ".join(retrieved_verses[:5]),
                "Relevant Hits": ", ".join(hits) if hits else "Tidak ada",
                "Hit Count": f"{hit_count} / {len(ground_truth)}",
                "First Hit Rank": first_hit_rank if first_hit_rank is not None else "> 5",
                "Precision@5": round(precision_at_k, 4),
                "Recall@5": round(recall_at_k, 4),
                "F1-Score": round(f1, 4),
                "Reciprocal Rank": round(rr, 4)
            })

        df_results = pd.DataFrame(results_list)

        summary_metrics = {
            "mean_precision_at_5": round(float(np.mean(precisions)), 4),
            "mean_recall_at_5": round(float(np.mean(recalls)), 4),
            "mean_f1_score": round(float(np.mean(f1_scores)), 4),
            "mrr": round(float(np.mean(reciprocal_ranks)), 4),
            "total_queries": len(BENCHMARK_QUERIES)
        }

        return df_results, summary_metrics

    @staticmethod
    def get_metric_explanations():
        """Mengembalikan teks akademik penjelasan dan alasan pemilihan metrik evaluasi."""
        return {
            "Precision@5": {
                "definisi": "Proporsi dokumen (ayat) yang relevan di antara 5 ayat teratas yang dikembalikan oleh sistem.",
                "rumus": r"$$\text{Precision@5} = \frac{|\text{Top 5 Retrieved} \cap \text{Relevant}|}{5}$$",
                "alasan": "Dalam sistem temu kembali ayat suci, pengguna manusia (user) umumnya hanya fokus membaca 5 rekomendasi teratas di layar tanpa menggulir terlalu jauh. Precision@5 mengukur seberapa bersih atau bebas dari 'noise' (false positive) ayat yang disajikan pada antarmuka utama."
            },
            "Recall@5": {
                "definisi": "Proporsi ayat relevan yang berhasil ditemukan dalam Top-5 dibandingkan seluruh ayat relevan yang ada di dalam Al-Qur'an untuk tema tersebut.",
                "rumus": r"$$\text{Recall@5} = \frac{|\text{Top 5 Retrieved} \cap \text{Relevant}|}{|\text{Total Relevant Ground Truth}|}$$",
                "alasan": "Recall@5 mengukur kelengkapan (comprehensiveness). Satu tema dalam Al-Qur'an (misal larangan riba atau hukum puasa) sering kali tersebar di beberapa ayat. Recall@5 menguji apakah Top-5 mampu merangkum sebanyak mungkin dalil penting yang ada."
            },
            "F1-Score": {
                "definisi": "Rata-rata harmonik (harmonic mean) antara Precision@5 dan Recall@5.",
                "rumus": r"$$\text{F1-Score} = 2 \times \frac{\text{Precision@5} \times \text{Recall@5}}{\text{Precision@5} + \text{Recall@5}}$$",
                "alasan": "Menyediakan metrik tunggal yang seimbang antara ketepatan (precision) dan kelengkapan (recall). Karena Precision dan Recall saling bertentangan (trade-off), F1-Score memastikan sistem tidak hanya mengejar presisi tinggi dengan mengorbankan kelengkapan ayat, atau sebaliknya."
            },
            "MRR (Mean Reciprocal Rank)": {
                "definisi": "Rata-rata dari nilai kebalikan (reciprocal) posisi rank ayat relevan pertama yang berhasil ditemukan di seluruh kumpulan query.",
                "rumus": r"$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$",
                "alasan": "Sangat krusial untuk evaluasi search engine! Ketika pengguna mencari dalil untuk suatu topik (misal 'puasa'), pengguna mengharapkan ayat paling primer (seperti QS 2:183) langsung muncul di Rank 1 (skor 1.0). Jika ayat pertama relevan baru muncul di Rank 2, skornya turun menjadi 0.5; jika di Rank 3 menjadi 0.33. MRR mengukur efisiensi sistem dalam menyajikan dalil valid sesegera mungkin di puncak ranking."
            }
        }
