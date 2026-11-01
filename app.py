import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json

from src.preprocessor import QuranPreprocessor
from src.vsm_tfidf import QuranSearchEngine
from src.evaluation import ModelEvaluator, BENCHMARK_QUERIES
from src.analysis_report import ANALYSIS_MARKDOWN

# Set Page Config
st.set_page_config(
    page_title="Pencarian Dalil Al-Qur'an - VSM & TF-IDF",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #40916c;
        margin-bottom: 1.5rem;
    }
    .arabic-text {
        font-family: 'Amiri', serif;
        font-size: 1.85rem;
        direction: rtl;
        text-align: right;
        line-height: 2.3;
        color: #1e293b;
        padding: 12px 16px;
        background-color: #f8fafc;
        border-radius: 8px;
        border-right: 4px solid #2d6a4f;
        margin-bottom: 12px;
    }
    .verse-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .verse-card:hover {
        box-shadow: 0 4px 12px rgba(45,106,79,0.12);
        border-color: #74c69d;
    }
    .rank-badge {
        background-color: #2d6a4f;
        color: white;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .score-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        border: 1px solid #c8e6c9;
        display: inline-block;
    }
    .keyword-tag {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-right: 4px;
        display: inline-block;
    }
    .metric-box {
        background: #f1f8f5;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #d8f3dc;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1b4332;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #52796f;
        text-transform: uppercase;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Cache search engine resource
@st.cache_resource(show_spinner="Memuat indeks korpus 6.236 ayat Al-Qur'an...")
def get_search_engine():
    preprocessor = QuranPreprocessor()
    engine = QuranSearchEngine(preprocessor=preprocessor)
    evaluator = ModelEvaluator(engine)
    return engine, preprocessor, evaluator

engine, preprocessor, evaluator = get_search_engine()

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Quran_Kareem.svg/640px-Quran_Kareem.svg.png", width=120)
    st.title("Pencarian Dalil")
    st.markdown("**Information Retrieval System**")
    st.caption("Metode: Vector Space Model (VSM) & TF-IDF Cosine Similarity")
    
    st.divider()
    st.subheader("⚙️ Parameter Pencarian")
    top_k = st.slider("Jumlah Ayat Teratas (Top-K):", min_value=1, max_value=20, value=5, step=1)
    
    st.divider()
    st.subheader("📚 Informasi Korpus")
    st.markdown(f"""
    - **Total Surah:** 114 Surah
    - **Total Ayat:** {len(engine.df):,} Ayat
    - **Total Fitur Kata (Vocab):** {len(engine.feature_names):,} Token
    - **Sumber Data:** Kemenag RI / Terjemahan Bahasa Indonesia
    - **Stemmer:** Sastrawi Nazief-Adriani
    """)
    st.divider()
    st.caption("Tugas Information Retrieval © 2026")

# Header Section
st.markdown('<div class="main-title">📖 Pencarian Dalil Al-Qur\'an</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Temu Kembali Informasi Dalil Ayat Al-Qur\'an Berdasarkan Dokumen/Artikel Input menggunakan Vector Space Model (VSM) & TF-IDF</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Pencarian Dalil (Search Engine)",
    "📊 Evaluasi Model IR",
    "📝 Analisis 6 Pertanyaan Tugas",
    "📚 Eksplorasi Korpus & VSM"
])

# =========================================================================
# TAB 1: PENCARIAN DALIL
# =========================================================================
with tab1:
    st.subheader("Input Dokumen / Artikel / Query")
    st.write("Masukkan teks artikel, esai, pertanyaan, atau kata kunci topik untuk menemukan ayat Al-Qur'an yang paling bersesuaian.")
    
    # Preset Contoh Query
    example_col1, example_col2, example_col3, example_col4 = st.columns(4)
    preset_query = ""
    with example_col1:
        if st.button("🌙 Puasa Ramadhan"):
            preset_query = "Kewajiban ibadah puasa di bulan Ramadhan bagi orang-orang yang beriman agar bertakwa"
    with example_col2:
        if st.button("💰 Larangan Riba"):
            preset_query = "Larangan memakan riba berlipat ganda dan ancaman Allah terhadap pelaku riba"
    with example_col3:
        if st.button("👨‍👩‍👧 Berbakti Orang Tua"):
            preset_query = "Perintah berbuat baik dan berbakti kepada kedua orang tua terutama ibu bapak"
    with example_col4:
        if st.button("🕊️ Toleransi Beragama"):
            preset_query = "Toleransi antar umat beragama lakum dinukum waliyadin tidak ada paksaan dalam agama"
            
    query_input = st.text_area(
        "Teks Dokumen Input:",
        value=preset_query if preset_query else "Kewajiban mendirikan shalat dan menunaikan zakat serta sabar dalam menghadapi cobaan musibah",
        height=110,
        help="Anda dapat memasukkan paragraf berita/artikel panjang atau kata kunci tematik."
    )
    
    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        search_clicked = st.button("🚀 Cari Dalil", type="primary", use_container_width=True)
        
    if search_clicked or query_input:
        if not query_input.strip():
            st.warning("Silakan masukkan teks terlebih dahulu!")
        else:
            t0 = time.time()
            results_df = engine.search(query_input, top_k=top_k)
            search_time = (time.time() - t0) * 1000
            
            if results_df.empty:
                st.info("Tidak ditemukan ayat yang memiliki kemiripan kata kunci dengan dokumen input.")
            else:
                st.success(f"Ditemukan {len(results_df)} ayat relevan dalam {search_time:.1f} ms!")
                
                # TAMPILKAN TABEL TOP-5 SESUAI KRITERIA TUGAS:
                # Kolom tabel: Rank, Surah & Ayat, Score, Teks Ayat & Terjemahan
                st.markdown("### 📋 Tabel Hasil Ranking Top-Ayat")
                
                table_display = results_df[['Rank', 'Surah & Ayat', 'Score', 'Teks Ayat & Terjemahan']].copy()
                st.dataframe(
                    table_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", width="small"),
                        "Surah & Ayat": st.column_config.TextColumn("Surah & Ayat", width="medium"),
                        "Score": st.column_config.ProgressColumn("Score (Cosine Sim)", min_value=0.0, max_value=1.0, format="%.4f"),
                        "Teks Ayat & Terjemahan": st.column_config.TextColumn("Teks Ayat & Terjemahan", width="large")
                    }
                )
                
                # DOWNLOAD CSV
                csv_data = table_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Hasil Tabel (CSV)",
                    data=csv_data,
                    file_name="hasil_pencarian_dalil_quran.csv",
                    mime="text/csv"
                )
                
                # KARTU VISUAL DETAIL
                st.markdown("### 🕌 Tampilan Kartu Detail Ayat Teratas")
                for _, r in results_df.iterrows():
                    st.markdown(f"""
                    <div class="verse-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div>
                                <span class="rank-badge">Rank #{r['Rank']}</span> &nbsp;
                                <strong style="font-size: 1.15rem; color: #1b4332;">{r['Surah & Ayat']}</strong>
                            </div>
                            <div>
                                <span class="score-badge">Cosine Similarity: {r['Score']:.4f}</span>
                            </div>
                        </div>
                        <div class="arabic-text">
                            {r['Teks Arab']}
                        </div>
                        <div style="font-size: 1rem; color: #334155; line-height: 1.6; margin-bottom: 12px;">
                            <strong>Terjemahan (Kemenag):</strong><br>
                            <em>"{r['Terjemahan']}"</em>
                        </div>
                        <div style="font-size: 0.85rem; color: #64748b;">
                            <strong>Kata Kunci Cocok (Stemmed):</strong> &nbsp;
                            {" ".join([f'<span class="keyword-tag">{w}</span>' for w in str(r["Matched Keywords"]).split(", ") if w != "-"])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # INSPEKTOR PREPROCESSING DOKUMEN INPUT
                with st.expander("🔍 Inspeksi Tahapan Preprocessing Teks Input (Transparansi IR)"):
                    steps = preprocessor.preprocess_steps(query_input)
                    st.markdown("**1. Teks Asli Input:**")
                    st.code(steps['original'], language="text")
                    
                    st.markdown("**2. Case Folding & Normalisasi Serapan Islami:**")
                    st.code(steps['normalized_loanwords'], language="text")
                    
                    st.markdown("**3. Cleaning (Hapus tanda baca & angka):**")
                    st.code(steps['cleaned'], language="text")
                    
                    st.markdown(f"**4. Tokenizing ({len(steps['tokenized'])} token):**")
                    st.write(steps['tokenized'])
                    
                    st.markdown(f"**5. Stopword Removal ({len(steps['filtered_stopwords'])} kata tersisa):**")
                    st.write(steps['filtered_stopwords'])
                    
                    st.markdown(f"**6. Stemming Sastrawi ({len(steps['stemmed'])} kata dasar):**")
                    st.write(steps['stemmed'])
                    
                    st.markdown("**7. Teks Final Terproses untuk VSM:**")
                    st.info(steps['final_text'])
                    
                    # Analisis Bobot TF-IDF Query
                    term_weights = engine.explain_query_weights(query_input)
                    if term_weights:
                        st.markdown("**8. Bobot TF-IDF Token Query Input:**")
                        df_weights = pd.DataFrame(term_weights)
                        st.dataframe(df_weights, use_container_width=True, hide_index=True)

# =========================================================================
# TAB 2: EVALUASI MODEL IR
# =========================================================================
with tab2:
    st.subheader("📊 Evaluasi Kinerja Model Information Retrieval")
    st.write("Evaluasi model dihitung secara kuantitatif berdasarkan pengujian pada himpunan query tematik kanonikal dan ground-truth ayat Al-Qur'an.")
    
    # Jalankan benchmark
    df_eval, summary = evaluator.evaluate_benchmark(top_k=5)
    
    # METRIC CARDS
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{summary['mean_precision_at_5']:.2%}</div>
            <div class="metric-lbl">Precision@5</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{summary['mean_recall_at_5']:.2%}</div>
            <div class="metric-lbl">Recall@5</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{summary['mean_f1_score']:.2%}</div>
            <div class="metric-lbl">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{summary['mrr']:.4f}</div>
            <div class="metric-lbl">MRR (Mean Reciprocal Rank)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # TABEL DETAIL BENCHMARK
    st.markdown("### 📋 Rincian Hasil Evaluasi per Benchmark Query")
    st.dataframe(
        df_eval[['ID', 'Tema', 'First Hit Rank', 'Precision@5', 'Recall@5', 'F1-Score', 'Reciprocal Rank', 'Relevant Hits', 'Top-5 Ditemukan']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Precision@5": st.column_config.NumberColumn(format="%.4f"),
            "Recall@5": st.column_config.NumberColumn(format="%.4f"),
            "F1-Score": st.column_config.NumberColumn(format="%.4f"),
            "Reciprocal Rank": st.column_config.NumberColumn(format="%.4f")
        }
    )
    
    # ALASAN PEMILIHAN METRIK (KRITERIA 6 TUGAS)
    st.divider()
    st.markdown("### 🎓 Penjelasan & Alasan Pemilihan Metrik Evaluasi")
    
    metric_info = evaluator.get_metric_explanations()
    
    for metric_name, info in metric_info.items():
        with st.container():
            st.markdown(f"#### 📌 {metric_name}")
            col_def, col_formula = st.columns([3, 2])
            with col_def:
                st.markdown(f"**Definisi:** {info['definisi']}")
                st.markdown(f"**Alasan Pemilihan untuk Pencarian Dalil:** {info['alasan']}")
            with col_formula:
                st.markdown("**Rumus Matematis:**")
                st.latex(info['rumus'].replace("$$", ""))
            st.markdown("---")

# =========================================================================
# TAB 3: ANALISIS 6 PERTANYAAN EVALUASI TUGAS
# =========================================================================
with tab3:
    st.markdown(ANALYSIS_MARKDOWN)

# =========================================================================
# TAB 4: EKSPLORASI KORPUS & VSM
# =========================================================================
with tab4:
    st.subheader("📚 Eksplorasi Korpus Terjemahan Al-Qur'an & Ruang Vektor")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Ayat Korpus", f"{len(engine.df):,} Ayat")
    with col_stat2:
        st.metric("Total Surah", f"{len(engine.df['surah_number'].unique()):,} Surah")
    with col_stat3:
        st.metric("Dimensi Vektor (Vocab)", f"{len(engine.feature_names):,} Dimensi")
        
    st.divider()
    st.subheader("🔍 Cari & Telusuri Ayat Spesifik")
    surah_list = engine.df['surah_name'].unique().tolist()
    selected_surah = st.selectbox("Pilih Surah:", surah_list)
    
    surah_df = engine.df[engine.df['surah_name'] == selected_surah]
    st.dataframe(
        surah_df[['ayah_number', 'arabic_text', 'translation_id', 'clean_text']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ayah_number": st.column_config.NumberColumn("Ayat Ke-", width="small"),
            "arabic_text": st.column_config.TextColumn("Teks Arab", width="medium"),
            "translation_id": st.column_config.TextColumn("Terjemahan Indonesia", width="large"),
            "clean_text": st.column_config.TextColumn("Tokens Bersih (Stemmed)", width="medium")
        }
    )
