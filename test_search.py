import os
from src.vsm_tfidf import QuranSearchEngine
from src.evaluation import ModelEvaluator

def main():
    print("=" * 60)
    print("TESTING VSM & TF-IDF SEARCH ENGINE")
    print("=" * 60)
    
    engine = QuranSearchEngine()
    print(f"Corpus loaded: {len(engine.df)} verses.")
    
    query = "kewajiban ibadah puasa di bulan ramadhan bagi orang beriman"
    print(f"\nQuery: '{query}'")
    results = engine.search(query, top_k=5)
    
    print("\nTop-5 Search Results:")
    for _, r in results.iterrows():
        print(f"Rank {r['Rank']}: {r['Surah & Ayat']} (Score: {r['Score']:.4f})")
        print(f"   Keywords: {r['Matched Keywords']}")
        print(f"   Translation: {r['Terjemahan'][:80]}...")
        
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK EVALUATION")
    print("=" * 60)
    
    evaluator = ModelEvaluator(engine)
    df_eval, summary = evaluator.evaluate_benchmark(top_k=5)
    
    print("\nDetailed Benchmark Results:")
    for _, r in df_eval.iterrows():
        print(f"Query {r['ID']} [{r['Tema']}]:")
        print(f"   Relevant Hits in Top-5: {r['Relevant Hits']}")
        print(f"   P@5: {r['Precision@5']} | R@5: {r['Recall@5']} | F1: {r['F1-Score']} | RR: {r['Reciprocal Rank']}")
        
    print("\nSummary Metrics:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    print("=" * 60)

if __name__ == '__main__':
    main()
