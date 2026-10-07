from search import vector_search, keyword_search, hybrid_search, print_results


def run_benchmarks():
    # -------------------------------------------------------------
    # Experiment 1: The Exact Identifier Query
    # ( I gotta notice how pure vector search can get distracted, but keywords excel)
    # -------------------------------------------------------------
    query_1 = "Find invoice INV-2026-904"
    print(f"\n=======================================================")
    print(f"QUERY 1: '{query_1}'")
    print(f"=======================================================")

    v_res = vector_search(query_1, limit=2)
    print_results("Pure Vector Search", v_res, score_key="score")

    k_res = keyword_search(query_1, limit=2)
    print_results("Pure Keyword Search (BM25/FTS)", k_res, score_key="score")

    h_res = hybrid_search(query_1, limit=2)
    print_results("Hybrid Search (RRF)", h_res, score_key="rrf_score")

    # -------------------------------------------------------------
    # Experiment 2: The Conceptual / Paraphrase Query
    # The document says: 'Broiler thermal management... ambient temperatures exceed 32°C... heat exhaustion'
    # But the query mentions: 'keep the birds from dying when it is too hot'
    # (so the keyword search finds ZERO hits, but vector search excels)
    # -------------------------------------------------------------
    query_2 = "How do we keep the birds from dying when it is too hot?"
    print(f"\n=======================================================")
    print(f"QUERY 2: '{query_2}'")
    print(f"=======================================================")

    v_res2 = vector_search(query_2, limit=2)
    print_results("Pure Vector Search", v_res2, score_key="score")

    k_res2 = keyword_search(query_2, limit=2)
    print_results("Pure Keyword Search (BM25/FTS)", k_res2, score_key="score")

    h_res2 = hybrid_search(query_2, limit=2)
    print_results("Hybrid Search (RRF)", h_res2, score_key="rrf_score")


if __name__ == "__main__":
    run_benchmarks()
