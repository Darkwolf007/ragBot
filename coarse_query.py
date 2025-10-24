"""
semantic_filter.py
------------------
3-Stage Hierarchical Semantic Filtering for Building Rules Retrieval
1. Coarse topic filtering
2. Fine semantic retrieval via FAISS
3. Cross-encoder reranking
"""

import faiss, json, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
OUT_DIR = "output"
INDEX_PATH = f"{OUT_DIR}/pdf_index.faiss"
META_PATH  = f"{OUT_DIR}/metadata.json"

# choose light models for speed (upgrade later)
COARSE_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# domain-specific coarse topics for TN Combined Building Rules
TOPICS = [
    "zoning regulations",
    "building height limits",
    "setbacks and open space",
    "floor area ratio FAR FSI",
    "parking provisions",
    "fire safety",
    "ventilation and lighting",
    "rainwater harvesting",
    "staircase and lift rules",
    "structural stability",
    "miscellaneous provisions"
]

# -------------------------------------------------------------------
# LOAD MODELS + INDEX
# -------------------------------------------------------------------
print("🔧 Loading models and index ...")
bi_encoder = SentenceTransformer(COARSE_MODEL)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
index = faiss.read_index(INDEX_PATH)
metadata = json.load(open(META_PATH, encoding="utf-8"))

topic_emb = bi_encoder.encode(TOPICS, normalize_embeddings=True)
topic_index = faiss.IndexFlatIP(topic_emb.shape[1])
topic_index.add(topic_emb)

# -------------------------------------------------------------------
# STAGE 1 — COARSE TOPIC FILTER
# -------------------------------------------------------------------
def coarse_filter(query, top_n=2):
    """Find most relevant topics for the query."""
    q_emb = bi_encoder.encode([query], normalize_embeddings=True)
    D, I = topic_index.search(np.array(q_emb, dtype="float32"), top_n)
    selected = [TOPICS[i] for i in I[0]]
    print(f"🧩 Coarse filter topics: {selected}")
    return selected

# -------------------------------------------------------------------
# STAGE 2 — FINE FAISS SIMILARITY
# -------------------------------------------------------------------
def fine_retrieve(query, k=20):
    """Retrieve top-k relevant chunks from FAISS index."""
    q_emb = bi_encoder.encode([query], normalize_embeddings=True)
    # match FAISS dimension (padding)
    if q_emb.shape[1] < index.d:
        pad = np.zeros((1, index.d - q_emb.shape[1]), dtype="float32")
        q_emb = np.concatenate([q_emb, pad], axis=1)
    elif q_emb.shape[1] > index.d:
        q_emb = q_emb[:, :index.d]

    D, I = index.search(np.array(q_emb, dtype="float32"), k)
    candidates = [metadata[i] | {"score_bi": float(D[0][r])} for r, i in enumerate(I[0])]
    return candidates

# -------------------------------------------------------------------
# STAGE 3 — CROSS-ENCODER RERANKING
# -------------------------------------------------------------------
def rerank(query, candidates, top_n=5):
    """Re-rank candidates using a cross-encoder."""
    pairs, valid = [], []
    for c in candidates:
        text = c.get("content", c.get("ocr_text", ""))
        if text.strip():
            pairs.append([query, text])
            valid.append(c)

    if not pairs:
        return []
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(valid, scores), key=lambda x: x[1], reverse=True)
    results = []
    for i, (c, s) in enumerate(ranked[:top_n]):
        results.append({**c, "score_cross": float(s), "rank": i + 1})
    return results

# -------------------------------------------------------------------
# MASTER FUNCTION
# -------------------------------------------------------------------
def hierarchical_search(query, k_fine=20, top_n=5):
    """Run all 3 stages in sequence."""
    print(f"\n🔍 Query: {query}")
    coarse_topics = coarse_filter(query)
    candidates = fine_retrieve(query, k=k_fine)
    reranked = rerank(query, candidates, top_n=top_n)
    return reranked

# -------------------------------------------------------------------
# CLI TEST
# -------------------------------------------------------------------
if __name__ == "__main__":
    while True:
        q = input("\n❓ Enter your query (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        results = hierarchical_search(q)
        print("\n🎯 Top Results:")
        for r in results:
            t = (r.get("content") or r.get("ocr_text", "")).replace("\n", " ")
            print(f"#{r['rank']} [Page {r['page']}] Score={r['score_cross']:.3f}")
            print("   ", t[:250], "...\n")
