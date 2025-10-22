"""
query_vector_db.py
------------------
Query your Tamil Nadu Combined Building Rules vector index.
Supports semantic search across text, OCR, and image embeddings.
"""

import json, numpy as np, faiss, os, textwrap
from sentence_transformers import SentenceTransformer
from PIL import Image

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
OUT_DIR = "output"
INDEX_PATH = f"{OUT_DIR}/pdf_index.faiss"
META_PATH  = f"{OUT_DIR}/metadata.json"
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------
# LOAD INDEX + MODEL
# --------------------------------------------------
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("❌ Vector index not found. Run build_pdf_index.py first.")

print("Loading vector DB & model ...")
index = faiss.read_index(INDEX_PATH)
metadata = json.load(open(META_PATH, "r", encoding="utf-8"))
model = SentenceTransformer(TEXT_MODEL_NAME)

print(f"✅ Loaded {len(metadata)} entries | Dim: {index.d} ")

# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------
def search_query(query: str, k: int = 5):
    """Search top-k matches for a natural-language query."""
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")

    # --- match FAISS dimension ---
    if q_emb.shape[1] < index.d:
        pad = np.zeros((1, index.d - q_emb.shape[1]), dtype="float32")
        q_emb = np.concatenate([q_emb, pad], axis=1)
    elif q_emb.shape[1] > index.d:
        q_emb = q_emb[:, :index.d]

    D, I = index.search(q_emb, k)
    results = []
    for rank, idx in enumerate(I[0]):
        item = metadata[idx]
        score = float(D[0][rank])
        results.append({"rank": rank + 1, "score": score, **item})
    return results



# --------------------------------------------------
# DISPLAY FUNCTION
# --------------------------------------------------
def pretty_print(results):
    print("\n🔎  Top Matches:\n" + "-"*80)
    for r in results:
        print(f"#{r['rank']}  (score {r['score']:.3f})  —  Page {r['page']}  |  Type: {r['type']}")
        if r["type"] == "text":
            snippet = textwrap.shorten(r["content"].replace("\n", " "), width=300)
            print(f"    {snippet}")
        elif r["type"] == "image":
            print(f"    🖼️  {r['path']}")
            ocr = textwrap.shorten(r.get('ocr_text','').replace("\n"," "), width=150)
            if ocr:
                print(f"    OCR: {ocr}")
        print("-"*80)
    print()


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
if __name__ == "__main__":
    print("\nTamil Nadu Combined Building Rules — Semantic Search")
    print("Type a question (or 'exit'):\n")
    while True:
        query = input("❓ Query > ").strip()
        if not query or query.lower() in ("exit", "quit", "q"):
            print("👋 Goodbye!")
            break
        results = search_query(query, k=5)
        pretty_print(results)
