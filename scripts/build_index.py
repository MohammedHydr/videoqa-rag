# scripts/build_index.py
"""
Build semantic + lexical indexes:
 ✔ FAISS (Flat + HNSW) for embeddings
 ✔ BM25 + TF-IDF for lexical search
"""
import pathlib
import sys

import faiss
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.common import PROC, load_transcript, np_load

# -----------------------------------------------------------------------------
print("▶ Loading transcript chunks...")
chunks = load_transcript()
docs = [c["text"] for c in chunks]

# -----------------------------------------------------------------------------
print("▶ Loading precomputed embeddings...")
text_vecs = np_load(PROC / "text.npy")
img_vecs = np_load(PROC / "img.npy")

# -----------------------------------------------------------------------------
print("▶ Building FAISS Flat index...")
dim_txt = text_vecs.shape[1]
dim_img = img_vecs.shape[1]

text_flat = faiss.IndexFlatL2(dim_txt)
img_flat = faiss.IndexFlatL2(dim_img)

text_flat.add(text_vecs)
img_flat.add(img_vecs)

faiss.write_index(text_flat, str(PROC / "text_flat.index"))
faiss.write_index(img_flat, str(PROC / "img_flat.index"))

# -----------------------------------------------------------------------------
print("▶ Building FAISS HNSW index...")
text_hnsw = faiss.IndexHNSWFlat(dim_txt, 32)
img_hnsw = faiss.IndexHNSWFlat(dim_img, 32)

text_hnsw.hnsw.efConstruction = 200
img_hnsw.hnsw.efConstruction = 200

text_hnsw.add(text_vecs)
img_hnsw.add(img_vecs)

faiss.write_index(text_hnsw, str(PROC / "text_hnsw.index"))
faiss.write_index(img_hnsw, str(PROC / "img_hnsw.index"))

# -----------------------------------------------------------------------------
print("▶ Building BM25 index...")
tokenized_docs = [d.lower().split() for d in docs]
bm25 = BM25Okapi(tokenized_docs)
with open(PROC / "bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

# -----------------------------------------------------------------------------
print("▶ Building TF-IDF index...")
tfidf = TfidfVectorizer(max_features=10000)
tfidf.fit(docs)
joblib.dump(tfidf, PROC / "tfidf.joblib")

print("✅ All indexes built and saved successfully.")
