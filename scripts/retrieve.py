"""
Multimodal Retrieval System:
- Text  : BGE-Large (FAISS HNSW)
- Image : CLIP ViT-B/32 (FAISS HNSW)
- Lexical: BM25 (exact match)
→ Late fusion with configurable weights.
"""
import pathlib
import sys

import numpy as np
import faiss
import pickle
import joblib
import torch
import open_clip
from open_clip import tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.common import PROC, load_transcript, np_load

# ─── Load Once ─────────────────────────────────────────────────────

chunks = load_transcript()
text_hnsw = faiss.read_index(str(PROC / "text_hnsw.index"))
img_hnsw = faiss.read_index(str(PROC / "img_hnsw.index"))
bm25 = pickle.load(open(PROC / "bm25.pkl", "rb"))
tfidf: TfidfVectorizer = joblib.load(PROC / "tfidf.joblib")

text_vecs = np_load(PROC / "text.npy")
frame_files = sorted((PROC / "frames").glob("*.jpg"), key=lambda p: float(p.stem.split('_')[1]))

# Models
bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.eval().to("cuda")


# ─── Embedding Encoders ────────────────────────────────────────────

def encode_text(text: str) -> np.ndarray:
    """BGE encoder"""
    emb = bge_model.encode([text], normalize_embeddings=True)
    return emb.astype("float32")


def encode_image_text(text: str) -> np.ndarray:
    """CLIP text encoder for visual grounding"""
    with torch.no_grad():
        tokens = tokenize([text]).to("cuda")
        vec = clip_model.encode_text(tokens)
        vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy().astype("float32")


# ─── Main Retrieval ────────────────────────────────────────────────

def retrieve(query: str,
             mode: str = "Fused (default)",
             k_text=8, k_img=8, k_lex=8,
             w_text=0.6, w_img=0.2, w_lex=0.2,
             thresh=0.12) -> dict | None:

    results = {}
    q_text = encode_text(query)

    # ── 1. Semantic Text Retrieval ─────────────────────
    if mode in ["Fused (default)", "Semantic only"]:
        D_txt, I_txt = text_hnsw.search(q_text, k_text)
        sim_txt = 1 - 0.5 * D_txt[0]  # L2 → cosine
        for idx, score in zip(I_txt[0], sim_txt):
            results.setdefault(idx, {})["score_text"] = float(score)

    # ── 2. Image Semantic Retrieval ─────────────────────
    if mode == "Fused (default)":
        q_img = encode_image_text(query)
        D_img, I_img = img_hnsw.search(q_img, k_img)
        sim_img = 1 - 0.5 * D_img[0]
        for idx, score in zip(I_img[0], sim_img):
            results.setdefault(idx, {})["score_img"] = float(score)

    # ── 3. Lexical (BM25) Retrieval ─────────────────────
    if mode in ["Fused (default)", "BM25 only"]:
        tokens = query.lower().split()
        bm_scores = bm25.get_scores(tokens)
        top_lex = np.argsort(bm_scores)[::-1][:k_lex]
        max_bm = max(bm_scores[top_lex[0]], 1e-6)
        for idx in top_lex:
            results.setdefault(idx, {})["score_lex"] = float(bm_scores[idx]) / max_bm

    # ── 4. Fusion Scoring ───────────────────────────────
    best_id, best_score = None, 0.0
    for idx, scores in results.items():
        final_score = (
            w_text * scores.get("score_text", 0.0) +
            w_img * scores.get("score_img", 0.0) +
            w_lex * scores.get("score_lex", 0.0)
        )
        if final_score > best_score:
            best_id, best_score = idx, final_score

    if best_score < thresh:
        return None

    chunk = chunks[best_id]
    return {
        "id": best_id,
        "start": chunk["start"],
        "end": chunk["end"],
        "text": chunk["text"],
        "score": best_score
    }
