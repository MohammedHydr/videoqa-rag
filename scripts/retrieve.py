import json
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

from scripts.common import PROC, load_transcript, extract_clip

# Load data
transcript = load_transcript()
text_chunks = [chunk["text"] for chunk in transcript]
text_starts = [chunk["start"] for chunk in transcript]

# Load embeddings
text_embeds = np.load(PROC / "text.npy")
img_embeds = np.load(PROC / "img.npy")

# Load OCR results (from preprocessed frames or compute live if needed)
ocr_data = json.loads((PROC / "ocr.json").read_text()) if (PROC / "ocr.json").exists() else []
ocr_texts = [item["text"] for item in ocr_data]
ocr_timestamps = [item["start"] for item in ocr_data]

# Init models
text_encoder = SentenceTransformer("intfloat/e5-large-v2", device="cuda" if torch.cuda.is_available() else "cpu")

# TF-IDF
tfidf = TfidfVectorizer().fit(text_chunks + ocr_texts)
tfidf_matrix = tfidf.transform(text_chunks + ocr_texts)

# BM25
bm25 = BM25Okapi([doc.split() for doc in text_chunks + ocr_texts])


def retrieve(query: str, mode="Fused (default)"):
    query_embed = text_encoder.encode("query: " + query, normalize_embeddings=True)

    sim_text = util.cos_sim(torch.tensor([query_embed]), torch.tensor(text_embeds))[0].cpu().numpy()
    sim_img = util.cos_sim(torch.tensor([query_embed]), torch.tensor(img_embeds))[0].cpu().numpy()
    sim_tfidf = (tfidf_matrix @ tfidf.transform([query]).T).toarray().flatten()
    sim_bm25 = np.array(bm25.get_scores(query.split()))

    # Fusion depending on selected mode
    if mode == "Semantic Only":
        scores = sim_text
    elif mode == "TF‑IDF Only":
        scores = sim_tfidf
    elif mode == "BM25 Only":
        scores = sim_bm25
    else:
        scores = 0.5 * sim_text + 0.3 * sim_img[:len(sim_text)] + 0.2 * sim_bm25[:len(sim_text)]

    idx = np.argmax(scores)
    best_score = scores[idx]

    # Rejection threshold
    if best_score < 0.25:
        return None

    start = text_starts[idx]
    text = text_chunks[idx]
    path = extract_clip(start)

    return {"start": start, "text": text, "clip": path}
