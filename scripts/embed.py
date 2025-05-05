"""
Generate and save text and image embeddings:
- Text  : BGE-Large (384-dim)  → text.npy
- Image : CLIP ViT-B/32 (512-dim) → img.npy
"""
import pathlib
import sys

import numpy as np
import torch
import open_clip
from PIL import Image
from sentence_transformers import SentenceTransformer
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.common import PROC, load_transcript


# ─── Text Embeddings ──────────────────────────────────────────────
def embed_text():
    print("📘 Encoding transcript using BGE-Large...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    texts = [chunk["text"] for chunk in load_transcript()]
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True)
    np.save(PROC / "text.npy", embeddings.astype("float32"))
    print(f"✅ Saved {len(embeddings)} text embeddings to text.npy")


# ─── Image Embeddings ─────────────────────────────────────────────
def embed_images():
    print("🖼️ Encoding keyframes using CLIP ViT-B/32...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.eval().to("cuda")

    frame_paths = sorted((PROC / "frames").glob("*.jpg"), key=lambda p: float(p.stem.split("_")[1]))
    embeddings = []

    with torch.no_grad():
        for i, path in enumerate(frame_paths):
            if i % 100 == 0:
                print(f"  → {i}/{len(frame_paths)} frames")
            image = preprocess(Image.open(path)).unsqueeze(0).to("cuda")
            vector = model.encode_image(image)
            vector /= vector.norm(dim=-1, keepdim=True)
            embeddings.append(vector.cpu())

    np.save(PROC / "img.npy", torch.vstack(embeddings).numpy().astype("float32"))
    print(f"✅ Saved {len(embeddings)} image embeddings to img.npy")


if __name__ == "__main__":
    embed_text()
    embed_images()
