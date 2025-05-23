# scripts/common.py
"""
Common paths & helpers used across the project.
"""

from pathlib import Path
import json
import numpy as np
import subprocess
import uuid

# ── Project root (two levels up from this file) ───────────────────
ROOT = Path(__file__).resolve().parents[1]

# Data directories
RAW = ROOT / "data" / "raw"  # original mp4 + wav
PROC = ROOT / "data" / "processed"  # transcript, frames, embeddings, indexes

RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)


# ── I/O helper functions ───────────────────────────────────────────
def save_json(obj, path: Path) -> None:
    """Pretty‑print JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(path: Path):
    return json.loads(path.read_text())


def load_transcript():
    """Return list[dict] with keys: text, start, end."""
    return load_json(PROC / "transcript.json")


def np_load(path: Path):
    """
    Memory‑map npy file unless it contains Python objects.
    """
    try:
        return np.load(path, mmap_mode="r")
    except ValueError:  # object dtype
        return np.load(path, allow_pickle=True)


def extract_clip(start_time: float, duration: int = 10):
    in_path = RAW / "video.mp4"
    out_path = RAW / "{uuid.uuid4()}.mp4"

    cmd = [
        "ffmpeg", "-ss", str(start_time),
        "-i", in_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y", out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path
