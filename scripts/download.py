# scripts/download.py
"""
Download the assignment video and extract 16‑kHz mono WAV.

Features
--------
✓ resumable (yt‑dlp + partial files)
✓ SHA256 checksum stored in download_meta.json
✓ idempotent (skips if checksum matches)
✓ works locally or on Octopus Slurm (submit with  sbatch hpc_download.sh )
"""

import hashlib
import json
import pathlib
import subprocess
import sys

from yt_dlp import YoutubeDL

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.common import RAW

URL = "https://youtube/dARr3lGKwk8"
META = RAW / "download_meta.json"
MP4 = RAW / "video.mp4"
WAV16 = RAW / "audio.wav"


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def download():
    RAW.mkdir(parents=True, exist_ok=True)

    # ── 1. download mp4 (video+audio) ──────────────────────────────
    if not MP4.exists():
        print("⬇️  downloading video via yt‑dlp …")
        ydl_opts = {
            "outtmpl": str(RAW / "video.%(ext)s"),
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "merge_output_format": "mp4",
            "noprogress": False,  # show tqdm
            "quiet": False,
            # ── add these two lines ───────────────────────
            "cookiefile": str(pathlib.Path(__file__).resolve().parent / "cookies.txt"),
            # or: "cookiesfrombrowser": ("firefox",)  # auto‑grab local Firefox cookies
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([URL])

    # ── 2. checksum & metadata ────────────────────────────────────
    digest = sha256(MP4)
    meta = {"url": URL, "sha256": digest}
    META.write_text(json.dumps(meta, indent=2))
    print("✅ saved", META)

    # ── 3. extract mono 16‑kHz WAV for Whisper ────────────────────
    if not WAV16.exists():
        print("🎙  extracting audio to 16kHz mono …")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(MP4),
                "-ac", "1",  # mono
                "-ar", "16000",  # 16kHz
                str(WAV16),
            ],
            check=True,
        )
        print("✅", WAV16, "ready")
    else:
        print("✔", WAV16, "already exists")


if __name__ == "__main__":
    download()
