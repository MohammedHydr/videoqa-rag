# scripts/transcribe.py
"""
Transcribe audio.wav using Faster-Whisper (CPU version for HPC compatibility)
▸ Output: transcript.json in data/processed/
"""

import pathlib
import sys

from faster_whisper import WhisperModel

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.common import RAW, PROC, save_json


def main(model_size="medium.en"):
    PROC.mkdir(parents=True, exist_ok=True)

    print("🟢 Loading Whisper model on CPU...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print("📥 Transcribing audio.wav...")
    segments, _ = model.transcribe(str(RAW / "audio.wav"), vad_filter=True)

    transcript = [
        {"text": seg.text.strip(), "start": seg.start, "end": seg.end}
        for seg in segments
    ]

    save_json(transcript, PROC / "transcript.json")
    print(f"✅ Done! Transcript saved to {PROC / 'transcript.json'}")


if __name__ == "__main__":
    main()
