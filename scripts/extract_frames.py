# scripts/extract_frames.py
"""
Extract keyframes from video.mp4 every N seconds.
▸ Output: data/processed/frames/frame_<start>.jpg
▸ Aligned with transcript segments by timestamp
"""
import json
import pathlib
import subprocess
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.common import RAW, PROC, load_transcript

FPS = 1 / 2.0  # Every 2 seconds (change to 1/3.0 for 3s interval)


def extract_frames():
    print("🎞  Extracting keyframes...")

    frames_dir = PROC / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    transcript = load_transcript()
    total_duration = transcript[-1]["end"]

    timestamps = [round(i, 2) for i in frange(0, total_duration, 1 / FPS)]

    for ts in timestamps:
        out_path = frames_dir / f"frame_{ts:.2f}.jpg"
        if out_path.exists():
            continue
        cmd = [
            "ffmpeg", "-ss", str(ts),
            "-i", str(RAW / "video.mp4"),
            "-frames:v", "1",
            "-q:v", "2",  # quality
            str(out_path),
            "-y",  # overwrite
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"✅ Extracted {len(timestamps)} frames to {frames_dir}")


def frange(start, stop, step):
    while start < stop:
        yield start
        start += step


def align_frames_to_transcript():
    transcript = load_transcript()
    frames_dir = PROC / "frames"
    meta = []

    for img_path in sorted(frames_dir.glob("*.jpg")):
        ts = float(img_path.stem.split("_")[1])
        segment = min(transcript, key=lambda c: abs(c["start"] - ts))
        meta.append({
            "frame": img_path.name,
            "start": ts,
            "segment_id": transcript.index(segment),
            "text": segment["text"],
        })

    out_path = PROC / "frames_meta.json"
    out_path.write_text(json.dumps(meta, indent=2))
    print(f"✅ Frame-text mapping saved to {out_path}")


if __name__ == "__main__":
    extract_frames()
    align_frames_to_transcript()
