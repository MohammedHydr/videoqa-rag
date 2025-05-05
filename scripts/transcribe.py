# scripts/transcribe.py
from faster_whisper import WhisperModel
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.common import RAW, PROC, save_json


def main(model_size="medium"):
    PROC.mkdir(parents=True, exist_ok=True)

    print("🟢 Loading Whisper model:", model_size)
    model = WhisperModel(
        model_size_or_path=model_size,
        device="cuda",
        compute_type="float16",  # Use float16 for best speed/quality tradeoff
    )

    print("📥 Transcribing audio.wav …")
    segments, _ = model.transcribe(str(RAW / "audio.wav"), vad_filter=True)

    output = []
    for seg in segments:
        output.append({
            "text": seg.text.strip(),
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
        })

    save_json(output, PROC / "transcript.json")
    print("✅ Saved:", PROC / "transcript.json")


if __name__ == "__main__":
    main()
