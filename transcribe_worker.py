import json
import os
import sys
from pathlib import Path

# Ensure paths and environment are properly configured
PROJECT_DIR = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(PROJECT_DIR / "models" / "huggingface")

from faster_whisper import WhisperModel


def find_local_whisper_snapshot():
    """Finds the local faster-whisper-base snapshot if available offline."""
    base_snapshot_dir = PROJECT_DIR / "models" / "huggingface" / "hub" / "models--Systran--faster-whisper-base" / "snapshots"
    if base_snapshot_dir.exists():
        snapshots = list(base_snapshot_dir.iterdir())
        if snapshots:
            return str(snapshots[0])
    return "base"


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No audio path provided"}), file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]
    model_path = find_local_whisper_snapshot()

    # Prefer CPU with int8 for fast, reliable local transcription without CUDA driver issues
    try:
        model = WhisperModel(model_path, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:
        model = WhisperModel(model_path, device="auto", compute_type="auto")

    segments, info = model.transcribe(audio_path, beam_size=5)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    print(json.dumps({"text": transcript, "language": info.language}, ensure_ascii=False))


if __name__ == "__main__":
    main()
