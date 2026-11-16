import json
import sys

from faster_whisper import WhisperModel


def main():
    audio_path = sys.argv[1]
    model = WhisperModel("base", device="auto", compute_type="auto")
    segments, info = model.transcribe(audio_path, beam_size=5)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    print(json.dumps({"text": transcript, "language": info.language}, ensure_ascii=False))


if __name__ == "__main__":
    main()
