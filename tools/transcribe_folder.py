#!/usr/bin/env python3
"""
transcribe_folder.py — Standalone CLI tool (from rasmurtech/Tacotron2-Wav2Vec-Transcription)

Transcribes a folder of numbered WAV files using faster-whisper (local model)
and writes an LJSpeech-format list.txt:
    /content/TTS-TT2/wavs/1.wav|TRANSCRIPT

Usage:
    python transcribe_folder.py <input_folder> [--output list.txt] [--model base] [--prefix /content/TTS-TT2/wavs]

Requirements: faster-whisper (already in voiceapp310 env)
"""

import os
import sys
import argparse
import glob
import re

def find_local_model(model_name="base"):
    """Search for a local faster-whisper model snapshot."""
    # Common search roots
    search_roots = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "huggingface", "hub"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.environ.get("HF_HOME", ""),
    ]
    patterns = [
        f"models--Systran--faster-whisper-{model_name}",
        f"models--openai--whisper-{model_name}",
    ]
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        for pat in patterns:
            model_dir = os.path.join(root, pat, "snapshots")
            if os.path.isdir(model_dir):
                snapshots = sorted(os.listdir(model_dir))
                if snapshots:
                    snap_path = os.path.join(model_dir, snapshots[-1])
                    print(f"[INFO] Found local model: {snap_path}")
                    return snap_path
    return None


def get_sorted_wav_files(folder):
    """Return WAV files sorted numerically (1.wav, 2.wav, ... before 10.wav)."""
    wav_files = glob.glob(os.path.join(folder, "*.wav"))

    def sort_key(path):
        name = os.path.splitext(os.path.basename(path))[0]
        # Try numeric sort first
        try:
            return (0, int(name), name)
        except ValueError:
            return (1, 0, name)

    return sorted(wav_files, key=sort_key)


def transcribe_folder(input_folder, output_file, model_name, wav_prefix, language=None, device="cpu"):
    """Transcribe all WAV files in folder and write LJSpeech list.txt."""
    from faster_whisper import WhisperModel

    # Locate model
    model_path = find_local_model(model_name)
    if model_path:
        model_kwargs = dict(
            model_size_or_path=model_path,
            device=device,
            compute_type="int8",
            local_files_only=True,
        )
    else:
        print(f"[INFO] Local model not found, downloading '{model_name}' from HuggingFace...")
        model_kwargs = dict(
            model_size_or_path=model_name,
            device=device,
            compute_type="int8",
        )

    print(f"[INFO] Loading faster-whisper model ({model_name})...")
    model = WhisperModel(**model_kwargs)

    wav_files = get_sorted_wav_files(input_folder)
    if not wav_files:
        print(f"[ERROR] No WAV files found in: {input_folder}")
        sys.exit(1)

    print(f"[INFO] Found {len(wav_files)} WAV file(s) to transcribe.")

    lines = []
    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        name_no_ext = os.path.splitext(filename)[0]

        # Build the path as it will appear in list.txt
        if wav_prefix:
            list_path = f"{wav_prefix.rstrip('/')}/{filename}"
        else:
            list_path = wav_path.replace("\\", "/")

        # Transcribe
        try:
            segments, info = model.transcribe(
                wav_path,
                language=language,
                beam_size=5,
                vad_filter=True,
            )
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
            transcript = re.sub(r"\s+", " ", transcript)
            if not transcript:
                transcript = "[SILENCE]"
            print(f"  {filename} → {transcript}")
        except Exception as e:
            print(f"  [WARN] Failed to transcribe {filename}: {e}")
            transcript = "[ERROR]"

        lines.append(f"{list_path}|{transcript}")

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[DONE] Wrote {len(lines)} entries to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a folder of WAV files to LJSpeech list.txt format using faster-whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_folder.py wavs/
  python transcribe_folder.py wavs/ --output list.txt --prefix /content/TTS-TT2/wavs
  python transcribe_folder.py wavs/ --model small --language en
        """
    )
    parser.add_argument("input_folder", help="Folder containing numbered WAV files (1.wav, 2.wav, ...)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file path (default: list.txt inside input_folder)")
    parser.add_argument("--model", "-m", default="base",
                        help="Whisper model size: tiny, base, small, medium, large (default: base)")
    parser.add_argument("--prefix", "-p", default="/content/TTS-TT2/wavs",
                        help="Path prefix for WAV entries in list.txt (default: /content/TTS-TT2/wavs)")
    parser.add_argument("--language", "-l", default=None,
                        help="Language code, e.g. 'en', 'fr' (default: auto-detect)")
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda", "auto"],
                        help="Device to run model on (default: cpu)")
    parser.add_argument("--no-prefix", action="store_true",
                        help="Use actual file paths instead of a prefix in list.txt")

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"[ERROR] Input folder does not exist: {args.input_folder}")
        sys.exit(1)

    output_file = args.output or os.path.join(args.input_folder, "list.txt")
    wav_prefix = None if args.no_prefix else args.prefix

    print(f"[INFO] Input folder : {args.input_folder}")
    print(f"[INFO] Output file  : {output_file}")
    print(f"[INFO] WAV prefix   : {wav_prefix or '(actual paths)'}")
    print(f"[INFO] Model        : {args.model}")
    print(f"[INFO] Language     : {args.language or 'auto-detect'}")
    print()

    transcribe_folder(
        input_folder=args.input_folder,
        output_file=output_file,
        model_name=args.model,
        wav_prefix=wav_prefix,
        language=args.language,
        device=args.device,
    )


if __name__ == "__main__":
    main()
