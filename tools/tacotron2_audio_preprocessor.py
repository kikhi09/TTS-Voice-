#!/usr/bin/env python3
"""
Tacotron 2 Audio Preprocessor
Based on: https://github.com/rasmurtech/Tacotron-2-Audio-Preprocessor

Preprocesses audio files for training Tacotron 2 or modern TTS voice models:
- Trims leading and trailing silence (configurable dB threshold)
- Normalizes volume to target peak (-0.5 dB)
- Converts stereo to mono
- Resamples to standard 22,050 Hz (or 24,000 Hz)
- Saves clean, consistent 16-bit PCM WAV files
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import soxr


def trim_silence(audio, sr, top_db=20, pad_ms=50):
    if len(audio) == 0:
        return audio
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio

    threshold = max_val * (10 ** (-top_db / 20.0))
    non_silent = np.where(np.abs(audio) > threshold)[0]
    if len(non_silent) == 0:
        return audio

    pad_samples = int(sr * (pad_ms / 1000.0))
    start_idx = max(0, non_silent[0] - pad_samples)
    end_idx = min(len(audio), non_silent[-1] + pad_samples)
    return audio[start_idx:end_idx]


def normalize_audio(audio, target_peak=0.95):
    if len(audio) == 0:
        return audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return (audio / max_val) * target_peak
    return audio


def preprocess_folder(
    input_folder,
    output_folder,
    target_sr=22050,
    top_db=20,
    pad_ms=50,
    normalize=True,
    sequential_naming=False,
):
    in_dir = Path(input_folder).resolve()
    out_dir = Path(output_folder).resolve()

    if not in_dir.exists():
        print(f"Error: Input folder '{in_dir}' does not exist.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    supported_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}
    files = sorted([f for f in in_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts])

    if not files:
        print(f"No audio files found in '{in_dir}'.")
        return 0

    print(f"Found {len(files)} audio files to preprocess.")
    print(f"Target settings: {target_sr} Hz, Mono, 16-bit PCM WAV, Silence Threshold: {top_db} dB")

    processed = 0
    for idx, f in enumerate(files, start=1):
        try:
            # Read audio
            try:
                data, sr = sf.read(str(f))
            except Exception:
                from pydub import AudioSegment
                seg = AudioSegment.from_file(str(f))
                data = np.array(seg.get_array_of_samples(), dtype=np.float32)
                if seg.channels > 1:
                    data = data.reshape((-1, seg.channels))
                max_int = float(1 << (8 * seg.sample_width - 1))
                data = data / max_int
                sr = seg.frame_rate

            # Convert to mono
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # Pre-trim normalization
            if normalize:
                data = normalize_audio(data)

            # Trim silence
            data = trim_silence(data, sr, top_db=top_db, pad_ms=pad_ms)

            # Final normalization
            if normalize:
                data = normalize_audio(data)

            # Resample
            if sr != target_sr:
                data = soxr.resample(data, sr, target_sr)

            # Output filename
            out_name = f"{idx}.wav" if sequential_naming else f"{f.stem}.wav"
            out_path = out_dir / out_name

            sf.write(str(out_path), data, target_sr, subtype="PCM_16")
            processed += 1
            print(f"[{processed}/{len(files)}] Preprocessed: {f.name} -> {out_name}")

        except Exception as e:
            print(f"Error processing '{f.name}': {e}")

    print(f"\nCompleted! {processed}/{len(files)} files saved to: {out_dir}")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio files for Tacotron 2 / TTS training.")
    parser.add_argument("--input", "-i", help="Input directory of audio files")
    parser.add_argument("--output", "-o", help="Output directory for preprocessed WAV files")
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate in Hz (default: 22050)")
    parser.add_argument("--top_db", type=int, default=20, help="Silence trimming threshold in dB (default: 20)")
    parser.add_argument("--sequential", action="store_true", help="Rename files to 1.wav, 2.wav, etc.")

    args = parser.parse_args()
    input_folder = args.input or input("Enter input audio folder: ").strip().strip('"').strip("'")
    output_folder = args.output or input("Enter output audio folder: ").strip().strip('"').strip("'")

    if not input_folder or not output_folder:
        print("Missing folder arguments. Exiting.")
        sys.exit(1)

    preprocess_folder(
        input_folder,
        output_folder,
        target_sr=args.sr,
        top_db=args.top_db,
        sequential_naming=args.sequential,
    )


if __name__ == "__main__":
    main()
