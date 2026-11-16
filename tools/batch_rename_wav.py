#!/usr/bin/env python3
"""
Batch WAV File Renamer (Tacotron 2 & TTS Dataset Organizer)
Based on: https://github.com/rasmurtech/tacotron2-wav-file-renamer

Renames all WAV audio files in a specified folder sequentially:
1.wav, 2.wav, 3.wav ...
Uses temporary filenames during renaming to avoid collisions.
"""

import argparse
import os
import sys
from pathlib import Path


def rename_wave_files(folder_path, start_index=1):
    folder = Path(folder_path).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Folder '{folder}' does not exist.")
        return 0

    wav_files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".wav"])

    print(f"Folder path: {folder}")
    print(f"Total files in folder: {len(list(folder.iterdir()))}")
    print(f"WAV files found: {len(wav_files)}")

    if not wav_files:
        print("No WAV files found to rename.")
        return 0

    # Step 1: Rename to temporary names to prevent any overwrite collisions
    temp_pairs = []
    for idx, f in enumerate(wav_files, start=start_index):
        temp_file = folder / f"__temp_renaming_{idx:06d}.wav"
        target_file = folder / f"{idx}.wav"
        f.rename(temp_file)
        temp_pairs.append((temp_file, target_file, f.name))

    # Step 2: Rename from temporary names to final 1.wav, 2.wav, etc.
    for temp_file, target_file, orig_name in temp_pairs:
        temp_file.rename(target_file)
        print(f"Renamed '{orig_name}' -> '{target_file.name}'")

    print(f"\nSuccessfully renamed {len(temp_pairs)} WAV files!")
    return len(temp_pairs)


def main():
    parser = argparse.ArgumentParser(description="Sequentially rename WAV files to 1.wav, 2.wav, etc.")
    parser.add_argument("folder", nargs="?", default="", help="Directory containing the WAV files")
    parser.add_argument("--start", type=int, default=1, help="Starting index number (default: 1)")

    args = parser.parse_args()
    folder = args.folder
    if not folder:
        folder = input("Enter path to the folder containing WAV files: ").strip().strip('"').strip("'")

    if not folder:
        print("No folder provided. Exiting.")
        sys.exit(1)

    rename_wave_files(folder, start_index=args.start)


if __name__ == "__main__":
    main()
