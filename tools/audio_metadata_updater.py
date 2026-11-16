#!/usr/bin/env python3
"""
audio_metadata_updater.py — Standalone CLI tool (from rasmurtech/Audio-Metadata-Updater)

Updates RIFF INFO metadata (title/track number) in WAV files using pure Python
(no taglib required). Uses the native write_riff_info_chunk() from audio_processing.

Usage:
    python audio_metadata_updater.py <input_folder> [--output <output_folder>]
                                     [--title-prefix "Recording"]
                                     [--start-track 1]
                                     [--inplace]

Requirements: Python 3.6+, no extra packages needed
"""

import os
import sys
import struct
import argparse
import shutil
import glob

# ---------------------------------------------------------------------------
# Pure-Python RIFF INFO chunk reader/writer (no taglib needed)
# ---------------------------------------------------------------------------

def read_riff_info(wav_path):
    """Read RIFF INFO tags from a WAV file. Returns dict of {tag: value}."""
    tags = {}
    try:
        with open(wav_path, "rb") as f:
            header = f.read(12)
            if len(header) < 12:
                return tags
            riff_id, file_size, wave_id = struct.unpack_from("<4sI4s", header)
            if riff_id != b"RIFF" or wave_id != b"WAVE":
                return tags
            end_pos = file_size + 8  # file_size is from byte 8
            while f.tell() < end_pos:
                chunk_hdr = f.read(8)
                if len(chunk_hdr) < 8:
                    break
                chunk_id, chunk_size = struct.unpack_from("<4sI", chunk_hdr)
                chunk_data = f.read(chunk_size)
                if chunk_size % 2:
                    f.read(1)  # pad byte
                if chunk_id == b"LIST":
                    list_type = chunk_data[:4]
                    if list_type == b"INFO":
                        pos = 4
                        while pos < len(chunk_data) - 8:
                            tag_id = chunk_data[pos:pos+4].decode("ascii", errors="replace")
                            tag_size = struct.unpack_from("<I", chunk_data, pos+4)[0]
                            tag_val = chunk_data[pos+8:pos+8+tag_size]
                            # Strip null terminator
                            tag_val = tag_val.rstrip(b"\x00").decode("utf-8", errors="replace")
                            tags[tag_id] = tag_val
                            pos += 8 + tag_size
                            if tag_size % 2:
                                pos += 1
    except Exception:
        pass
    return tags


def write_riff_info(wav_path, out_path, info_dict):
    """
    Write RIFF INFO tags to a WAV file.
    Reads existing audio data, strips old LIST INFO chunk, inserts new one.
    info_dict: {tag_fourcc: value_string}, e.g. {"INAM": "My Title", "ITRK": "1"}
    """
    # Read original file
    with open(wav_path, "rb") as f:
        data = f.read()

    if len(data) < 12:
        shutil.copy2(wav_path, out_path)
        return False

    riff_id = data[0:4]
    wave_id = data[8:12]
    if riff_id != b"RIFF" or wave_id != b"WAVE":
        shutil.copy2(wav_path, out_path)
        return False

    # Collect all chunks except existing LIST INFO
    chunks = []
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos+4]
        chunk_size = struct.unpack_from("<I", data, pos+4)[0]
        chunk_data = data[pos+8:pos+8+chunk_size]
        padded_size = chunk_size + (chunk_size % 2)
        next_pos = pos + 8 + padded_size

        # Skip existing LIST INFO chunks
        if chunk_id == b"LIST" and len(chunk_data) >= 4 and chunk_data[:4] == b"INFO":
            pos = next_pos
            continue

        chunks.append((chunk_id, chunk_data))
        pos = next_pos

    # Build new LIST INFO chunk
    info_payload = b"INFO"
    for tag_key, tag_val in info_dict.items():
        if not tag_val:
            continue
        tag_bytes = tag_key.encode("ascii")[:4].ljust(4, b"\x00")
        val_bytes = tag_val.encode("utf-8") + b"\x00"
        if len(val_bytes) % 2:
            val_bytes += b"\x00"  # pad to even
        info_payload += tag_bytes
        info_payload += struct.pack("<I", len(val_bytes) - (1 if len(val_bytes) % 2 == 0 else 0))
        # Actually store raw (with null term, padded)
        raw_val = tag_val.encode("utf-8") + b"\x00"
        if len(raw_val) % 2:
            raw_val += b"\x00"
        info_payload = info_payload[:-4]  # undo last size pack
        size_val = len(raw_val) - 1  # size without pad
        info_payload += struct.pack("<I", size_val)
        info_payload += raw_val

    list_chunk = b"LIST" + struct.pack("<I", len(info_payload)) + info_payload

    # Reconstruct WAV
    wav_body = b"WAVE"
    for cid, cdata in chunks:
        wav_body += cid + struct.pack("<I", len(cdata)) + cdata
        if len(cdata) % 2:
            wav_body += b"\x00"
    wav_body += list_chunk

    new_data = b"RIFF" + struct.pack("<I", len(wav_body)) + wav_body

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(new_data)
    return True


# Try to import from parent package for a richer implementation
def _try_import_audio_processing():
    try:
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from audio_processing import write_riff_info_chunk
        return write_riff_info_chunk
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def update_folder_metadata(input_folder, output_folder, title_prefix, start_track, inplace, dry_run):
    """Update RIFF INFO metadata for all WAV files in input_folder."""

    wav_files = sorted(
        glob.glob(os.path.join(input_folder, "*.wav")),
        key=lambda p: (
            int(os.path.splitext(os.path.basename(p))[0])
            if os.path.splitext(os.path.basename(p))[0].isdigit()
            else os.path.basename(p)
        )
    )

    if not wav_files:
        print(f"[ERROR] No WAV files found in: {input_folder}")
        sys.exit(1)

    print(f"[INFO] Found {len(wav_files)} WAV file(s).")
    if dry_run:
        print("[INFO] DRY RUN — no files will be modified.")
    print()

    if output_folder and not inplace and not dry_run:
        os.makedirs(output_folder, exist_ok=True)

    success = 0
    for idx, wav_path in enumerate(wav_files):
        track_num = start_track + idx
        filename = os.path.basename(wav_path)
        name_no_ext = os.path.splitext(filename)[0]

        # Build metadata
        title = f"{title_prefix} {track_num}" if title_prefix else name_no_ext
        info_dict = {
            "INAM": title,       # Title / Name
            "ITRK": str(track_num),  # Track number
            "ISFT": "VoiceApp Studio",  # Software
        }

        # Determine output path
        if inplace:
            out_path = wav_path
        elif output_folder:
            out_path = os.path.join(output_folder, filename)
        else:
            out_path = wav_path  # fallback = inplace

        if dry_run:
            print(f"  [{track_num:4d}] {filename}")
            print(f"         INAM={title!r}  ITRK={track_num}")
            continue

        result = write_riff_info(wav_path, out_path, info_dict)
        if result:
            print(f"  [{track_num:4d}] {filename} → INAM={title!r}  ITRK={track_num}")
            success += 1
        else:
            print(f"  [WARN] Skipped (not a valid WAV): {filename}")

    if not dry_run:
        print(f"\n[DONE] Updated {success}/{len(wav_files)} WAV files.")
        if output_folder and not inplace:
            print(f"       Output folder: {output_folder}")
    else:
        print(f"\n[DRY RUN] Would update {len(wav_files)} WAV files.")


def show_metadata(wav_path):
    """Show existing RIFF INFO tags for a WAV file."""
    tags = read_riff_info(wav_path)
    if tags:
        print(f"[INFO] Metadata in {os.path.basename(wav_path)}:")
        for k, v in tags.items():
            print(f"  {k}: {v!r}")
    else:
        print(f"[INFO] No RIFF INFO metadata found in {os.path.basename(wav_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Update RIFF INFO metadata (title, track number) in WAV files. Pure Python, no taglib needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all WAVs in-place with auto-numbered titles
  python audio_metadata_updater.py wavs/ --inplace

  # Write to a new folder, title prefix "Voice"
  python audio_metadata_updater.py wavs/ --output wavs_tagged/ --title-prefix "Voice"

  # Start track numbering at 100
  python audio_metadata_updater.py wavs/ --inplace --start-track 100

  # Dry run to preview changes
  python audio_metadata_updater.py wavs/ --dry-run

  # Show existing metadata for one file
  python audio_metadata_updater.py --show-file wavs/1.wav
        """
    )
    parser.add_argument("input_folder", nargs="?", help="Folder containing WAV files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output folder (default: same as input, requires --inplace or --output)")
    parser.add_argument("--title-prefix", "-t", default="",
                        help="Title prefix for INAM tag, e.g. 'Recording' → 'Recording 1' (default: use filename)")
    parser.add_argument("--start-track", "-s", type=int, default=1,
                        help="Starting track number for ITRK tag (default: 1)")
    parser.add_argument("--inplace", "-i", action="store_true",
                        help="Modify WAV files in-place (overwrites originals)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Preview what would be changed without modifying files")
    parser.add_argument("--show-file", metavar="WAV_FILE",
                        help="Show existing metadata for a single WAV file and exit")

    args = parser.parse_args()

    # Show metadata for a single file
    if args.show_file:
        if not os.path.isfile(args.show_file):
            print(f"[ERROR] File not found: {args.show_file}")
            sys.exit(1)
        show_metadata(args.show_file)
        return

    if not args.input_folder:
        parser.print_help()
        sys.exit(1)

    if not os.path.isdir(args.input_folder):
        print(f"[ERROR] Input folder does not exist: {args.input_folder}")
        sys.exit(1)

    if not args.inplace and not args.output and not args.dry_run:
        print("[ERROR] Specify --inplace to modify in-place, or --output <folder> for a copy, or --dry-run to preview.")
        sys.exit(1)

    update_folder_metadata(
        input_folder=args.input_folder,
        output_folder=args.output,
        title_prefix=args.title_prefix,
        start_track=args.start_track,
        inplace=args.inplace,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
