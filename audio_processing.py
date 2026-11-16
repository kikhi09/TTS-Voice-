import os
import struct
from pathlib import Path
import numpy as np
import soundfile as sf
import soxr

from settings import SUPPORTED_AUDIO_EXTENSIONS


def write_riff_info_chunk(wav_path, title=None, track_number=None, artist=None):
    """
    Appends or updates a RIFF INFO LIST chunk in a standard WAV file.
    Embeds Title (INAM), Track Number (ITRK), and Artist (IART) natively
    without requiring external C-libraries like taglib.
    """
    try:
        wav_path = str(wav_path)
        with open(wav_path, "rb") as f:
            data = bytearray(f.read())

        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return False

        entries = []
        if title is not None:
            raw_title = str(title).encode("utf-8") + b"\x00"
            if len(raw_title) % 2 != 0:
                raw_title += b"\x00"
            entries.append(b"INAM" + struct.pack("<I", len(str(title).encode("utf-8") + b"\x00")) + raw_title)

        if track_number is not None:
            raw_track = str(track_number).encode("utf-8") + b"\x00"
            if len(raw_track) % 2 != 0:
                raw_track += b"\x00"
            entries.append(b"ITRK" + struct.pack("<I", len(str(track_number).encode("utf-8") + b"\x00")) + raw_track)

        if artist is not None:
            raw_artist = str(artist).encode("utf-8") + b"\x00"
            if len(raw_artist) % 2 != 0:
                raw_artist += b"\x00"
            entries.append(b"IART" + struct.pack("<I", len(str(artist).encode("utf-8") + b"\x00")) + raw_artist)

        if not entries:
            return True

        info_body = b"".join(entries)
        list_chunk = b"LIST" + struct.pack("<I", len(info_body) + 4) + b"INFO" + info_body

        idx = 12
        list_idx = -1
        list_len = 0
        while idx + 8 <= len(data):
            chunk_id = data[idx:idx+4]
            chunk_size = struct.unpack("<I", data[idx+4:idx+8])[0]
            if chunk_id == b"LIST" and data[idx+8:idx+12] == b"INFO":
                list_idx = idx
                list_len = chunk_size + 8
                break
            idx += 8 + chunk_size
            if chunk_size % 2 != 0:
                idx += 1

        if list_idx != -1:
            data = data[:list_idx] + list_chunk + data[list_idx+list_len:]
        else:
            data.extend(list_chunk)

        riff_size = len(data) - 8
        data[4:8] = struct.pack("<I", riff_size)

        with open(wav_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def trim_audio_silence(audio_data, sample_rate, top_db=20, pad_ms=50):
    """
    Trims silence from leading and trailing edges of audio.
    top_db: decibels below peak considered silence (default 20 dB, matching Tacotron2 preprocessor).
    pad_ms: milliseconds of silence to retain at beginning and end (default 50ms).
    """
    if len(audio_data) == 0:
        return audio_data

    max_val = np.max(np.abs(audio_data))
    if max_val == 0:
        return audio_data

    threshold = max_val * (10 ** (-top_db / 20.0))
    non_silent = np.where(np.abs(audio_data) > threshold)[0]

    if len(non_silent) == 0:
        return audio_data

    pad_samples = int(sample_rate * (pad_ms / 1000.0))
    start_idx = max(0, non_silent[0] - pad_samples)
    end_idx = min(len(audio_data), non_silent[-1] + pad_samples)

    return audio_data[start_idx:end_idx]


def normalize_audio(audio_data, target_peak=0.95):
    """
    Normalizes audio peak to target_peak (default 0.95 / ~ -0.5 dB).
    """
    if len(audio_data) == 0:
        return audio_data

    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return (audio_data / max_val) * target_peak
    return audio_data


def preprocess_audio_file(
    input_path,
    output_path,
    target_sr=22050,
    trim_silence=True,
    top_db=20,
    normalize=True,
    update_tags=True,
    title=None,
    track_number=None,
    artist=None,
):
    """
    Preprocesses a single audio file:
    - Loads audio (supports any format supported by soundfile or pydub)
    - Converts to mono
    - Trims silence (if trim_silence=True)
    - Normalizes amplitude (if normalize=True)
    - Resamples to target_sr (22050Hz for Tacotron 2, 24000Hz for XTTS)
    - Saves as 16-bit PCM WAV
    - Updates RIFF metadata tags (if update_tags=True)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load audio
    try:
        data, sr = sf.read(str(input_path))
    except Exception:
        # Fallback to pydub for non-WAV / exotic audio formats
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_file(str(input_path))
        data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        if audio_segment.channels > 1:
            data = data.reshape((-1, audio_segment.channels))
        max_int = float(1 << (8 * audio_segment.sample_width - 1))
        data = data / max_int
        sr = audio_segment.frame_rate

    # Convert to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Normalize before trimming for consistent dB threshold
    if normalize:
        data = normalize_audio(data)

    # Trim silence
    if trim_silence:
        data = trim_audio_silence(data, sr, top_db=top_db)

    # Final normalization
    if normalize:
        data = normalize_audio(data)

    # Resample if needed
    if sr != target_sr:
        data = soxr.resample(data, sr, target_sr)
        sr = target_sr

    # Save as 16-bit PCM WAV
    sf.write(str(output_path), data, sr, subtype="PCM_16")

    # Update metadata tags
    if update_tags:
        write_riff_info_chunk(
            output_path,
            title=title if title is not None else output_path.stem,
            track_number=track_number,
            artist=artist,
        )

    return output_path


def batch_rename_wav_files(folder_path, start_index=1):
    """
    Batch renames all .wav files in a directory sequentially to 1.wav, 2.wav, etc.
    Matches the functionality of rasmurtech/tacotron2-wav-file-renamer.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder_path}")

    files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".wav"])
    if not files:
        return 0, []

    # Two-step rename using temporary names to prevent collisions
    renamed_items = []
    temp_paths = []
    for idx, old_path in enumerate(files, start=start_index):
        temp_name = folder / f"__temp_rename_{idx:06d}.wav"
        old_path.rename(temp_name)
        temp_paths.append((temp_name, folder / f"{idx}.wav", old_path.name))

    for temp_path, final_path, orig_name in temp_paths:
        temp_path.rename(final_path)
        renamed_items.append((orig_name, final_path.name))

    return len(renamed_items), renamed_items


def batch_preprocess_audio_folder(
    input_folder,
    output_folder,
    target_sr=22050,
    trim_silence=True,
    top_db=20,
    normalize=True,
    update_tags=True,
    sequential_naming=False,
):
    """
    Preprocesses an entire folder of audio files and outputs clean 16-bit PCM WAV files.
    Matches the functionality of rasmurtech/Tacotron-2-Audio-Preprocessor.
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([
        f for f in in_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ])

    if not files:
        return 0, "No supported audio files found in input directory."

    processed_count = 0
    log_lines = []
    for idx, file_path in enumerate(files, start=1):
        if sequential_naming:
            out_filename = f"{idx}.wav"
            track_num = idx
            title_tag = str(idx)
        else:
            out_filename = f"{file_path.stem}.wav"
            track_num = idx
            title_tag = file_path.stem

        out_path = out_dir / out_filename
        try:
            preprocess_audio_file(
                file_path,
                out_path,
                target_sr=target_sr,
                trim_silence=trim_silence,
                top_db=top_db,
                normalize=normalize,
                update_tags=update_tags,
                title=title_tag,
                track_number=track_num,
            )
            processed_count += 1
            log_lines.append(f"Processed: {file_path.name} -> {out_filename}")
        except Exception as e:
            log_lines.append(f"Failed {file_path.name}: {str(e)}")

    summary = (
        f"Preprocessed {processed_count}/{len(files)} files into:\n{out_dir}\n"
        f"Settings: Sample Rate = {target_sr} Hz, Silence Trim = {trim_silence} (top_db={top_db}), "
        f"Normalize = {normalize}, Tags = {update_tags}\n\n" + "\n".join(log_lines[:20])
    )
    if len(log_lines) > 20:
        summary += f"\n... and {len(log_lines) - 20} more files."

    return processed_count, summary
