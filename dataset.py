from shutil import copyfile
import csv
import json
import os
import subprocess
import traceback
from collections import Counter
from pathlib import Path

import gradio as gr

from audio_processing import (
    batch_preprocess_audio_folder,
    batch_rename_wav_files,
    preprocess_audio_file,
    write_riff_info_chunk,
)
from errors import log_error
from language import clean_label
from profiles import *
from settings import *


def next_sample_path(emotion, profile_name=DEFAULT_PROFILE):
    emotion_dir = profile_dataset_dir(profile_name) / emotion
    emotion_dir.mkdir(parents=True, exist_ok=True)
    sample_number = len(list(emotion_dir.glob(f"{emotion}_*.wav"))) + 1
    return emotion_dir / f"{emotion}_{sample_number:03d}.wav"


def relative_to_project(path):
    return path.resolve().relative_to(PROJECT_DIR).as_posix()


def relative_to_profile(path, profile_name=DEFAULT_PROFILE):
    return path.resolve().relative_to(profile_base_dir(profile_name)).as_posix()


def append_metadata(audio_path, emotion, transcript="", profile_name=DEFAULT_PROFILE):
    metadata_path = profile_metadata_path(profile_name)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = metadata_path.exists()

    with metadata_path.open("a", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=["audio_path", "text", "emotion"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "audio_path": relative_to_profile(audio_path, profile_name),
                "text": transcript,
                "emotion": emotion,
            }
        )


def append_generation_history(
    input_text,
    translated_text,
    language,
    emotion,
    reference_audio,
    output_audio,
    profile_name=DEFAULT_PROFILE,
):
    outputs_dir = profile_outputs_dir(profile_name)
    history_path = profile_generation_history_path(profile_name)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    file_exists = history_path.exists()

    with history_path.open("a", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "input_text",
                "translated_text",
                "language",
                "emotion",
                "reference_audio",
                "output_audio",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "input_text": input_text,
                "translated_text": translated_text,
                "language": language,
                "emotion": emotion,
                "reference_audio": relative_to_profile(reference_audio, profile_name),
                "output_audio": relative_to_profile(output_audio, profile_name),
            }
        )


def read_metadata_rows(profile_name=DEFAULT_PROFILE):
    metadata_path = profile_metadata_path(profile_name)
    if not metadata_path.exists():
        return []

    with metadata_path.open("r", newline="", encoding="utf-8") as metadata_file:
        return list(csv.DictReader(metadata_file))


def write_metadata_rows(rows, profile_name=DEFAULT_PROFILE):
    metadata_path = profile_metadata_path(profile_name)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=["audio_path", "text", "emotion"],
        )
        writer.writeheader()
        writer.writerows(rows)


def metadata_path_choices(profile_name=DEFAULT_PROFILE):
    rows = read_metadata_rows(profile_name)
    return [row["audio_path"] for row in rows]


def audio_file_path(audio_input):
    if isinstance(audio_input, (list, tuple)):
        return audio_input[0] if audio_input else None
    if isinstance(audio_input, dict):
        return audio_input.get("path") or audio_input.get("name")
    return audio_input


def transcribe_audio(audio_path):
    worker_path = PROJECT_DIR / "transcribe_worker.py"
    result = subprocess.run(
        [sys.executable, str(worker_path), str(audio_path)],
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Transcription worker failed.")

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return payload.get("text", ""), payload.get("language", "unknown")


def save_audio_sample(
    audio_path,
    selected_emotion,
    custom_emotion,
    transcript,
    auto_transcribe,
    profile_name,
    preprocess_audio_flag=True,
):
    if not profile_is_active(profile_name):
        return profile_required_message()

    audio_path = audio_file_path(audio_path)

    if not audio_path:
        return "Upload or record audio first."

    try:
        emotion = clean_label(custom_emotion or selected_emotion)
        output_path = next_sample_path(emotion, profile_name)

        source_path = Path(audio_path)
        if not source_path.exists():
            return f"Audio file was not found:\n{source_path}"

        if preprocess_audio_flag:
            preprocess_audio_file(
                source_path,
                output_path,
                target_sr=24000,
                trim_silence=True,
                top_db=20,
                normalize=True,
                update_tags=True,
                title=output_path.stem,
            )
        elif source_path.suffix.lower() == ".wav":
            copyfile(source_path, output_path)
            write_riff_info_chunk(output_path, title=output_path.stem)
        else:
            try:
                from pydub import AudioSegment
            except ImportError:
                return "Install dependencies first:\npip install -r requirements.txt"

            audio = AudioSegment.from_file(source_path)
            audio = audio.set_channels(1).set_frame_rate(24000)
            audio.export(output_path, format="wav")
            write_riff_info_chunk(output_path, title=output_path.stem)

        transcript = transcript.strip()
        detected_language = "not detected"
        transcription_error = ""
        if auto_transcribe and not transcript:
            try:
                transcript, detected_language = transcribe_audio(output_path)
            except Exception:
                transcription_error = traceback.format_exc()
                log_error("transcribe_audio", transcription_error)

        append_metadata(output_path, emotion, transcript, profile_name)

        result = (
            f"Saved sample:\n{output_path}\n\n"
            f"Transcript:\n{transcript or '[empty]'}\n\n"
            f"Detected language: {detected_language}\n\n"
            f"Updated metadata:\n{profile_metadata_path(profile_name)}"
        )
        if transcription_error:
            result += (
                "\n\nAuto-transcription failed, but the audio sample was saved. "
                "You can edit the transcript in the Edit Dataset tab.\n\n"
                f"{transcription_error}"
            )
        return result
    except Exception:
        error_text = traceback.format_exc()
        log_error("save_audio_sample", error_text)
        return "Could not save audio sample:\n\n" + error_text


def save_audio_batch(
    audio_files,
    selected_emotion,
    custom_emotion,
    auto_transcribe,
    profile_name,
    preprocess_audio_flag=True,
):
    if not profile_is_active(profile_name):
        return profile_required_message()

    if not audio_files:
        return "Select one or more audio files first."

    emotion = clean_label(custom_emotion or selected_emotion)
    results = []

    for audio_file in audio_files:
        audio_path = audio_file_path(audio_file)
        if not audio_path:
            results.append("Skipped empty audio input.")
            continue

        try:
            source_path = Path(audio_path)
            output_path = next_sample_path(emotion, profile_name)

            if preprocess_audio_flag:
                preprocess_audio_file(
                    source_path,
                    output_path,
                    target_sr=24000,
                    trim_silence=True,
                    top_db=20,
                    normalize=True,
                    update_tags=True,
                    title=output_path.stem,
                )
            elif source_path.suffix.lower() == ".wav":
                copyfile(source_path, output_path)
                write_riff_info_chunk(output_path, title=output_path.stem)
            else:
                try:
                    from pydub import AudioSegment
                except ImportError:
                    raise RuntimeError("Install dependencies first: pip install -r requirements.txt")

                audio = AudioSegment.from_file(source_path)
                audio = audio.set_channels(1).set_frame_rate(24000)
                audio.export(output_path, format="wav")
                write_riff_info_chunk(output_path, title=output_path.stem)

            transcript = ""
            detected_language = "not detected"
            if auto_transcribe:
                try:
                    transcript, detected_language = transcribe_audio(output_path)
                except Exception:
                    error_text = traceback.format_exc()
                    log_error("batch_transcribe_audio", error_text)
                    results.append(
                        f"Saved {output_path.name}, but transcription failed. "
                        "Edit transcript later in Edit Dataset."
                    )

            append_metadata(output_path, emotion, transcript, profile_name)
            if transcript:
                results.append(
                    f"Saved {output_path.name} | {emotion} | {detected_language} | {transcript}"
                )
            elif not auto_transcribe:
                results.append(f"Saved {output_path.name} | {emotion} | transcript empty")

        except Exception:
            error_text = traceback.format_exc()
            log_error("save_audio_batch", error_text)
            results.append(f"Failed: {audio_path}\n{error_text}")

    return "\n\n".join(results)


def batch_upload_preview_rows(audio_files, page_index=0):
    audio_files = audio_files or []
    audio_paths = [audio_file_path(audio_file) for audio_file in audio_files]
    audio_paths = [path for path in audio_paths if path]

    if not audio_paths:
        return [], 0, "No batch files selected."

    max_page = max(0, (len(audio_paths) - 1) // BATCH_UPLOAD_PAGE_SIZE)
    page_index = max(0, min(int(page_index or 0), max_page))
    start = page_index * BATCH_UPLOAD_PAGE_SIZE
    end = min(start + BATCH_UPLOAD_PAGE_SIZE, len(audio_paths))
    rows = []

    for audio_path in audio_paths[start:end]:
        path = Path(audio_path)
        size = ""
        if path.exists():
            size = f"{path.stat().st_size / 1024:.1f} KB"
        rows.append([path.name, size, str(path)])

    return rows, page_index, f"Batch {start + 1}-{end} of {len(audio_paths)}"


def load_batch_upload_preview(audio_files):
    _, page_index, status = batch_upload_preview_rows(audio_files, 0)
    return page_index, status


def change_batch_upload_page(direction, audio_files, page_index):
    _, page_index, status = batch_upload_preview_rows(
        audio_files,
        int(page_index or 0) + direction,
    )
    return page_index, status


def import_emotion_folder(
    folder_path,
    selected_emotion,
    custom_emotion,
    auto_transcribe,
    profile_name,
    preprocess_audio_flag=True,
):
    if not profile_is_active(profile_name):
        return profile_required_message()

    folder = Path(folder_path.strip()) if folder_path else None
    if not folder or not folder.exists() or not folder.is_dir():
        return f"Folder not found or invalid: {folder_path}"

    emotion = clean_label(custom_emotion or selected_emotion or folder.name)
    audio_files = [
        path for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]

    if not audio_files:
        return f"No supported audio files found in:\n{folder}"

    results = []
    success_count = 0
    for source_path in audio_files:
        try:
            output_path = next_sample_path(emotion, profile_name)

            if preprocess_audio_flag:
                preprocess_audio_file(
                    source_path,
                    output_path,
                    target_sr=24000,
                    trim_silence=True,
                    top_db=20,
                    normalize=True,
                    update_tags=True,
                    title=output_path.stem,
                )
            elif source_path.suffix.lower() == ".wav":
                copyfile(source_path, output_path)
                write_riff_info_chunk(output_path, title=output_path.stem)
            else:
                try:
                    from pydub import AudioSegment
                except ImportError:
                    raise RuntimeError("Install dependencies first: pip install -r requirements.txt")

                audio = AudioSegment.from_file(source_path)
                audio = audio.set_channels(1).set_frame_rate(24000)
                audio.export(output_path, format="wav")
                write_riff_info_chunk(output_path, title=output_path.stem)

            transcript = ""
            detected_language = "not detected"
            if auto_transcribe:
                try:
                    transcript, detected_language = transcribe_audio(output_path)
                except Exception:
                    error_text = traceback.format_exc()
                    log_error("folder_import_transcribe_audio", error_text)
                    results.append(
                        f"Imported {source_path.name} as {output_path.name}, "
                        "but transcription failed."
                    )

            append_metadata(output_path, emotion, transcript, profile_name)
            success_count += 1
            if transcript:
                results.append(
                    f"✓ {source_path.name} -> {output_path.name} | {emotion} | {detected_language} | \"{transcript}\""
                )
            elif not auto_transcribe:
                results.append(f"✓ {source_path.name} -> {output_path.name} | {emotion} | transcript empty")

        except Exception:
            error_text = traceback.format_exc()
            log_error("import_emotion_folder", error_text)
            results.append(f"✗ Failed: {source_path.name}\n{error_text}")

    header = f"Successfully imported {success_count}/{len(audio_files)} audio files into profile '{display_profile_name(profile_name)}' under emotion '{emotion}'.\n\n"
    return header + "\n".join(results)


def reprocess_profile_dataset(profile_name=DEFAULT_PROFILE, target_sr=24000, trim_silence=True, normalize=True):
    """
    Cleans up all audio files in the selected profile's dataset:
    trims silence (20 dB) and normalizes audio volume in-place, and updates RIFF metadata tags.
    """
    if not profile_is_active(profile_name):
        return profile_required_message()

    profile = clean_profile_name(profile_name)
    rows = read_metadata_rows(profile)
    base_dir = profile_base_dir(profile)

    if not rows:
        return f"No audio samples found in profile '{display_profile_name(profile)}'."

    processed = 0
    errors = 0
    for idx, row in enumerate(rows, start=1):
        rel_path = row.get("audio_path")
        if not rel_path:
            continue
        full_path = base_dir / rel_path
        if not full_path.exists():
            continue

        try:
            preprocess_audio_file(
                full_path,
                full_path,
                target_sr=target_sr,
                trim_silence=trim_silence,
                top_db=20,
                normalize=normalize,
                update_tags=True,
                title=full_path.stem,
                track_number=idx,
                artist=profile,
            )
            processed += 1
        except Exception as e:
            errors += 1
            log_error("reprocess_profile_dataset", f"File: {full_path}\nError: {traceback.format_exc()}")

    return (
        f"Cleaned and preprocessed {processed} audio files for profile '{display_profile_name(profile)}'.\n"
        f"Settings: Sample Rate = {target_sr} Hz, Silence Trimming = {trim_silence} (20 dB threshold), "
        f"Volume Normalization = {normalize}, Tags = Embedded.\n"
        f"Errors: {errors}"
    )


def ui_batch_preprocess_folder(
    input_folder,
    output_folder,
    target_sr_choice="22050 Hz (Tacotron 2)",
    trim_silence=True,
    normalize=True,
    update_tags=True,
    sequential_naming=True,
):
    """
    UI action for standalone batch audio preprocessing tool.
    Matches rasmurtech/Tacotron-2-Audio-Preprocessor.
    """
    if not input_folder or not input_folder.strip():
        return "Enter an input folder path first."
    if not output_folder or not output_folder.strip():
        return "Enter an output folder path first."

    sr = 22050 if "22050" in target_sr_choice else (24000 if "24000" in target_sr_choice else 16000)
    count, summary = batch_preprocess_audio_folder(
        input_folder.strip(),
        output_folder.strip(),
        target_sr=sr,
        trim_silence=trim_silence,
        top_db=20,
        normalize=normalize,
        update_tags=update_tags,
        sequential_naming=sequential_naming,
    )
    return summary


def ui_batch_rename_folder(folder_path):
    """
    UI action for standalone batch WAV renaming tool.
    Matches rasmurtech/tacotron2-wav-file-renamer.
    """
    if not folder_path or not folder_path.strip():
        return "Enter a folder path first."
    try:
        count, renamed = batch_rename_wav_files(folder_path.strip())
        lines = [f"Renamed {count} WAV files sequentially in: {folder_path}\n"]
        lines.extend(f"{orig} -> {final}" for orig, final in renamed[:20])
        if len(renamed) > 20:
            lines.append(f"... and {len(renamed) - 20} more files.")
        return "\n".join(lines)
    except Exception as e:
        return f"Error renaming files: {str(e)}"


def load_metadata_preview(profile_name=DEFAULT_PROFILE):
    if not profile_is_active(profile_name):
        return profile_required_message(), []

    rows = read_metadata_rows(profile_name)
    if not rows:
        return "No metadata.csv file yet. Save an audio sample first.", []

    counts = Counter(row["emotion"] for row in rows)
    summary_lines = [
        "Dataset summary",
        "",
        f"Total samples: {len(rows)}",
        "",
        "Samples by emotion:",
    ]
    summary_lines.extend(f"- {emotion}: {count}" for emotion, count in sorted(counts.items()))

    table_rows = [
        [row["audio_path"], row["text"], row["emotion"]]
        for row in rows
    ]
    return "\n".join(summary_lines), table_rows


def load_metadata_for_selected_profile(profile_name):
    if profile_name == NO_PROFILE_SELECTION:
        return profile_required_message(), []
    return load_metadata_preview(clean_profile_name(profile_name))


def refresh_editor_choices(profile_name=DEFAULT_PROFILE):
    choices = metadata_path_choices(profile_name)
    value = choices[0] if choices else None
    transcript, emotion, audio = load_selected_sample(value, profile_name)
    return gr.update(choices=choices, value=value), transcript, emotion, audio


def sample_audio_output(selected_path, profile_name=DEFAULT_PROFILE):
    if not profile_is_active(profile_name):
        return None

    if not selected_path:
        return None

    audio_path = profile_base_dir(profile_name) / selected_path
    return str(audio_path) if audio_path.exists() else None


def load_selected_sample(selected_path, profile_name=DEFAULT_PROFILE):
    if not selected_path:
        return "", "", None

    for row in read_metadata_rows(profile_name):
        if row["audio_path"] == selected_path:
            return row.get("text", ""), row.get("emotion", ""), sample_audio_output(selected_path, profile_name)

    return "", "", None


def update_selected_sample(selected_path, new_text, new_emotion, profile_name=DEFAULT_PROFILE):
    if not selected_path:
        return "Select a sample first."

    rows = read_metadata_rows(profile_name)
    updated = False
    for row in rows:
        if row["audio_path"] == selected_path:
            row["text"] = new_text.strip()
            row["emotion"] = clean_label(new_emotion)
            updated = True
            break

    if not updated:
        return f"Sample not found in metadata:\n{selected_path}"

    write_metadata_rows(rows, profile_name)
    return f"Updated metadata row:\n{selected_path}"


def delete_selected_sample(selected_path, profile_name=DEFAULT_PROFILE):
    if not selected_path:
        return "Select a sample first."

    rows = read_metadata_rows(profile_name)
    remaining_rows = [row for row in rows if row["audio_path"] != selected_path]

    if len(remaining_rows) == len(rows):
        return f"Sample not found in metadata:\n{selected_path}"

    audio_path = profile_base_dir(profile_name) / selected_path
    if audio_path.exists():
        os.remove(audio_path)

    write_metadata_rows(remaining_rows, profile_name)
    return f"Deleted sample:\n{selected_path}"


def delete_selected_sample_and_refresh(selected_path, profile_name=DEFAULT_PROFILE):
    result = delete_selected_sample(selected_path, profile_name)
    choices = metadata_path_choices(profile_name)
    value = choices[0] if choices else None
    transcript, emotion, audio = load_selected_sample(value, profile_name)
    return result, gr.update(choices=choices, value=value), transcript, emotion, audio


def emotion_editor_choices(profile_name=DEFAULT_PROFILE):
    choices = emotion_choices_from_dataset(profile_name)
    return choices or DEFAULT_EMOTIONS


def rows_for_emotion(emotion_filter, profile_name=DEFAULT_PROFILE):
    if not profile_is_active(profile_name):
        return []

    rows = []
    seen_audio_paths = set()
    for row in read_metadata_rows(profile_name):
        if row.get("emotion") == emotion_filter:
            audio_path_value = row.get("audio_path", "")
            if audio_path_value in seen_audio_paths:
                continue
            audio_path = profile_base_dir(profile_name) / row.get("audio_path", "")
            if audio_path.exists():
                seen_audio_paths.add(audio_path_value)
                rows.append(
                    [
                        audio_path_value,
                        row.get("text", ""),
                        row.get("emotion", ""),
                    ]
                )
    return rows


def transcript_table_update(table_rows):
    row_count = len(table_rows)
    return gr.update(
        value=table_rows,
        row_count=row_count,
        column_count=3,
    )


def audio_page_choices(table_rows, page_index=0):
    audio_paths = [row[0] for row in table_rows]
    if not audio_paths:
        return [], None, 0, "No audio samples in this mood."

    max_page = max(0, (len(audio_paths) - 1) // EDITOR_PAGE_SIZE)
    page_index = max(0, min(int(page_index or 0), max_page))
    start = page_index * EDITOR_PAGE_SIZE
    end = min(start + EDITOR_PAGE_SIZE, len(audio_paths))
    page_audio_paths = audio_paths[start:end]
    status = f"Batch {start + 1}-{end} of {len(audio_paths)}"
    return page_audio_paths, page_audio_paths[0], page_index, status


def audio_page_update(table_rows, page_index=0):
    choices, selected_audio, page_index, status = audio_page_choices(table_rows, page_index)
    return gr.update(choices=choices, value=selected_audio), page_index, status


def load_emotion_editor(emotion_filter, profile_name=DEFAULT_PROFILE):
    if not profile_is_active(profile_name):
        return (
            gr.update(choices=DEFAULT_EMOTIONS, value="neutral"),
            gr.update(choices=[], value=None),
            None,
            transcript_table_update([]),
            profile_required_message(),
            0,
            "No active profile.",
        )

    emotions = emotion_editor_choices(profile_name)
    selected_emotion = emotion_filter if emotion_filter in emotions else emotions[0]
    table_rows = rows_for_emotion(selected_emotion, profile_name)
    audio_list_update, page_index, page_status = audio_page_update(table_rows, 0)
    selected_audio = audio_list_update.get("value")
    status = (
        f"{selected_emotion}: {len(table_rows)} sample(s) | {page_status}"
        if table_rows
        else f"No saved audio found for emotion: {selected_emotion}"
    )
    return (
        gr.update(choices=emotions, value=selected_emotion),
        audio_list_update,
        sample_audio_output(selected_audio, profile_name),
        transcript_table_update(table_rows),
        status,
        page_index,
        page_status,
    )


def change_editor_audio_page(direction, emotion_filter, page_index, profile_name=DEFAULT_PROFILE):
    if not profile_is_active(profile_name):
        return gr.update(choices=[], value=None), None, 0, "No active profile."

    table_rows = rows_for_emotion(emotion_filter, profile_name)
    next_page_index = int(page_index or 0) + direction
    audio_list_update, next_page_index, page_status = audio_page_update(table_rows, next_page_index)
    selected_audio = audio_list_update.get("value")
    status = (
        f"{emotion_filter}: {len(table_rows)} sample(s) | {page_status}"
        if table_rows
        else f"No saved audio found for emotion: {emotion_filter}"
    )
    return (
        audio_list_update,
        sample_audio_output(selected_audio, profile_name),
        next_page_index,
        status,
    )


def normalize_editor_table(table_data):
    if table_data is None:
        return []
    if hasattr(table_data, "to_dict"):
        records = table_data.to_dict("records")
        return [
            [
                record.get("audio_path", ""),
                record.get("text", ""),
                record.get("emotion", ""),
            ]
            for record in records
        ]
    if isinstance(table_data, dict):
        data = table_data.get("data", [])
        return [list(row) for row in data]
    return [list(row) for row in table_data]


def save_emotion_transcript_table(emotion_filter, table_data, profile_name=DEFAULT_PROFILE):
    if not profile_is_active(profile_name):
        return profile_required_message(), transcript_table_update([])

    edited_rows = normalize_editor_table(table_data)
    if not edited_rows:
        return "No transcript rows to save.", transcript_table_update([])

    rows = read_metadata_rows(profile_name)
    rows_by_path = {row.get("audio_path", ""): row for row in rows}
    updated_count = 0
    skipped_count = 0

    for edited in edited_rows:
        if not edited:
            continue
        audio_path = str(edited[0]).strip()
        if not audio_path or audio_path not in rows_by_path:
            skipped_count += 1
            continue

        absolute_audio_path = profile_base_dir(profile_name) / audio_path
        if not absolute_audio_path.exists():
            skipped_count += 1
            continue

        rows_by_path[audio_path]["text"] = str(edited[1]).strip() if len(edited) > 1 else ""
        if len(edited) > 2 and str(edited[2]).strip():
            rows_by_path[audio_path]["emotion"] = clean_label(str(edited[2]))
        updated_count += 1

    valid_rows = []
    seen_audio_paths = set()
    for row in rows:
        audio_path = row.get("audio_path")
        if (
            audio_path
            and audio_path not in seen_audio_paths
            and (profile_base_dir(profile_name) / audio_path).exists()
        ):
            valid_rows.append(row)
            seen_audio_paths.add(audio_path)
    write_metadata_rows(valid_rows, profile_name)
    refreshed_rows = rows_for_emotion(emotion_filter, profile_name)
    return (
        f"Saved {updated_count} row(s). Skipped {skipped_count} missing/deleted row(s).",
        transcript_table_update(refreshed_rows),
    )


def cleanup_duplicate_metadata(profile_name=DEFAULT_PROFILE):
    if not profile_is_active(profile_name):
        return (
            profile_required_message(),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            None,
            transcript_table_update([]),
            0,
            "No active profile.",
        )

    rows = read_metadata_rows(profile_name)
    cleaned_rows = []
    seen_audio_paths = set()
    duplicate_count = 0
    missing_count = 0

    for row in rows:
        audio_path = row.get("audio_path", "")
        if not audio_path or not (profile_base_dir(profile_name) / audio_path).exists():
            missing_count += 1
            continue
        if audio_path in seen_audio_paths:
            duplicate_count += 1
            continue
        cleaned_rows.append(row)
        seen_audio_paths.add(audio_path)

    write_metadata_rows(cleaned_rows, profile_name)
    emotions = emotion_editor_choices(profile_name)
    selected_emotion = emotions[0] if emotions else "neutral"
    table_rows = rows_for_emotion(selected_emotion, profile_name)
    audio_list_update, page_index, page_status = audio_page_update(table_rows, 0)
    selected_audio = audio_list_update.get("value")
    status = (
        f"Cleaned metadata: removed {duplicate_count} duplicate row(s) "
        f"and {missing_count} missing-audio row(s). {page_status}"
    )
    return (
        status,
        gr.update(choices=emotions, value=selected_emotion),
        audio_list_update,
        sample_audio_output(selected_audio, profile_name),
        transcript_table_update(table_rows),
        page_index,
        page_status,
    )


def move_editor_sample(selected_path, new_emotion, emotion_filter, profile_name=DEFAULT_PROFILE):
    if not profile_is_active(profile_name):
        return profile_required_message(), gr.update(choices=[], value=None), None, transcript_table_update([]), 0, "No active profile."

    if not selected_path:
        return "Select an audio sample first.", gr.update(), None, transcript_table_update([]), 0, "No audio selected."

    rows = read_metadata_rows(profile_name)
    updated = False
    target_emotion = clean_label(new_emotion)
    for row in rows:
        if row.get("audio_path") == selected_path:
            row["emotion"] = target_emotion
            updated = True
            break

    if not updated:
        return (
            f"Sample not found in metadata:\n{selected_path}",
            gr.update(),
            None,
            transcript_table_update(rows_for_emotion(emotion_filter, profile_name)),
            0,
            "No matching audio sample.",
        )

    write_metadata_rows(rows, profile_name)
    table_rows = rows_for_emotion(emotion_filter, profile_name)
    audio_list_update, page_index, page_status = audio_page_update(table_rows, 0)
    next_audio = audio_list_update.get("value")
    return (
        f"Moved sample to emotion: {target_emotion}",
        audio_list_update,
        sample_audio_output(next_audio, profile_name),
        transcript_table_update(table_rows),
        page_index,
        page_status,
    )


def delete_editor_sample(selected_path, emotion_filter, profile_name=DEFAULT_PROFILE):
    if not profile_is_active(profile_name):
        return profile_required_message(), gr.update(choices=[], value=None), None, transcript_table_update([]), 0, "No active profile."

    result = delete_selected_sample(selected_path, profile_name)
    table_rows = rows_for_emotion(emotion_filter, profile_name)
    audio_list_update, page_index, page_status = audio_page_update(table_rows, 0)
    next_audio = audio_list_update.get("value")
    return (
        result,
        audio_list_update,
        sample_audio_output(next_audio, profile_name),
        transcript_table_update(table_rows),
        page_index,
        page_status,
    )


def emotion_choices_from_dataset(profile_name=DEFAULT_PROFILE):
    if not profile_is_active(profile_name):
        return []
    emotions = sorted({row["emotion"] for row in read_metadata_rows(profile_name) if row.get("emotion")})
    return emotions or DEFAULT_EMOTIONS


def reference_audio_for_emotion(emotion, profile_name=DEFAULT_PROFILE):
    references = reference_audios_for_emotion(emotion, profile_name)
    return references[0] if references else None


def reference_audios_for_emotion(emotion, profile_name=DEFAULT_PROFILE, limit=None):
    references = []
    rows = read_metadata_rows(profile_name)
    for row in rows:
        if row.get("emotion") == emotion:
            audio_path = profile_base_dir(profile_name) / row["audio_path"]
            if audio_path.exists():
                references.append(audio_path)
                if limit is not None and len(references) >= limit:
                    return references

    emotion_dir = profile_dataset_dir(profile_name) / clean_label(emotion)
    if emotion_dir.exists():
        audio_files = sorted(emotion_dir.glob("*.wav"))
        for audio_path in audio_files:
            if audio_path not in references:
                references.append(audio_path)
                if limit is not None and len(references) >= limit:
                    return references

    return references

