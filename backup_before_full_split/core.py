from shutil import copyfile
import csv
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from threading import Event
import platform

import gradio as gr

from project_config import *

WHISPER_MODEL = None
XTTS_MODEL = None
XTTS_MODEL_SOURCE = None
TRANSLATION_TOKENIZER = None
TRANSLATION_MODEL = None

def japanese_support_available():
    return importlib.util.find_spec("cutlet") is not None


def available_language_names():
    languages = list(LANGUAGE_CODES.keys())
    if not japanese_support_available():
        languages = [language for language in languages if language != "Japanese"]
    return languages


def language_unavailable_message(target_language):
    if target_language == "Japanese" and not japanese_support_available():
        return (
            "Japanese speech generation needs the optional XTTS text-cleaner package 'cutlet'.\n"
            "For now, choose another target language, or install cutlet in the voiceapp310 environment."
        )
    return ""


def clean_label(label):
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in label.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "neutral"


def clean_profile_name(profile_name):
    if profile_name == NO_PROFILE_SELECTION:
        return DEFAULT_PROFILE
    cleaned = clean_label(profile_name)
    return cleaned or DEFAULT_PROFILE


def display_profile_name(profile_name):
    profile = clean_profile_name(profile_name)
    return PROFILE_DISPLAY_NAMES.get(profile, profile)


def profile_is_active(profile_name):
    return bool(profile_name) and profile_name != NO_PROFILE_SELECTION


def profile_required_message():
    return "Select a saved profile and click Use Selected Profile first."


def profile_base_dir(profile_name):
    profile = clean_profile_name(profile_name)
    if profile == DEFAULT_PROFILE:
        return PROJECT_DIR
    return USERS_DIR / profile


def profile_dataset_dir(profile_name):
    return profile_base_dir(profile_name) / "dataset"


def profile_metadata_path(profile_name):
    return profile_dataset_dir(profile_name) / "metadata.csv"


def profile_outputs_dir(profile_name):
    return profile_base_dir(profile_name) / "outputs"


def profile_models_dir(profile_name):
    return profile_base_dir(profile_name) / "models"


def profile_generation_history_path(profile_name):
    return profile_outputs_dir(profile_name) / "generation_history.csv"


def list_profiles():
    profiles = []
    if USERS_DIR.exists():
        profiles.extend(
            sorted(
                display_profile_name(path.name)
                for path in USERS_DIR.iterdir()
                if path.is_dir() and path.name not in HIDDEN_PROFILE_NAMES
            )
        )
    return profiles or [DEFAULT_PROFILE]


def profile_selector_choices():
    return [NO_PROFILE_SELECTION] + list_profiles()


def create_profile(profile_name):
    typed_profile = (profile_name or "").strip()
    if not typed_profile:
        return gr.update(choices=profile_selector_choices()), "Type a new profile name first."

    profile = clean_profile_name(typed_profile)
    if profile in HIDDEN_PROFILE_NAMES:
        return gr.update(choices=profile_selector_choices()), f"'{profile}' is reserved. Choose a speaker name instead."

    if profile == DEFAULT_PROFILE:
        return gr.update(choices=profile_selector_choices(), value=DEFAULT_PROFILE), "Using default profile."

    base_dir = profile_base_dir(profile)
    (base_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (base_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (base_dir / "models").mkdir(parents=True, exist_ok=True)

    display_name = display_profile_name(profile)
    return gr.update(choices=profile_selector_choices(), value=display_name), f"Created/selected profile: {display_name}"


def create_profile_and_status(profile_name):
    profile_update, status = create_profile(profile_name)
    return profile_update, status, status


def select_profile(profile_name):
    if profile_name == NO_PROFILE_SELECTION:
        return "No profile selected. Choose a saved speaker profile, then click Use Selected Profile."
    profile = clean_profile_name(profile_name)
    return f"Ready to use profile: {display_profile_name(profile)}\nClick Use Selected Profile to load this speaker for dataset editing and speech generation."


def use_selected_profile(profile_name):
    if profile_name == NO_PROFILE_SELECTION:
        return "No profile selected. Please select a saved speaker profile first.", NO_PROFILE_SELECTION
    profile = clean_profile_name(profile_name)
    display_name = display_profile_name(profile)
    base_dir = profile_base_dir(profile)
    metadata_path = profile_metadata_path(profile)
    checkpoint_path, config_path = finetuned_xtts_model_files(profile)
    model_dir = checkpoint_path.parent if checkpoint_path else next(finetuned_xtts_model_dirs(profile))
    expected_model_files = [
        "model.pth",
        "config.json",
        "config.runtime.json",
        "vocab.json",
        "mel_stats.pth",
        "dvae.pth",
    ]
    model_files_ready = model_dir.exists() and all(
        (model_dir / file_name).exists() for file_name in expected_model_files
    )
    trained_model_ready = bool(checkpoint_path and config_path and model_files_ready)
    rows = read_metadata_rows(profile)
    counts = Counter(row.get("emotion", "") for row in rows)
    lines = [
        f"Profile in use: {display_name}",
        "",
        f"Dataset records: {len(rows)}",
        f"Metadata file: {metadata_path}",
        f"Profile folder: {base_dir}",
        "",
        "Fine-tuned XTTS model:",
        f"Model folder: {model_dir}",
    ]
    if trained_model_ready:
        lines.extend(
            [
                "Status: ready to use",
                f"Runtime model: {checkpoint_path}",
                f"Runtime config: {config_path}",
            ]
        )
    elif model_dir.exists():
        missing_files = [
            file_name for file_name in expected_model_files
            if not (model_dir / file_name).exists()
        ]
        lines.extend(
            [
                "Status: model folder found, but not ready",
                "Missing files: " + ", ".join(missing_files),
                "Speech generation will fall back to the stock XTTS-v2 model until this is complete.",
            ]
        )
    else:
        lines.extend(
            [
                "Status: no fine-tuned model found for this profile",
                "Speech generation can still use stock XTTS-v2 with this profile's reference clips.",
            ]
        )

    lines.extend(
        [
            "",
        "Emotion coverage:",
        ]
    )
    if counts:
        lines.extend(f"- {emotion}: {count}" for emotion, count in sorted(counts.items()))
    else:
        lines.append("- no emotion-labeled samples yet")
    return "\n".join(lines), profile


def refresh_profiles():
    profiles = list_profiles()
    return gr.update(choices=profiles, value=profiles[0])


def refresh_profile_selector():
    profiles = profile_selector_choices()
    value = NO_PROFILE_SELECTION
    return gr.update(choices=profiles, value=value), select_profile(value)


def profile_dashboard(profile_name=DEFAULT_PROFILE):
    profile = clean_profile_name(profile_name)
    base_dir = profile_base_dir(profile)
    dataset_dir = profile_dataset_dir(profile)
    outputs_dir = profile_outputs_dir(profile)
    models_dir = base_dir / "models"
    training_data_dir = base_dir / "training_data"
    metadata_path = profile_metadata_path(profile)
    history_path = profile_generation_history_path(profile)

    rows = read_metadata_rows(profile)
    counts = Counter(row.get("emotion", "") for row in rows)
    missing_transcripts = sum(1 for row in rows if not row.get("text", "").strip())
    wav_count = len(list(dataset_dir.rglob("*.wav"))) if dataset_dir.exists() else 0
    output_count = len(list(outputs_dir.glob("*.wav"))) if outputs_dir.exists() else 0
    model_files = list(models_dir.rglob("*")) if models_dir.exists() else []
    model_file_count = sum(1 for path in model_files if path.is_file())

    lines = [
        f"Profile: {display_profile_name(profile)}",
        "",
        "Folders",
        f"Base: {base_dir}",
        f"Dataset: {dataset_dir}",
        f"Outputs: {outputs_dir}",
        f"Models: {models_dir}",
        f"Training data: {training_data_dir}",
        "",
        "Dataset",
        f"Metadata: {metadata_path} ({'exists' if metadata_path.exists() else 'missing'})",
        f"Metadata rows: {len(rows)}",
        f"WAV files: {wav_count}",
        f"Missing transcripts: {missing_transcripts}",
        "",
        "Samples by emotion:",
    ]
    if counts:
        lines.extend(f"- {emotion or '[empty]'}: {count}" for emotion, count in sorted(counts.items()))
    else:
        lines.append("- no samples yet")

    lines.extend(
        [
            "",
            "Outputs",
            f"Generated WAV files: {output_count}",
            f"Generation history: {history_path} ({'exists' if history_path.exists() else 'missing'})",
            "",
            "Training / model storage",
            f"train.csv: {training_data_dir / 'train.csv'} ({'exists' if (training_data_dir / 'train.csv').exists() else 'missing'})",
            f"val.csv: {training_data_dir / 'val.csv'} ({'exists' if (training_data_dir / 'val.csv').exists() else 'missing'})",
            f"Stored model files: {model_file_count}",
        ]
    )

    emotion_table = [[emotion or "[empty]", count] for emotion, count in sorted(counts.items())]
    return "\n".join(lines), emotion_table


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


def log_error(context, error_text):
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n--- {context} ---\n")
        log_file.write(error_text)
        log_file.write("\n")


def log_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        return

    error_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    log_error("uncaught_exception", error_text)


sys_excepthook_original = None
try:
    import sys

    sys_excepthook_original = sys.excepthook
    sys.excepthook = log_uncaught_exception
except Exception:
    pass


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


def save_audio_sample(audio_path, selected_emotion, custom_emotion, transcript, auto_transcribe, profile_name):
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

        if source_path.suffix.lower() == ".wav":
            copyfile(source_path, output_path)
        else:
            try:
                from pydub import AudioSegment
            except ImportError:
                return "Install dependencies first:\npip install -r requirements.txt"

            audio = AudioSegment.from_file(source_path)
            audio = audio.set_channels(1).set_frame_rate(24000)
            audio.export(output_path, format="wav")

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


def save_audio_batch(audio_files, selected_emotion, custom_emotion, auto_transcribe, profile_name):
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

            if source_path.suffix.lower() == ".wav":
                copyfile(source_path, output_path)
            else:
                try:
                    from pydub import AudioSegment
                except ImportError:
                    raise RuntimeError("Install dependencies first: pip install -r requirements.txt")

                audio = AudioSegment.from_file(source_path)
                audio = audio.set_channels(1).set_frame_rate(24000)
                audio.export(output_path, format="wav")

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


def import_emotion_folder(folder_path, selected_emotion, custom_emotion, auto_transcribe, profile_name):
    folder = Path(folder_path.strip()) if folder_path else None
    if not folder or not folder.exists() or not folder.is_dir():
        return "Enter a valid folder path."

    emotion = clean_label(custom_emotion or selected_emotion or folder.name)
    audio_files = [
        path for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]

    if not audio_files:
        return f"No supported audio files found in:\n{folder}"

    results = []
    for source_path in audio_files:
        try:
            output_path = next_sample_path(emotion, profile_name)

            if source_path.suffix.lower() == ".wav":
                copyfile(source_path, output_path)
            else:
                try:
                    from pydub import AudioSegment
                except ImportError:
                    raise RuntimeError("Install dependencies first: pip install -r requirements.txt")

                audio = AudioSegment.from_file(source_path)
                audio = audio.set_channels(1).set_frame_rate(24000)
                audio.export(output_path, format="wav")

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
            if transcript:
                results.append(
                    f"Imported {source_path.name} -> {output_path.name} | "
                    f"{emotion} | {detected_language} | {transcript}"
                )
            elif not auto_transcribe:
                results.append(
                    f"Imported {source_path.name} -> {output_path.name} | "
                    f"{emotion} | transcript empty"
                )

        except Exception:
            error_text = traceback.format_exc()
            log_error("import_emotion_folder", error_text)
            results.append(f"Failed: {source_path}\n{error_text}")

    return "\n\n".join(results)


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


def preview_project(text, target_language, emotion):
    if not text.strip():
        return "Enter English text first."

    return (
        "Local app is working.\n\n"
        f"Input text: {text}\n"
        f"Target language: {target_language}\n"
        f"Emotion: {emotion}\n\n"
        "Next step: connect audio upload, transcription, translation, and XTTS-v2."
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


def prepare_finetuned_xtts_model(model_dir, checkpoint_path, config_path):
    runtime_model_path = model_dir / "model.pth"
    if not runtime_model_path.exists():
        try:
            os.link(checkpoint_path, runtime_model_path)
        except OSError:
            copyfile(checkpoint_path, runtime_model_path)

    runtime_config_path = model_dir / "config.runtime.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_args = config.setdefault("model_args", {})
        model_args["xtts_checkpoint"] = str(runtime_model_path)
        model_args["tokenizer_file"] = str(model_dir / "vocab.json")
        model_args["mel_norm_file"] = str(model_dir / "mel_stats.pth")
        model_args["dvae_checkpoint"] = str(model_dir / "dvae.pth")
        config["model_dir"] = str(model_dir)
        runtime_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return runtime_model_path, runtime_config_path
    except Exception:
        log_error("prepare_finetuned_xtts_model", traceback.format_exc())
        return runtime_model_path, config_path


def finetuned_xtts_model_dirs(profile_name):
    profile = clean_label(profile_name)
    folder_names = [profile, *PROFILE_MODEL_FOLDER_ALIASES.get(profile, [])]
    seen = set()
    for folder_name in folder_names:
        if folder_name in seen:
            continue
        seen.add(folder_name)
        yield profile_models_dir(profile) / f"{folder_name}_xtts_finetuned"


def finetuned_xtts_model_files(profile_name):
    for model_dir in finetuned_xtts_model_dirs(profile_name):
        config_path = model_dir / "config.json"
        checkpoints = sorted(model_dir.glob("best_model*.pth"))
        if not checkpoints:
            checkpoints = sorted(model_dir.glob("model*.pth"))
        if config_path.exists() and checkpoints:
            return prepare_finetuned_xtts_model(model_dir, checkpoints[-1], config_path)
    return None, None


def load_xtts_model(profile_name=DEFAULT_PROFILE):
    global XTTS_MODEL, XTTS_MODEL_SOURCE
    checkpoint_path, config_path = finetuned_xtts_model_files(profile_name)
    model_source = str(checkpoint_path) if checkpoint_path else "tts_models/multilingual/multi-dataset/xtts_v2"

    if XTTS_MODEL is None or XTTS_MODEL_SOURCE != model_source:
        import torch
        from TTS.api import TTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if checkpoint_path and config_path:
            try:
                XTTS_MODEL = TTS(model_path=str(checkpoint_path.parent), config_path=str(config_path)).to(device)
                XTTS_MODEL_SOURCE = model_source
                return XTTS_MODEL
            except Exception:
                log_error("load_xtts_model", traceback.format_exc())

        XTTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        XTTS_MODEL_SOURCE = "tts_models/multilingual/multi-dataset/xtts_v2"
    return XTTS_MODEL


def load_translation_model():
    global TRANSLATION_TOKENIZER, TRANSLATION_MODEL

    if TRANSLATION_TOKENIZER is None or TRANSLATION_MODEL is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "facebook/nllb-200-distilled-600M"
        # The translation model is already cached on the local machine. Loading it
        # offline prevents failed proxy/network calls from blocking speech generation.
        TRANSLATION_TOKENIZER = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        TRANSLATION_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        TRANSLATION_MODEL.to("cpu")

    return TRANSLATION_TOKENIZER, TRANSLATION_MODEL


def translate_text_if_needed(text, target_language, enable_translation):
    if not enable_translation or target_language == "English":
        return text.strip()

    import torch

    tokenizer, model = load_translation_model()
    tokenizer.src_lang = NLLB_LANGUAGE_CODES["English"]
    target_code = NLLB_LANGUAGE_CODES[target_language]

    inputs = tokenizer(text.strip(), return_tensors="pt", truncation=True, max_length=256)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
        )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


def generate_xtts_speech(text, target_language, emotion, enable_translation, profile_name):
    if not profile_is_active(profile_name):
        return "", None, profile_required_message()

    if not text.strip():
        return "", None, "Enter text first."

    reference_audios = reference_audios_for_emotion(emotion, profile_name)
    if not reference_audios:
        return "", None, f"No reference WAV found for emotion: {emotion}"
    primary_reference_audio = reference_audios[0]
    speaker_wav = [str(path) for path in reference_audios]

    language_code = LANGUAGE_CODES.get(target_language)
    if not language_code:
        return "", None, f"Unsupported target language: {target_language}"

    unavailable_message = language_unavailable_message(target_language)
    if unavailable_message:
        return "", None, unavailable_message

    try:
        text_for_speech = translate_text_if_needed(text, target_language, enable_translation)
        model = load_xtts_model(profile_name)
        outputs_dir = profile_outputs_dir(profile_name)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = outputs_dir / f"xtts_{clean_label(emotion)}_{len(list(outputs_dir.glob('xtts_*.wav'))) + 1:03d}.wav"

        model.tts_to_file(
            text=text_for_speech,
            speaker_wav=speaker_wav,
            language=language_code,
            file_path=str(output_path),
        )

        append_generation_history(
            input_text=text.strip(),
            translated_text=text_for_speech,
            language=target_language,
            emotion=emotion,
            reference_audio=primary_reference_audio,
            output_audio=output_path,
            profile_name=profile_name,
        )

        return text_for_speech, str(output_path), (
            f"Generated speech:\n{output_path}\n\n"
            f"Model source:\n{XTTS_MODEL_SOURCE}\n\n"
            f"Reference audio:\n{primary_reference_audio}\n\n"
            f"Reference clips used: {len(reference_audios)}\n\n"
            f"Text spoken:\n{text_for_speech}\n\n"
            f"Language: {target_language} ({language_code})\n"
            f"Emotion reference: {emotion}"
        )
    except Exception:
        error_text = traceback.format_exc()
        log_error("generate_xtts_speech", error_text)
        return "", None, "XTTS generation failed:\n\n" + error_text


def run_diagnostics():
    lines = []
    lines.append("Runtime")
    lines.append(f"Python: {platform.python_version()}")
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Project dir: {PROJECT_DIR}")
    lines.append(f"HF_HOME: {os.environ.get('HF_HOME', '')}")
    lines.append(f"TORCH_HOME: {os.environ.get('TORCH_HOME', '')}")
    lines.append(f"TTS_HOME: {os.environ.get('TTS_HOME', '')}")
    lines.append("")

    lines.append("Key paths")
    for path in [
        DATASET_DIR,
        METADATA_PATH,
        OUTPUTS_DIR,
        GENERATION_HISTORY_PATH,
        ERROR_LOG_PATH,
    ]:
        lines.append(f"{path}: {'exists' if path.exists() else 'missing'}")
    lines.append("")

    lines.append("FFmpeg")
    try:
        from pydub.utils import which

        lines.append(f"ffmpeg: {which('ffmpeg')}")
        lines.append(f"ffprobe: {which('ffprobe')}")
    except Exception:
        lines.append(traceback.format_exc())
    lines.append("")

    lines.append("CUDA")
    try:
        import torch

        lines.append(f"torch: {torch.__version__}")
        lines.append(f"cuda available: {torch.cuda.is_available()}")
        lines.append(f"cuda devices: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            lines.append(f"cuda device 0: {torch.cuda.get_device_name(0)}")
    except Exception:
        lines.append(traceback.format_exc())
    lines.append("")

    lines.append("Packages")
    for module_name in ["gradio", "TTS", "transformers", "faster_whisper", "numpy"]:
        try:
            module = __import__(module_name)
            lines.append(f"{module_name}: {getattr(module, '__version__', 'installed')}")
        except Exception as exc:
            lines.append(f"{module_name}: error: {exc}")

    return "\n".join(lines)


def read_error_log():
    if not ERROR_LOG_PATH.exists():
        return "No app errors logged yet."

    text = ERROR_LOG_PATH.read_text(encoding="utf-8")
    return text[-8000:] if len(text) > 8000 else text


def clear_error_log():
    ERROR_LOG_PATH.write_text("", encoding="utf-8")
    return "Error log cleared."


def dataset_health_report(profile_name=DEFAULT_PROFILE):
    rows = read_metadata_rows(profile_name)
    if not rows:
        return "No dataset records found. Save audio samples first.", []

    counts = Counter(row.get("emotion", "") for row in rows)
    missing_transcripts = [row for row in rows if not row.get("text", "").strip()]
    missing_audio = [
        row for row in rows
        if row.get("audio_path") and not (profile_base_dir(profile_name) / row["audio_path"]).exists()
    ]

    lines = [
        "Training dataset health",
        "",
        f"Total samples: {len(rows)}",
        f"Missing transcripts: {len(missing_transcripts)}",
        f"Missing audio files: {len(missing_audio)}",
        "",
        "Samples by emotion:",
    ]
    lines.extend(f"- {emotion or '[empty]'}: {count}" for emotion, count in sorted(counts.items()))
    lines.extend(
        [
            "",
            "Suggested minimum for early fine-tuning experiments:",
            "- 30-50 clean samples per emotion",
            "- each sample should have a correct transcript",
            "- audio should be short, clear, and consistently labeled",
            "",
            "Status:",
        ]
    )

    if missing_audio:
        lines.append("Not ready: some metadata rows point to missing audio files.")
    elif missing_transcripts:
        lines.append("Not ready: some samples are missing transcripts.")
    elif any(count < 5 for count in counts.values()):
        lines.append("Prototype-ready only: add more samples per emotion before real fine-tuning.")
    else:
        lines.append("Ready for a small training-data export.")

    issue_rows = [
        [row.get("audio_path", ""), row.get("text", ""), row.get("emotion", "")]
        for row in missing_transcripts + missing_audio
    ]

    return "\n".join(lines), issue_rows


def dataset_health_summary_only(profile_name=DEFAULT_PROFILE):
    if not profile_is_active(profile_name):
        return profile_required_message()

    summary, _ = dataset_health_report(profile_name)
    return summary


def export_training_split(profile_name=DEFAULT_PROFILE, validation_ratio=0.2):
    rows = [
        row for row in read_metadata_rows(profile_name)
        if row.get("audio_path")
        and row.get("text", "").strip()
        and row.get("emotion", "").strip()
        and (profile_base_dir(profile_name) / row["audio_path"]).exists()
    ]

    if len(rows) < 2:
        return "Need at least 2 complete samples with audio, transcript, and emotion.", [], []

    random.seed(42)
    rows_by_emotion = {}
    for row in rows:
        rows_by_emotion.setdefault(row["emotion"], []).append(row)

    train_rows = []
    val_rows = []
    for emotion_rows in rows_by_emotion.values():
        shuffled = emotion_rows[:]
        random.shuffle(shuffled)
        val_count = 1 if len(shuffled) > 1 else 0
        val_count = max(val_count, int(len(shuffled) * validation_ratio))
        val_rows.extend(shuffled[:val_count])
        train_rows.extend(shuffled[val_count:])

    training_data_dir = profile_base_dir(profile_name) / "training_data"
    training_data_dir.mkdir(parents=True, exist_ok=True)
    train_path = training_data_dir / "train.csv"
    val_path = training_data_dir / "val.csv"

    for path, split_rows in [(train_path, train_rows), (val_path, val_rows)]:
        with path.open("w", newline="", encoding="utf-8") as split_file:
            writer = csv.DictWriter(split_file, fieldnames=["audio_path", "text", "emotion"])
            writer.writeheader()
            writer.writerows(split_rows)

    summary = (
        f"Exported training split:\n\n"
        f"Train rows: {len(train_rows)}\n"
        f"Validation rows: {len(val_rows)}\n\n"
        f"Train CSV:\n{train_path}\n\n"
        f"Validation CSV:\n{val_path}"
    )

    train_table = [[row["audio_path"], row["text"], row["emotion"]] for row in train_rows]
    val_table = [[row["audio_path"], row["text"], row["emotion"]] for row in val_rows]
    return summary, train_table, val_table


def load_generation_history(profile_name=DEFAULT_PROFILE):
    history_path = profile_generation_history_path(profile_name)
    if not history_path.exists():
        return []

    with history_path.open("r", newline="", encoding="utf-8") as history_file:
        rows = list(csv.DictReader(history_file))

    return [
        [
            row["input_text"],
            row["translated_text"],
            row["language"],
            row["emotion"],
            row["reference_audio"],
            row["output_audio"],
        ]
        for row in rows
    ]


def model_info_for_profile(profile_name):
    if not profile_is_active(profile_name):
        return profile_required_message()

    profile = clean_profile_name(profile_name)
    checkpoint_path, config_path = finetuned_xtts_model_files(profile)
    model_dir = checkpoint_path.parent if checkpoint_path else next(finetuned_xtts_model_dirs(profile))
    rows = read_metadata_rows(profile)
    counts = Counter(row.get("emotion", "") for row in rows)

    lines = [
        f"Profile: {display_profile_name(profile)}",
        f"Dataset records: {len(rows)}",
        "",
        "Model status:",
    ]
    if checkpoint_path and config_path:
        lines.extend(
            [
                "Fine-tuned XTTS-v2: ready",
                f"Model folder: {model_dir}",
                f"Checkpoint: {checkpoint_path.name}",
            ]
        )
    else:
        lines.extend(
            [
                "Fine-tuned XTTS-v2: not ready",
                "Using stock XTTS-v2 with selected profile reference clips.",
            ]
        )

    lines.append("")
    lines.append("Available moods:")
    if counts:
        lines.extend(f"- {emotion}: {count}" for emotion, count in sorted(counts.items()))
    else:
        lines.append("- no labeled voice samples")
    return "\n".join(lines)


def generate_speech_ui(text, target_language, emotion, enable_translation, profile_name):
    translated, audio_path, result = generate_xtts_speech(
        text,
        target_language,
        emotion,
        enable_translation,
        profile_name,
    )
    return (
        translated,
        audio_path,
        result,
        model_info_for_profile(profile_name),
    )


def refresh_speech_emotion_choices(profile_name):
    choices = emotion_choices_from_dataset(profile_name) if profile_is_active(profile_name) else DEFAULT_EMOTIONS
    value = choices[0] if choices else "neutral"
    return gr.update(choices=choices, value=value)


def show_app_page(page_name):
    pages = ["profile", "dataset", "metadata", "edit", "speech", "training"]
    return [gr.update(visible=page_name == page) for page in pages]




