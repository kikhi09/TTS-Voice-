import csv
import os
from pathlib import Path
import platform
import random
import shutil
import traceback
import zipfile
from collections import Counter

import gradio as gr
import soundfile as sf

from audio_processing import preprocess_audio_file, write_riff_info_chunk
from dataset import audio_file_path, read_metadata_rows
from errors import clear_error_log, log_error, read_error_log
from profiles import clean_profile_name, display_profile_name, profile_base_dir, profile_generation_history_path, profile_is_active, profile_required_message
from settings import *


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
        PROJECT_DIR / "training_exports",
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
    for module_name in ["gradio", "TTS", "transformers", "faster_whisper", "soundfile", "soxr", "numpy"]:
        try:
            module = __import__(module_name)
            lines.append(f"{module_name}: {getattr(module, '__version__', 'installed')}")
        except Exception as exc:
            lines.append(f"{module_name}: error: {exc}")

    return "\n".join(lines)


def dataset_health_report(profile_name=DEFAULT_PROFILE):
    rows = read_metadata_rows(profile_name)
    if not rows:
        return "No dataset records found. Save audio samples first.", []

    counts = Counter(row.get("emotion", "") for row in rows)
    missing_transcripts = [row for row in rows if not row.get("text", "").strip()]
    base_dir = profile_base_dir(profile_name)

    missing_audio = []
    durations = []
    for row in rows:
        audio_rel = row.get("audio_path")
        if not audio_rel:
            missing_audio.append(row)
            continue
        full_path = base_dir / audio_rel
        if not full_path.exists():
            missing_audio.append(row)
        else:
            try:
                info = sf.info(str(full_path))
                durations.append(info.duration)
            except Exception:
                pass

    total_duration_sec = sum(durations)
    total_minutes = total_duration_sec / 60.0
    avg_duration = (total_duration_sec / len(durations)) if durations else 0

    lines = [
        f"### Training Dataset Health: {display_profile_name(profile_name)}",
        "",
        f"- **Total recorded samples**: {len(rows)}",
        f"- **Total audio duration**: {total_minutes:.2f} minutes ({total_duration_sec:.1f} seconds)",
        f"- **Average clip duration**: {avg_duration:.2f} seconds",
        f"- **Missing transcripts**: {len(missing_transcripts)}",
        f"- **Missing audio files**: {len(missing_audio)}",
        "",
        "#### Samples by mood / emotion:",
    ]
    lines.extend(f"- **{emotion or '[empty]'}**: {count} samples" for emotion, count in sorted(counts.items()))
    lines.extend(
        [
            "",
            "#### Training Readiness Criteria:",
            "- **Tacotron 2 / FastSpeech / XTTS**: 100+ clean samples (recommended: 15-30+ minutes of speech)",
            "- **Zero-shot XTTS cloning**: 5-20 clean reference samples (already functional)",
            "- Each sample should have an accurate transcript matching spoken words",
            "- Clean audio: no background noise, consistent microphone distance, trimmed silence",
            "",
            "#### Current Status:",
        ]
    )

    if missing_audio:
        lines.append("❌ **Not ready**: some metadata rows point to missing audio files.")
    elif missing_transcripts:
        lines.append("⚠️ **Incomplete**: some samples are missing transcripts. Use the Edit Data tab to fill them.")
    elif len(rows) < 15:
        lines.append("⚠️ **Prototype only**: Add more samples before fine-tuning.")
    elif len(rows) < 100:
        lines.append("⚡ **Small fine-tuning dataset**: Ready for initial test training.")
    else:
        lines.append("✅ **Ready for model training!** Full dataset export available below.")

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


def export_training_package(
    profile_name=DEFAULT_PROFILE,
    target_format="Tacotron 2 / FakeYou (LJSpeech)",
    validation_ratio=0.15,
    preprocess_audio_flag=True,
    target_sr_choice="22050 Hz (Tacotron 2)",
    trim_silence_flag=True,
    normalize_flag=True,
):
    """
    Exports the profile's audio dataset into a complete, standard training package:
    - Preprocesses all audio (silence trimming, volume normalization, mono, resampling)
    - Updates RIFF WAV metadata tags (Title and Track Number)
    - Splits into Train and Validation subsets
    - Generates standard format metadata (LJSpeech list.txt / metadata_train.txt / metadata_eval.txt)
    - Creates a ready-to-upload ZIP archive for Google Colab (FakeYou Tacotron 2 or XTTS)
    """
    if not profile_is_active(profile_name):
        return profile_required_message(), None, [], []

    profile = clean_profile_name(profile_name)
    base_dir = profile_base_dir(profile)
    rows = [
        row for row in read_metadata_rows(profile)
        if row.get("audio_path")
        and row.get("text", "").strip()
        and (base_dir / row["audio_path"]).exists()
    ]

    if len(rows) < 2:
        return "Need at least 2 complete samples with audio and transcript to export training data.", None, [], []

    # Parse target sample rate
    if "22050" in str(target_sr_choice):
        target_sr = 22050
    elif "24000" in str(target_sr_choice):
        target_sr = 24000
    elif "16000" in str(target_sr_choice):
        target_sr = 16000
    else:
        target_sr = 22050

    is_tacotron = "Tacotron" in target_format

    # Stratified split by emotion
    random.seed(42)
    rows_by_emotion = {}
    for row in rows:
        rows_by_emotion.setdefault(row.get("emotion", "neutral"), []).append(row)

    train_rows = []
    val_rows = []
    for emotion, emotion_rows in rows_by_emotion.items():
        shuffled = emotion_rows[:]
        random.shuffle(shuffled)
        val_count = 1 if len(shuffled) > 1 else 0
        val_count = max(val_count, int(len(shuffled) * validation_ratio))
        val_rows.extend(shuffled[:val_count])
        train_rows.extend(shuffled[val_count:])

    # Setup export directories
    format_tag = "tacotron2" if is_tacotron else "xtts"
    export_folder_name = f"{profile}_{format_tag}_{len(rows)}"
    exports_base_dir = PROJECT_DIR / "training_exports"
    exports_base_dir.mkdir(parents=True, exist_ok=True)

    export_dir = exports_base_dir / export_folder_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    wavs_dir = export_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    processed_train = []
    processed_val = []

    # Process and copy audio files
    current_index = 1

    def process_split_rows(split_list, is_val=False):
        nonlocal current_index
        output_records = []
        for row in split_list:
            src_audio = base_dir / row["audio_path"]
            emotion = row.get("emotion", "neutral")
            transcript = row.get("text", "").strip()

            if is_tacotron:
                wav_filename = f"{current_index}.wav"
                rel_wav_path = f"wavs/{wav_filename}"
                track_id = current_index
            else:
                prefix = "val" if is_val else "train"
                wav_filename = f"{prefix}_{emotion}_{current_index:05d}.wav"
                rel_wav_path = f"wavs/{wav_filename}"
                track_id = current_index

            dest_audio = wavs_dir / wav_filename

            if preprocess_audio_flag:
                preprocess_audio_file(
                    src_audio,
                    dest_audio,
                    target_sr=target_sr,
                    trim_silence=trim_silence_flag,
                    top_db=20,
                    normalize=normalize_flag,
                    update_tags=True,
                    title=str(track_id),
                    track_number=track_id,
                    artist=profile,
                )
            else:
                shutil.copy2(src_audio, dest_audio)
                write_riff_info_chunk(dest_audio, title=str(track_id), track_number=track_id, artist=profile)

            output_records.append({
                "audio_path": rel_wav_path,
                "text": transcript,
                "emotion": emotion,
                "id": str(track_id),
            })
            current_index += 1
        return output_records

    processed_train = process_split_rows(train_rows, is_val=False)
    processed_val = process_split_rows(val_rows, is_val=True)

    # Write metadata files according to target format
    metadata_train_path = export_dir / "metadata_train.txt"
    metadata_eval_path = export_dir / "metadata_eval.txt"
    train_csv_path = export_dir / "train.csv"
    val_csv_path = export_dir / "val.csv"

    # CSV files with emotions preserved
    for path, data_records in [(train_csv_path, processed_train), (val_csv_path, processed_val)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path", "text", "emotion"])
            writer.writeheader()
            for r in data_records:
                writer.writerow({"audio_path": r["audio_path"], "text": r["text"], "emotion": r["emotion"]})

    if is_tacotron:
        # Standard LJSpeech pipe format: audio_path|transcript|normalized_transcript
        list_txt_path = export_dir / "list.txt"
        with list_txt_path.open("w", encoding="utf-8") as f_all:
            for r in processed_train + processed_val:
                f_all.write(f"{r['audio_path']}|{r['text']}\n")

        with metadata_train_path.open("w", encoding="utf-8") as f_train:
            for r in processed_train:
                f_train.write(f"{r['audio_path']}|{r['text']}|{r['text']}\n")

        with metadata_eval_path.open("w", encoding="utf-8") as f_val:
            for r in processed_val:
                f_val.write(f"{r['audio_path']}|{r['text']}|{r['text']}\n")

    else:
        # Coqui XTTS pipe format: audio_path|transcript|speaker_name
        with metadata_train_path.open("w", encoding="utf-8") as f_train:
            for r in processed_train:
                f_train.write(f"{r['audio_path']}|{r['text']}|{profile}\n")

        with metadata_eval_path.open("w", encoding="utf-8") as f_val:
            for r in processed_val:
                f_val.write(f"{r['audio_path']}|{r['text']}|{profile}\n")

    # Generate README instructions
    readme_path = export_dir / "README.txt"
    instructions = f"""========================================================================
Training Dataset Export: {display_profile_name(profile)}
Target Architecture: {target_format}
Sample Rate: {target_sr} Hz (Mono 16-bit PCM WAV)
Silence Trimmed: {trim_silence_flag} (20 dB threshold)
Volume Normalized: {normalize_flag} (Peak -0.5 dB)
Total Samples: {len(rows)} (Train: {len(processed_train)}, Validation: {len(processed_val)})
========================================================================

FILES INCLUDED IN THIS ARCHIVE:
- wavs/ : Directory containing all clean, preprocessed WAV files
- metadata_train.txt : Pipe-delimited training dataset
- metadata_eval.txt : Pipe-delimited validation/evaluation dataset
- train.csv / val.csv : CSV formatted copies with emotion labels
"""
    if is_tacotron:
        instructions += """- list.txt : Standard Tacotron 2 / LJSpeech file list

HOW TO TRAIN ON GOOGLE COLAB (FakeYou Tacotron 2):
1. Open the FakeYou Tacotron 2 Colab Notebook:
   https://github.com/justinjohn0306/FakeYou-Tacotron2-Notebook
   or directly in Colab:
   https://colab.research.google.com/github/justinjohn0306/FakeYou-Tacotron2-Notebook/blob/main/FakeYou_Tacotron_2_Training.ipynb

2. In Google Colab, connect to a GPU runtime (Runtime -> Change runtime type -> T4 GPU).

3. Upload this exported .zip archive to your Google Drive or directly to the Colab session.

4. Unzip the dataset:
   !unzip your_dataset.zip -d /content/dataset/

5. Point the notebook data loader to:
   - Audio directory: /content/dataset/wavs
   - Training list: /content/dataset/metadata_train.txt
   - Validation list: /content/dataset/metadata_eval.txt

6. Run the training cells. Monitor tensorboard for alignment and loss.

7. Once trained, download the generated checkpoint (.pth) and import it into VoiceApp Studio!
"""
    else:
        instructions += """
HOW TO FINE-TUNE XTTS-v2:
1. Upload this exported .zip archive to your Google Drive or Colab GPU runtime.
2. Unzip into /content/dataset/
3. Run Coqui TTS XTTS fine-tuning with:
   - train_csv: /content/dataset/metadata_train.txt
   - eval_csv: /content/dataset/metadata_eval.txt
   - wavs_dir: /content/dataset/wavs
4. Download the best_model.pth and config.json and place them in:
   users/{profile}/models/{profile}_xtts_finetuned/
"""
    readme_path.write_text(instructions, encoding="utf-8")

    # Create ZIP archive
    zip_path = exports_base_dir / f"{profile}_{format_tag}_training.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(str(export_dir)):
            for file in files:
                file_abs = Path(root) / file
                arcname = file_abs.relative_to(export_dir)
                zip_file.write(str(file_abs), str(arcname))

    summary_md = f"""### 🎉 Training Package Exported Successfully!

- **Profile**: `{display_profile_name(profile)}`
- **Model Architecture**: `{target_format}`
- **Total Samples Exported**: **{len(rows)}**
  - **Train Set**: {len(processed_train)} samples
  - **Validation Set**: {len(processed_val)} samples
- **Audio Preprocessing Applied**:
  - Sample Rate: `{target_sr} Hz`
  - Channels: `Mono 16-bit PCM WAV`
  - Silence Trimming: `{'Enabled (20 dB threshold)' if trim_silence_flag else 'Disabled'}`
  - Volume Normalization: `{'Enabled (Peak -0.5 dB)' if normalize_flag else 'Disabled'}`
  - RIFF Tags: Embedded Title & Track numbers
- **Export Directory**: `{export_dir}`
- **ZIP Package Ready for Colab**: `{zip_path.name}` ({zip_path.stat().st_size / (1024*1024):.2f} MB)

*Download the ZIP archive below and follow the Google Colab Guide to train your custom model.*
"""

    train_table = [[r["audio_path"], r["text"], r["emotion"]] for r in processed_train]
    val_table = [[r["audio_path"], r["text"], r["emotion"]] for r in processed_val]

    return summary_md, str(zip_path), train_table, val_table


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
