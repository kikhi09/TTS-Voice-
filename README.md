# VoiceApp Studio — Project README

## Overview
VoiceApp Studio is a **Gradio-based** web application for recording, managing, and exporting multi-emotion voice datasets for Tacotron 2 / XTTS voice cloning.

Implements all tools from [rasmurtech's Tacotron 2 tutorial](https://youtu.be/e71H--vxRvo).

---

## Quick Start

```bat
# Option 1: Double-click the launcher
start_voiceapp.bat

# Option 2: Command line
"D:\Miniconda3\envs\voiceapp310\python.exe" app.py
```

Then open: **http://localhost:7860**

---

## App Pages

| Page | Description |
|------|-------------|
| **Create Profile** | Create / select a speaker profile |
| **Dataset Builder** | Record audio, import folders, batch process WAVs |
| **Metadata** | View dataset statistics and CSV |
| **Edit Data** | Correct transcripts, move/delete samples |
| **Speech Generator** | Generate speech with XTTS-v2 (local model) |
| **Training Prep & Tools** | Export dataset, run standalone tools, Colab guide |

---

## Training Prep & Tools Page

### Tab 1 — Dataset Health
- Full statistics: total samples, duration, per-emotion breakdown
- Audio validity checks

### Tab 2 — Export Training Dataset
- Exports to **LJSpeech format** (Tacotron 2 / FakeYou compatible)
- Also generates **Coqui XTTS format** (metadata.csv)
- Audio preprocessing: 22050 Hz, mono, silence-trimmed, peak-normalized
- Train/val split (86/14%)
- Creates ZIP ready for Google Colab upload

### Tab 3 — Standalone Tools
- **Batch Rename WAVs** — Rename to `1.wav, 2.wav, ...`
- **Batch Audio Preprocessor** — Resample, trim, normalize
- **Batch Transcribe** — Transcribe folder with faster-whisper
- **Reprocess Dataset** — Apply preprocessing to existing profile audio

### Tab 4 — Google Colab Guide
- Step-by-step Tacotron 2 training instructions
- Links to FakeYou/JustinJohn notebooks

---

## Standalone CLI Tools (`tools/`)

### `batch_rename_wav.py`
```bash
python tools/batch_rename_wav.py <folder> [--start 1] [--prefix ""] [--dry-run]
```
Renames WAV files to `1.wav, 2.wav, ...` (from [rasmurtech/tacotron2-wav-file-renamer](https://github.com/rasmurtech/tacotron2-wav-file-renamer))

### `tacotron2_audio_preprocessor.py`
```bash
python tools/tacotron2_audio_preprocessor.py <input_folder> [--output <out>] [--sample-rate 22050] [--dry-run]
```
Resample + mono + normalize WAV files for Tacotron 2 (from [rasmurtech/Tacotron-2-Audio-Preprocessor](https://github.com/rasmurtech/Tacotron-2-Audio-Preprocessor))

### `transcribe_folder.py`
```bash
python tools/transcribe_folder.py <input_folder> [--output list.txt] [--model base] [--prefix /content/TTS-TT2/wavs]
```
Transcribes numbered WAVs with faster-whisper → LJSpeech `list.txt` (based on [rasmurtech/Tacotron2-Wav2Vec-Transcription](https://github.com/rasmurtech/Tacotron2-Wav2Vec-Transcription))

### `audio_metadata_updater.py`
```bash
python tools/audio_metadata_updater.py <folder> --inplace [--title-prefix "Voice"] [--start-track 1]
python tools/audio_metadata_updater.py <folder> --output <out_folder>
python tools/audio_metadata_updater.py --show-file audio.wav
```
Updates RIFF INFO tags (title, track number) — pure Python, no taglib needed (from [rasmurtech/Audio-Metadata-Updater](https://github.com/rasmurtech/Audio-Metadata-Updater))

---

## Project Structure

```
D:\voiceapp\
├── app.py                      # Main Gradio app (6 pages)
├── main_code.py                # Imports all modules for app
├── audio_processing.py         # Audio I/O, preprocessing, RIFF tags
├── dataset.py                  # Profile dataset management
├── training.py                 # Export, diagnostics, health reports
├── speech.py                   # XTTS-v2 inference (local model priority)
├── transcribe_worker.py        # faster-whisper auto-transcription
├── settings.py                 # Paths, env vars, constants
├── start_voiceapp.bat          # Windows launcher
├── tools/
│   ├── batch_rename_wav.py
│   ├── tacotron2_audio_preprocessor.py
│   ├── transcribe_folder.py
│   └── audio_metadata_updater.py
├── models/
│   ├── tts/tts/tts_models--multilingual--multi-dataset--xtts_v2/
│   ├── huggingface/hub/models--Systran--faster-whisper-base/
│   └── huggingface/hub/models--facebook--nllb-200-distilled-600M/
├── users/<profile_name>/
│   └── dataset/<emotion>/*.wav
└── training_exports/           # Generated ZIPs for Colab
```

---

## Environment

- **Python**: `D:\Miniconda3\envs\voiceapp310\python.exe` (3.10)
- **CUDA**: RTX 2050 (torch 2.5.1+cu121)
- **Key packages**: TTS 0.22.0, gradio 6.13.0, faster-whisper 1.2.1, soundfile 0.13.1, soxr 1.0.0

---

## Google Colab Training

After exporting your dataset ZIP:
1. Upload `my_voice_tacotron2_training.zip` to Google Drive
2. Open [FakeYou Tacotron2 Notebook](https://github.com/justinjohn0306/FakeYou-Tacotron2-Notebook)
3. Mount Drive, extract ZIP, run training cells
4. Download checkpoint for inference
