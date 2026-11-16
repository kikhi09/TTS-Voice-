from pathlib import Path
import os
import sys
import faulthandler


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
METADATA_PATH = DATASET_DIR / "metadata.csv"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
GENERATION_HISTORY_PATH = OUTPUTS_DIR / "generation_history.csv"
ERROR_LOG_PATH = PROJECT_DIR / "error_log.txt"
TRAINING_DATA_DIR = PROJECT_DIR / "training_data"
USERS_DIR = PROJECT_DIR / "users"

faulthandler.enable()

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / "models" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_DIR / "models" / "torch"))
os.environ.setdefault("TTS_HOME", str(PROJECT_DIR / "models" / "tts"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

CONDA_ENV_DIR = Path(sys.executable).resolve().parent
CONDA_BIN_DIR = CONDA_ENV_DIR / "Library" / "bin"
if CONDA_BIN_DIR.exists():
    os.environ["PATH"] = str(CONDA_BIN_DIR) + os.pathsep + os.environ.get("PATH", "")

DEFAULT_EMOTIONS = ["neutral", "happy", "sad", "excited", "angry"]
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}
DEFAULT_PROFILE = "default"
PROFILE_DISPLAY_NAMES = {"my_voice": "MY VOICE", "voice_2": "VOICE 2"}
PROFILE_MODEL_FOLDER_ALIASES = {"voice_2": ["my_voice"]}
HIDDEN_PROFILE_NAMES = {DEFAULT_PROFILE, "neutral", "russ"}
NO_PROFILE_SELECTION = "NONE"
EDITOR_PAGE_SIZE = 10
BATCH_UPLOAD_PAGE_SIZE = 10

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Arabic": "ar",
    "Korean": "ko",
}

NLLB_LANGUAGE_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Japanese": "jpn_Jpan",
    "Arabic": "arb_Arab",
    "Korean": "kor_Hang",
}

