from collections import Counter

import gradio as gr

from language import clean_label
from settings import *


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
    from dataset import read_metadata_rows
    from speech import finetuned_xtts_model_dirs, finetuned_xtts_model_files

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

