from shutil import copyfile
import json
import os
import traceback
from collections import Counter

import gradio as gr

from dataset import append_generation_history, emotion_choices_from_dataset, read_metadata_rows, reference_audios_for_emotion
from errors import log_error
from language import clean_label, language_unavailable_message
from profiles import *
from settings import *

XTTS_MODEL = None
XTTS_MODEL_SOURCE = None
TRANSLATION_TOKENIZER = None
TRANSLATION_MODEL = None


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

        local_xtts_dir = PROJECT_DIR / "models" / "tts" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2"
        local_config = local_xtts_dir / "config.json"
        if local_xtts_dir.exists() and local_config.exists():
            try:
                XTTS_MODEL = TTS(model_path=str(local_xtts_dir), config_path=str(local_config)).to(device)
                XTTS_MODEL_SOURCE = str(local_xtts_dir)
                return XTTS_MODEL
            except Exception:
                log_error("load_local_xtts_model", traceback.format_exc())

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

        original_text = text.strip()
        display_text = (
            f"Original English text:\n{original_text}\n\n"
            f"Translated / spoken text:\n{text_for_speech}"
        )

        return display_text, str(output_path), (
            f"Generated speech:\n{output_path}\n\n"
            f"Model source:\n{XTTS_MODEL_SOURCE}\n\n"
            f"Reference audio:\n{primary_reference_audio}\n\n"
            f"Reference clips used: {len(reference_audios)}\n\n"
            f"Original English text:\n{original_text}\n\n"
            f"Translated / spoken text:\n{text_for_speech}\n\n"
            f"Language: {target_language} ({language_code})\n"
            f"Emotion reference: {emotion}"
        )
    except Exception:
        error_text = traceback.format_exc()
        log_error("generate_xtts_speech", error_text)
        return "", None, "XTTS generation failed:\n\n" + error_text


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

