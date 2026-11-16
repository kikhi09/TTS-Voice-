import importlib.util

from settings import *


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

