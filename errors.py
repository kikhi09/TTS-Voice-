import sys
import traceback

from settings import *


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


def read_error_log():
    if not ERROR_LOG_PATH.exists():
        return "No app errors logged yet."

    text = ERROR_LOG_PATH.read_text(encoding="utf-8")
    return text[-8000:] if len(text) > 8000 else text


def clear_error_log():
    ERROR_LOG_PATH.write_text("", encoding="utf-8")
    return "Error log cleared."

