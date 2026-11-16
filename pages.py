

import gradio as gr


def show_app_page(page_name):
    pages = ["profile", "dataset", "metadata", "edit", "speech", "training"]
    return [gr.update(visible=page_name == page) for page in pages]

