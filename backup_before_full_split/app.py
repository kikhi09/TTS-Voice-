import time

import gradio as gr

from core import *
from ui_config import APP_CSS, APP_THEME
with gr.Blocks(
    title="VoiceApp Studio",
    elem_classes=["app-shell"],
) as demo:
    with gr.Group(elem_classes=["sidebar"]):
        gr.HTML(
            """
            <div class="brand-mark"><span>Emotion Voice Cloning</span></div>
            <div class="side-section-title">Multilingual XTTS-v2</div>
            <div class="side-nav">
            """
        )
        profile_page_button = gr.Button("Create profile", variant="secondary")
        dataset_page_button = gr.Button("Dataset builder", variant="secondary")
        metadata_page_button = gr.Button("Metadata", variant="secondary")
        edit_page_button = gr.Button("Edit data", variant="secondary")
        speech_page_button = gr.Button("Speech generator", variant="secondary")
        training_page_button = gr.Button("Training prep", variant="secondary")
        gr.HTML("</div>")

    with gr.Group(elem_classes=["main-workspace"]):
      with gr.Group(visible=True, elem_classes=["workspace-page"]) as profile_page:
        gr.Markdown("## Create Profile", elem_classes=["page-heading"])
        initial_profiles = profile_selector_choices()
        initial_profile = NO_PROFILE_SELECTION
        active_profile_state = gr.State(NO_PROFILE_SELECTION)
        with gr.Column(elem_classes=["profile-section"]):
            gr.Markdown("## New Profile", elem_classes=["section-title"])
            with gr.Row():
                new_profile_name = gr.Textbox(
                    label="New profile name",
                    placeholder="Example: user_001, teacher_voice, client_a",
                    scale=4,
                )
                create_profile_button = gr.Button(
                    "Create / Select",
                    variant="primary",
                    scale=1,
                    elem_classes=["primary-action"],
                )
                refresh_profiles_button = gr.Button("Refresh", variant="secondary", scale=1)
            profile_status = gr.Textbox(
                label="Create profile status",
                lines=5,
                value="Type a new profile name, then click Create / Select.",
                elem_classes=["status-box", "wide-output"],
            )

        with gr.Column(elem_classes=["profile-section"]):
            gr.Markdown("## Select Saved Profile", elem_classes=["section-title"])
            with gr.Row():
                with gr.Column(scale=1):
                    active_profile = gr.Radio(
                        choices=initial_profiles,
                        value=initial_profile,
                        label="Saved profiles",
                        elem_classes=["profile-none-option"],
                    )
                    refresh_saved_profiles_button = gr.Button(
                        "Refresh Saved Profiles",
                        variant="secondary",
                    )
                    use_profile_button = gr.Button(
                        "Use Selected Profile",
                        variant="primary",
                    )
                with gr.Column(scale=1):
                    selected_profile_status = gr.Textbox(
                        label="Selected profile status",
                        lines=10,
                        value="No profile selected. Choose a saved speaker profile, then click Use Selected Profile.",
                        elem_classes=["status-box", "wide-output"],
                    )

        refresh_profiles_button.click(
            fn=refresh_profile_selector,
            inputs=[],
            outputs=[active_profile, selected_profile_status],
        )
        create_profile_button.click(
            fn=create_profile_and_status,
            inputs=new_profile_name,
            outputs=[active_profile, profile_status, selected_profile_status],
        )
        refresh_saved_profiles_button.click(
            fn=refresh_profile_selector,
            inputs=[],
            outputs=[active_profile, selected_profile_status],
        )
        active_profile.change(
            fn=select_profile,
            inputs=active_profile,
            outputs=selected_profile_status,
        )
        use_profile_button.click(
            fn=use_selected_profile,
            inputs=active_profile,
            outputs=[selected_profile_status, active_profile_state],
        )

      with gr.Group(visible=False, elem_classes=["workspace-page"]) as dataset_page:
        gr.Markdown("## Dataset Builder", elem_classes=["page-heading"])
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("## Single Sample", elem_classes=["section-title"])
                    audio = gr.Audio(
                        label="Upload or record emotion audio",
                        sources=["upload", "microphone"],
                        type="filepath",
                    )
                    with gr.Row():
                        selected_emotion = gr.Dropdown(
                            DEFAULT_EMOTIONS,
                            label="Emotion label",
                            value="neutral",
                        )
                        custom_emotion = gr.Textbox(
                            label="Custom emotion",
                            placeholder="Optional, for example: calm, fearful, surprised",
                        )

                    transcript = gr.Textbox(
                        label="Transcript",
                        placeholder="Optional. Leave empty to auto-transcribe with faster-whisper.",
                        lines=3,
                    )
                    auto_transcribe = gr.Checkbox(
                        label="Auto-transcribe if transcript is empty",
                        value=True,
                    )
                    save_button = gr.Button(
                        "Save Audio Sample",
                        variant="primary",
                        elem_classes=["primary-action"],
                    )
            with gr.Column(scale=2):
                save_output = gr.Textbox(
                    label="Dataset result",
                    lines=13,
                    elem_classes=["result-box"],
                )

        save_button.click(
            fn=save_audio_sample,
            inputs=[audio, selected_emotion, custom_emotion, transcript, auto_transcribe, active_profile_state],
            outputs=save_output,
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("## Batch Upload", elem_classes=["section-title"])
                    batch_audio = gr.File(
                        label="Upload multiple audio files",
                        file_count="multiple",
                        file_types=["audio"],
                        height=180,
                    )
                    batch_upload_page_state = gr.State(0)
                    batch_upload_status = gr.Textbox(
                        label="Uploaded file batch",
                        lines=1,
                        value="No batch files selected.",
                        elem_classes=["status-box"],
                    )
                    with gr.Row():
                        previous_batch_upload = gr.Button("Previous 10", variant="secondary")
                        next_batch_upload = gr.Button("Next 10", variant="secondary")
                    with gr.Row():
                        batch_emotion = gr.Dropdown(
                            DEFAULT_EMOTIONS,
                            label="Batch emotion label",
                            value="neutral",
                        )
                        batch_custom_emotion = gr.Textbox(
                            label="Batch custom emotion",
                            placeholder="Optional, for example: calm, fearful, surprised",
                        )
                    batch_auto_transcribe = gr.Checkbox(
                        label="Auto-transcribe batch files",
                        value=True,
                    )
                    save_batch_button = gr.Button("Save Batch", variant="primary")
            with gr.Column(scale=1):
                batch_output = gr.Textbox(
                    label="Batch result",
                    lines=12,
                    elem_classes=["result-box"],
                )

        save_batch_button.click(
            fn=save_audio_batch,
            inputs=[batch_audio, batch_emotion, batch_custom_emotion, batch_auto_transcribe, active_profile_state],
            outputs=batch_output,
        )
        batch_audio.change(
            fn=load_batch_upload_preview,
            inputs=batch_audio,
            outputs=[batch_upload_page_state, batch_upload_status],
        )
        previous_batch_upload.click(
            fn=lambda files, page: change_batch_upload_page(-1, files, page),
            inputs=[batch_audio, batch_upload_page_state],
            outputs=[batch_upload_page_state, batch_upload_status],
        )
        next_batch_upload.click(
            fn=lambda files, page: change_batch_upload_page(1, files, page),
            inputs=[batch_audio, batch_upload_page_state],
            outputs=[batch_upload_page_state, batch_upload_status],
        )

      with gr.Group(visible=False, elem_classes=["workspace-page"]) as metadata_page:
        gr.Markdown("## Metadata", elem_classes=["page-heading"])
        refresh_metadata = gr.Button("Refresh Metadata", variant="primary")
        with gr.Row():
            metadata_summary = gr.Textbox(
                label="Dataset summary",
                lines=10,
                elem_classes=["status-box"],
            )
            metadata_table = gr.Dataframe(
                headers=["audio_path", "text", "emotion"],
                datatype=["str", "str", "str"],
                label="metadata.csv",
                interactive=False,
            )

        refresh_metadata.click(
            fn=load_metadata_preview,
            inputs=active_profile_state,
            outputs=[metadata_summary, metadata_table],
        )
        use_profile_button.click(
            fn=load_metadata_for_selected_profile,
            inputs=active_profile,
            outputs=[metadata_summary, metadata_table],
        )

      with gr.Group(visible=False, elem_classes=["workspace-page"]) as edit_page:
        gr.Markdown("## Edit Data", elem_classes=["page-heading"])
        initial_edit_emotions = emotion_editor_choices(initial_profile)
        if not initial_edit_emotions:
            initial_edit_emotions = DEFAULT_EMOTIONS
        initial_edit_emotion = initial_edit_emotions[0]
        initial_edit_rows = rows_for_emotion(initial_edit_emotion, initial_profile)
        initial_audio_choices, initial_selected_audio, initial_audio_page, initial_audio_page_status = audio_page_choices(
            initial_edit_rows,
            0,
        )
        edit_audio_page_state = gr.State(initial_audio_page)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("## Mood Filter", elem_classes=["section-title"])
                    edit_emotion_filter = gr.Radio(
                        choices=initial_edit_emotions,
                        value=initial_edit_emotion,
                        label="Select mood",
                    )
                    refresh_emotion_editor = gr.Button("Refresh Mood Samples", variant="primary")
                    edit_filter_status = gr.Textbox(
                        label="Mood status",
                        lines=3,
                        value=f"{initial_edit_emotion}: {len(initial_edit_rows)} sample(s) | {initial_audio_page_status}",
                        elem_classes=["status-box"],
                    )
                    cleanup_metadata_button = gr.Button(
                        "Clean Duplicate / Missing Metadata",
                        variant="secondary",
                    )

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("## Audio In This Mood", elem_classes=["section-title"])
                    edit_audio_list = gr.Radio(
                        choices=initial_audio_choices,
                        value=initial_selected_audio,
                        label="Audio samples",
                    )
                    edit_audio_page_status = gr.Textbox(
                        label="Current audio batch",
                        lines=1,
                        value=initial_audio_page_status,
                        elem_classes=["status-box"],
                    )
                    with gr.Row():
                        previous_audio_batch = gr.Button("Previous 10", variant="secondary")
                        next_audio_batch = gr.Button("Next 10", variant="secondary")
                    edit_sample_audio = gr.Audio(
                        label="Listen selected audio",
                        type="filepath",
                        value=sample_audio_output(initial_selected_audio, initial_profile),
                        interactive=False,
                    )
                    move_sample_emotion = gr.Dropdown(
                        choices=DEFAULT_EMOTIONS,
                        value=initial_edit_emotion if initial_edit_emotion in DEFAULT_EMOTIONS else "neutral",
                        label="Move selected audio to mood",
                    )
                    with gr.Row():
                        move_sample_button = gr.Button("Change Mood", variant="primary")
                        delete_editor_sample_button = gr.Button(
                            "Delete Audio",
                            variant="stop",
                            elem_classes=["danger-action"],
                        )
                    editor_action_status = gr.Textbox(
                        label="Audio action status",
                        lines=3,
                        elem_classes=["result-box"],
                    )

            with gr.Column(scale=3):
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("## Transcript Correction", elem_classes=["section-title"])
                    mood_transcript_table = gr.Dataframe(
                        headers=["audio_path", "text", "emotion"],
                        datatype=["str", "str", "str"],
                        value=initial_edit_rows,
                        row_count=len(initial_edit_rows),
                        column_count=3,
                        label="Transcripts for selected mood",
                        interactive=True,
                        elem_classes=["transcript-grid"],
                    )
                    save_mood_transcripts = gr.Button("Save Transcript Corrections", variant="primary")
                    transcript_save_status = gr.Textbox(
                        label="Save result",
                        lines=4,
                        elem_classes=["result-box"],
                    )

        refresh_emotion_editor.click(
            fn=load_emotion_editor,
            inputs=[edit_emotion_filter, active_profile_state],
            outputs=[
                edit_emotion_filter,
                edit_audio_list,
                edit_sample_audio,
                mood_transcript_table,
                edit_filter_status,
                edit_audio_page_state,
                edit_audio_page_status,
            ],
        )
        edit_emotion_filter.change(
            fn=load_emotion_editor,
            inputs=[edit_emotion_filter, active_profile_state],
            outputs=[
                edit_emotion_filter,
                edit_audio_list,
                edit_sample_audio,
                mood_transcript_table,
                edit_filter_status,
                edit_audio_page_state,
                edit_audio_page_status,
            ],
        )
        previous_audio_batch.click(
            fn=lambda emotion_name, page, profile: change_editor_audio_page(-1, emotion_name, page, profile),
            inputs=[edit_emotion_filter, edit_audio_page_state, active_profile_state],
            outputs=[
                edit_audio_list,
                edit_sample_audio,
                edit_audio_page_state,
                edit_audio_page_status,
            ],
        )
        next_audio_batch.click(
            fn=lambda emotion_name, page, profile: change_editor_audio_page(1, emotion_name, page, profile),
            inputs=[edit_emotion_filter, edit_audio_page_state, active_profile_state],
            outputs=[
                edit_audio_list,
                edit_sample_audio,
                edit_audio_page_state,
                edit_audio_page_status,
            ],
        )
        edit_audio_list.change(
            fn=sample_audio_output,
            inputs=[edit_audio_list, active_profile_state],
            outputs=edit_sample_audio,
        )
        save_mood_transcripts.click(
            fn=save_emotion_transcript_table,
            inputs=[edit_emotion_filter, mood_transcript_table, active_profile_state],
            outputs=[transcript_save_status, mood_transcript_table],
        )
        move_sample_button.click(
            fn=move_editor_sample,
            inputs=[edit_audio_list, move_sample_emotion, edit_emotion_filter, active_profile_state],
            outputs=[
                editor_action_status,
                edit_audio_list,
                edit_sample_audio,
                mood_transcript_table,
                edit_audio_page_state,
                edit_audio_page_status,
            ],
        )
        delete_editor_sample_button.click(
            fn=delete_editor_sample,
            inputs=[edit_audio_list, edit_emotion_filter, active_profile_state],
            outputs=[
                editor_action_status,
                edit_audio_list,
                edit_sample_audio,
                mood_transcript_table,
                edit_audio_page_state,
                edit_audio_page_status,
            ],
        )
        cleanup_metadata_button.click(
            fn=cleanup_duplicate_metadata,
            inputs=active_profile_state,
            outputs=[
                editor_action_status,
                edit_emotion_filter,
                edit_audio_list,
                edit_sample_audio,
                mood_transcript_table,
                edit_audio_page_state,
                edit_audio_page_status,
            ],
        )

      with gr.Group(visible=False, elem_classes=["workspace-page"]) as speech_page:
        gr.Markdown("## Speech Generator", elem_classes=["page-heading"])
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("## Voice Controls", elem_classes=["section-title"])
                    text = gr.Textbox(
                        label="English text",
                        placeholder="Type the sentence you want to translate and speak...",
                        lines=5,
                    )
                    with gr.Row():
                        target_language = gr.Dropdown(
                            available_language_names(),
                            label="Target language",
                            value="English",
                        )
                        emotion = gr.Dropdown(
                            DEFAULT_EMOTIONS,
                            label="Mood",
                            value="neutral",
                        )
                    enable_translation = gr.Checkbox(
                        label="Translate text when target language is not English",
                        value=False,
                        elem_classes=["speech-toggle"],
                    )
                    with gr.Row():
                        run_button = gr.Button(
                            "Generate Speech",
                            variant="primary",
                            elem_classes=["primary-action"],
                        )
                        refresh_emotions = gr.Button("Refresh Moods", variant="secondary")
                generated_audio = gr.Audio(label="Generated audio", type="filepath")
            with gr.Column(scale=2):
                model_info = gr.Textbox(
                    label="Model in use",
                    lines=12,
                    value=profile_required_message(),
                    elem_classes=["status-box"],
                )
                output = gr.Textbox(
                    label="Generation result",
                    lines=12,
                    elem_classes=["result-box"],
                )
                translated_text = gr.Textbox(label="Translated / spoken text", lines=3)

        run_button.click(
            fn=generate_speech_ui,
            inputs=[text, target_language, emotion, enable_translation, active_profile_state],
            outputs=[translated_text, generated_audio, output, model_info],
        )
        refresh_emotions.click(
            fn=refresh_speech_emotion_choices,
            inputs=active_profile_state,
            outputs=emotion,
        )
        use_profile_button.click(
            fn=model_info_for_profile,
            inputs=active_profile,
            outputs=model_info,
        )

      with gr.Group(visible=False, elem_classes=["workspace-page"]) as training_page:
        gr.Markdown("## Training Prep", elem_classes=["page-heading"])
        check_dataset = gr.Button("Check Dataset Health", variant="primary")
        training_summary = gr.Textbox(
            label="Training readiness",
            lines=14,
            elem_classes=["status-box"],
        )

        check_dataset.click(
            fn=dataset_health_summary_only,
            inputs=active_profile_state,
            outputs=training_summary,
        )

      page_outputs = [
          profile_page,
          dataset_page,
          metadata_page,
          edit_page,
          speech_page,
          training_page,
      ]
      profile_page_button.click(
          fn=lambda: show_app_page("profile"),
          inputs=[],
          outputs=page_outputs,
      )
      dataset_page_button.click(
          fn=lambda: show_app_page("dataset"),
          inputs=[],
          outputs=page_outputs,
      )
      metadata_page_button.click(
          fn=lambda: show_app_page("metadata"),
          inputs=[],
          outputs=page_outputs,
      )
      edit_page_button.click(
          fn=lambda: show_app_page("edit"),
          inputs=[],
          outputs=page_outputs,
      )
      speech_page_button.click(
          fn=lambda: show_app_page("speech"),
          inputs=[],
          outputs=page_outputs,
      )
      training_page_button.click(
          fn=lambda: show_app_page("training"),
          inputs=[],
          outputs=page_outputs,
      )


if __name__ == "__main__":
    demo.launch(
        show_error=True,
        prevent_thread_lock=True,
        allowed_paths=[str(PROJECT_DIR)],
        theme=APP_THEME,
        css=APP_CSS,
    )
    while True:
        time.sleep(3600)


