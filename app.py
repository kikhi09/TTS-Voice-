import time

import gradio as gr

from main_code import *
from style import APP_CSS, APP_THEME

with gr.Blocks(
    title="VoiceApp Studio",
    elem_classes=["app-shell"],
) as demo:
    with gr.Group(elem_classes=["sidebar"]):
        gr.HTML(
            """
            <div class="brand-mark"><span>Emotion Voice Cloning</span></div>
            <div class="side-section-title">Multilingual XTTS & Tacotron 2</div>
            <div class="side-nav">
            """
        )
        profile_page_button = gr.Button("Create profile", variant="secondary")
        dataset_page_button = gr.Button("Dataset builder", variant="secondary")
        metadata_page_button = gr.Button("Metadata", variant="secondary")
        edit_page_button = gr.Button("Edit data", variant="secondary")
        speech_page_button = gr.Button("Speech generator", variant="secondary")
        training_page_button = gr.Button("Training prep & tools", variant="secondary")
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
                    with gr.Row():
                        auto_transcribe = gr.Checkbox(
                            label="Auto-transcribe if transcript is empty",
                            value=True,
                        )
                        single_preprocess_audio = gr.Checkbox(
                            label="Preprocess audio (Trim silence & normalize)",
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
            inputs=[audio, selected_emotion, custom_emotion, transcript, auto_transcribe, active_profile_state, single_preprocess_audio],
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
                    with gr.Row():
                        batch_auto_transcribe = gr.Checkbox(
                            label="Auto-transcribe batch files",
                            value=True,
                        )
                        batch_preprocess_audio = gr.Checkbox(
                            label="Preprocess audio (Trim silence & normalize)",
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
            inputs=[batch_audio, batch_emotion, batch_custom_emotion, batch_auto_transcribe, active_profile_state, batch_preprocess_audio],
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

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("## Folder Import & Preprocess", elem_classes=["section-title"])
                    gr.Markdown("Import an entire directory of audio files directly into your active voice profile.")
                    folder_path_input = gr.Textbox(
                        label="Folder path containing audio files",
                        placeholder="e.g. D:/audio_recordings, C:/Users/.../wavs",
                    )
                    with gr.Row():
                        folder_emotion = gr.Dropdown(
                            DEFAULT_EMOTIONS,
                            label="Emotion label",
                            value="neutral",
                        )
                        folder_custom_emotion = gr.Textbox(
                            label="Custom emotion",
                            placeholder="Optional, for example: calm, fearful, surprised",
                        )
                    with gr.Row():
                        folder_auto_transcribe = gr.Checkbox(
                            label="Auto-transcribe with faster-whisper",
                            value=True,
                        )
                        folder_preprocess_check = gr.Checkbox(
                            label="Preprocess audio (Trim silence & normalize)",
                            value=True,
                        )
                    import_folder_button = gr.Button("Import & Preprocess Folder", variant="primary")
            with gr.Column(scale=1):
                folder_output = gr.Textbox(
                    label="Folder import result",
                    lines=12,
                    elem_classes=["result-box"],
                )

        import_folder_button.click(
            fn=import_emotion_folder,
            inputs=[
                folder_path_input,
                folder_emotion,
                folder_custom_emotion,
                folder_auto_transcribe,
                active_profile_state,
                folder_preprocess_check,
            ],
            outputs=folder_output,
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
        gr.Markdown("## Training Prep & Model Studio", elem_classes=["page-heading"])

        with gr.Tabs():
            with gr.Tab("1. Dataset Health"):
                gr.Markdown("### Training Readiness & Health Analysis", elem_classes=["section-title"])
                with gr.Row():
                    check_dataset = gr.Button("Check Dataset Health", variant="primary")
                    reprocess_dataset_button = gr.Button("Clean & Normalize Existing Audio", variant="secondary")
                training_summary = gr.Markdown(
                    value="Select a profile and click **Check Dataset Health** to verify readiness."
                )
                reprocess_status = gr.Textbox(
                    label="Audio cleaning status",
                    lines=3,
                    elem_classes=["status-box"],
                )

                check_dataset.click(
                    fn=dataset_health_summary_only,
                    inputs=active_profile_state,
                    outputs=training_summary,
                )
                reprocess_dataset_button.click(
                    fn=reprocess_profile_dataset,
                    inputs=active_profile_state,
                    outputs=reprocess_status,
                )

            with gr.Tab("2. Export Training Dataset"):
                gr.Markdown("### Export & Package Dataset for Google Colab / Local Training", elem_classes=["section-title"])
                gr.Markdown(
                    "Exports your voice recordings with standard LJSpeech / Tacotron 2 or Coqui XTTS formatting, "
                    "applies silence trimming and normalization, and bundles everything into a ready-to-train ZIP archive."
                )
                with gr.Row():
                    target_model_format = gr.Radio(
                        ["Tacotron 2 / FakeYou (LJSpeech)", "Coqui XTTS-v2"],
                        label="Target Model Architecture",
                        value="Tacotron 2 / FakeYou (LJSpeech)",
                    )
                    export_target_sr = gr.Dropdown(
                        ["22050 Hz (Tacotron 2)", "24000 Hz (XTTS-v2)", "16000 Hz (Wav2Vec2)"],
                        label="Target Audio Sample Rate",
                        value="22050 Hz (Tacotron 2)",
                    )
                with gr.Row():
                    export_val_ratio = gr.Slider(
                        minimum=0.05,
                        maximum=0.30,
                        step=0.05,
                        value=0.15,
                        label="Validation Split Ratio",
                    )
                    export_preprocess_flag = gr.Checkbox(
                        label="Preprocess audio (Trim silence, normalize, mono)",
                        value=True,
                    )
                    export_trim_silence_flag = gr.Checkbox(
                        label="Silence trimming (20 dB threshold)",
                        value=True,
                    )
                    export_normalize_flag = gr.Checkbox(
                        label="Peak normalization (-0.5 dB)",
                        value=True,
                    )

                export_package_button = gr.Button(
                    "📦 Export & Package Training Dataset (ZIP)",
                    variant="primary",
                    elem_classes=["primary-action"],
                )
                export_summary_output = gr.Markdown()
                download_export_file = gr.File(label="Download Generated Training Archive (.zip)")

                with gr.Accordion("Preview Train Split & Validation Split", open=False):
                    with gr.Row():
                        train_split_table = gr.Dataframe(
                            headers=["audio_path", "text", "emotion"],
                            datatype=["str", "str", "str"],
                            label="Train Split Preview",
                            interactive=False,
                        )
                        val_split_table = gr.Dataframe(
                            headers=["audio_path", "text", "emotion"],
                            datatype=["str", "str", "str"],
                            label="Validation Split Preview",
                            interactive=False,
                        )

                export_package_button.click(
                    fn=export_training_package,
                    inputs=[
                        active_profile_state,
                        target_model_format,
                        export_val_ratio,
                        export_preprocess_flag,
                        export_target_sr,
                        export_trim_silence_flag,
                        export_normalize_flag,
                    ],
                    outputs=[
                        export_summary_output,
                        download_export_file,
                        train_split_table,
                        val_split_table,
                    ],
                )

            with gr.Tab("3. Standalone Tools"):
                gr.Markdown("### Standalone Audio Tools (From Video Tutorial)", elem_classes=["section-title"])
                with gr.Row():
                    with gr.Column():
                        with gr.Group(elem_classes=["panel"]):
                            gr.Markdown("#### Batch WAV Renamer", elem_classes=["section-title"])
                            gr.Markdown("Renames all WAV files in a folder sequentially to `1.wav`, `2.wav`, etc.")
                            rename_folder_input = gr.Textbox(label="Folder path with WAV files", placeholder="e.g. D:/recordings/wavs")
                            run_rename_button = gr.Button("Run Batch Renamer", variant="secondary")
                            rename_status = gr.Textbox(label="Renamer output", lines=6)

                        run_rename_button.click(
                            fn=ui_batch_rename_folder,
                            inputs=rename_folder_input,
                            outputs=rename_status,
                        )

                    with gr.Column():
                        with gr.Group(elem_classes=["panel"]):
                            gr.Markdown("#### Batch Audio Preprocessor", elem_classes=["section-title"])
                            gr.Markdown("Trims silence, normalizes volume, resamples, and embeds RIFF metadata tags.")
                            preproc_in_folder = gr.Textbox(label="Input folder", placeholder="e.g. D:/raw_audio")
                            preproc_out_folder = gr.Textbox(label="Output folder", placeholder="e.g. D:/preprocessed_audio")
                            with gr.Row():
                                preproc_sr = gr.Dropdown(
                                    ["22050 Hz (Tacotron 2)", "24000 Hz (XTTS-v2)", "16000 Hz (Wav2Vec2)"],
                                    value="22050 Hz (Tacotron 2)",
                                    label="Target Sample Rate",
                                )
                                preproc_seq = gr.Checkbox(label="Sequential numbering (1.wav, 2.wav...)", value=True)
                            with gr.Row():
                                preproc_trim = gr.Checkbox(label="Trim silence (20 dB)", value=True)
                                preproc_norm = gr.Checkbox(label="Normalize volume", value=True)
                                preproc_tags = gr.Checkbox(label="Update metadata tags", value=True)
                            run_preproc_button = gr.Button("Run Audio Preprocessor", variant="primary")
                            preproc_status = gr.Textbox(label="Preprocessor output", lines=6)

                        run_preproc_button.click(
                            fn=ui_batch_preprocess_folder,
                            inputs=[
                                preproc_in_folder,
                                preproc_out_folder,
                                preproc_sr,
                                preproc_trim,
                                preproc_norm,
                                preproc_tags,
                                preproc_seq,
                            ],
                            outputs=preproc_status,
                        )

            with gr.Tab("4. Google Colab Guide"):
                gr.Markdown(
                    """
### 🚀 How to Train Your Voice Model on Google Colab

Follow these steps to train your custom voice model using the exported training archive:

#### Option A: Tacotron 2 Voice Model (From Video Tutorial)
1. **Launch the FakeYou Tacotron 2 Training Notebook**:
   - [FakeYou-Tacotron2-Notebook on GitHub](https://github.com/justinjohn0306/FakeYou-Tacotron2-Notebook)
   - [Open Directly in Google Colab](https://colab.research.google.com/github/justinjohn0306/FakeYou-Tacotron2-Notebook/blob/main/FakeYou_Tacotron_2_Training.ipynb)
2. **Connect to GPU Runtime**:
   - In Google Colab, go to **Runtime** > **Change runtime type** > Select **T4 GPU**.
3. **Upload Your Exported Dataset**:
   - In the **Export Training Dataset** tab of VoiceApp Studio, download the exported `.zip` file (e.g. `my_voice_tacotron2_training.zip`).
   - Upload it to Google Drive or directly to the Colab files panel.
4. **Extract and Train**:
   - Run the cells to extract the dataset. The files are already formatted with `wavs/` and LJSpeech pipe-formatted `metadata_train.txt` and `metadata_eval.txt`!
   - Follow the notebook prompts to begin training.
5. **Train HiFi-GAN Vocoder (Optional)**:
   - For ultra-realistic output, fine-tune HiFi-GAN with [FakeYou HiFi-GAN Notebook](https://github.com/justinjohn0306/FakeYou-Tacotron2-Notebook/blob/main/FakeYou_HiFi_GAN_Fine_Tuning.ipynb).

#### Option B: Coqui XTTS-v2 Fine-Tuning
1. Export with **Coqui XTTS-v2** format.
2. Upload the exported `.zip` to Colab or run locally.
3. Place the resulting `best_model.pth` and `config.json` in:
   `users/{profile}/models/{profile}_xtts_finetuned/`
4. VoiceApp Studio will automatically detect and load your fine-tuned model in the **Speech Generator** tab!
                    """
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
