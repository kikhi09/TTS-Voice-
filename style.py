import gradio as gr

APP_THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="md",
    spacing_size="md",
).set(
    body_background_fill="#070d1a",
    body_text_color="#f8fafc",
    block_background_fill="#1d293a",
    block_border_color="#34445a",
    block_shadow="none",
    button_primary_background_fill="#0f766e",
    button_primary_background_fill_hover="#115e59",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#eef7f6",
    input_background_fill="#fbfcfc",
)

APP_CSS = """
.gradio-container {
    max-width: none !important;
    margin: 0 !important;
    padding: 22px 26px 32px 334px !important;
    box-sizing: border-box !important;
}

.app-shell {
    min-height: 100vh;
    box-sizing: border-box;
    padding: 0;
    background: #070d1a;
}

.hero {
    display: none;
}

.hero h1 {
    margin: 0 0 4px;
    font-size: 28px;
    line-height: 1.1;
    letter-spacing: 0;
}

.hero p {
    margin: 0;
    max-width: 760px;
    color: #52616b;
    font-size: 16px;
}

.hero-badge {
    min-width: 144px;
    padding: 10px 14px;
    border: 1px solid #d9e1e2;
    border-radius: 8px;
    background: #0b0c0d;
    color: #ffffff;
    font-weight: 700;
    text-align: center;
}

.sidebar {
    position: fixed !important;
    top: 0;
    left: 0;
    z-index: 20;
    width: 304px;
    height: 100vh;
    overflow-y: auto;
    padding: 20px 14px !important;
    border: 0 !important;
    border-right: 1px solid #34445a !important;
    border-radius: 0 !important;
    background: #1d293a !important;
    box-shadow: none !important;
}

.sidebar .block,
.sidebar .form,
.sidebar .gradio-column,
.sidebar .gradio-row {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.brand-mark {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
    padding: 0 8px;
    color: #f8fafc;
    font-size: 22px;
    font-weight: 800;
}

.brand-dot {
    display: inline-flex;
    width: 28px;
    height: 28px;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: #0f766e;
    color: #ffffff;
    font-size: 15px;
}

.side-section-title {
    margin: 18px 8px 8px;
    color: #9aa8bd;
    font-size: 13px;
    font-weight: 800;
}

.side-nav {
    display: grid;
    gap: 0;
    margin-top: 12px;
    border: 1px solid #34445a;
    border-radius: 8px;
    overflow: hidden;
}

.side-nav button {
    width: 100%;
    justify-content: flex-start !important;
    min-height: 42px !important;
    border: 0 !important;
    border-bottom: 1px solid #34445a !important;
    border-radius: 0 !important;
    background: #243348 !important;
    color: #f8fafc !important;
    box-shadow: none !important;
    font-weight: 700 !important;
    text-align: left !important;
}

.side-nav button:hover {
    background: #0f766e !important;
    color: #ffffff !important;
}

.profile-none-option label:has(input[value="NONE"]) {
    background: #243348 !important;
    color: #9aa8bd !important;
    border: 1px solid #34445a !important;
}

.profile-none-option label:has(input[value="NONE"][checked]),
.profile-none-option label:has(input[value="NONE"]:checked) {
    background: #2b384c !important;
    color: #e5e7eb !important;
}

.profile-none-option label:has(input[value="MY VOICE"]),
.profile-none-option label:has(input[value="my_voice"]),
.profile-none-option label:has(input[value="my voice"]),
.profile-none-option label:has(input[value="VOICE 2"]),
.profile-none-option label:has(input[value="voice_2"]),
.profile-none-option label:has(input[value="voice 2"]),
.profile-none-option label:has(input[value="MY VOICE"]) span,
.profile-none-option label:has(input[value="my_voice"]) span,
.profile-none-option label:has(input[value="my voice"]) span,
.profile-none-option label:has(input[value="VOICE 2"]) span,
.profile-none-option label:has(input[value="voice_2"]) span,
.profile-none-option label:has(input[value="voice 2"]) span {
    text-transform: uppercase !important;
}

.main-workspace {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: #070d1a !important;
    box-shadow: none !important;
}

.main-workspace > div,
.workspace-page {
    width: 100% !important;
    max-width: none !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.main-workspace > .block,
.main-workspace > .form,
.main-workspace > div > .block,
.main-workspace > div > .form,
.workspace-page > .block,
.workspace-page > .form,
.workspace-page > div:not(.profile-section):not(.panel) {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.workspace-page {
    min-height: 0;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.workspace-page[style*="display: none"],
.workspace-page.hidden,
.workspace-page.hide,
.hide.workspace-page,
.hidden.workspace-page {
    display: none !important;
    min-height: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

.workspace-page .gradio-row,
.workspace-page .gradio-column,
.workspace-page .gradio-group,
.workspace-page .form,
.workspace-page .block {
    max-width: none !important;
}

.workspace-page .dataframe,
.workspace-page table {
    width: 100% !important;
}

.gradio-container > .gap,
.main-workspace > .empty,
.workspace-page > .empty,
.workspace-page > div:empty,
.profile-section > div:empty {
    display: none !important;
}

.profile-section {
    width: 100% !important;
    margin-bottom: 18px;
    padding: 14px !important;
    border: 1px solid rgba(226, 232, 240, 0.32) !important;
    border-radius: 8px !important;
    background: #1d293a !important;
    color: #f8fafc !important;
    box-shadow: none !important;
}

.profile-section .block,
.profile-section .form,
.profile-section .gradio-row,
.profile-section .gradio-column {
    border: 0 !important;
    background: #1d293a !important;
    color: #f8fafc !important;
    box-shadow: none !important;
}

.profile-section input,
.profile-section textarea,
.profile-section select {
    background: #34445a !important;
    color: #f8fafc !important;
    border: 0 !important;
}

.profile-section .gradio-row {
    gap: 12px !important;
}

.profile-section textarea,
.panel textarea,
.result-box textarea,
.status-box textarea {
    background: #34445a !important;
    color: #f8fafc !important;
    border: 0 !important;
}

.profile-section .wrap,
.panel .wrap {
    border-color: rgba(226, 232, 240, 0.32) !important;
}

.block:empty,
.form:empty {
    display: none !important;
}

.wide-output textarea {
    min-height: 220px !important;
}

.page-heading h2 {
    margin: 0 0 18px;
    padding: 10px 12px;
    font-size: 32px;
    line-height: 1.15;
    text-align: center;
    letter-spacing: 0;
    border-radius: 8px;
    background: #1d293a;
}

.section-title h2 {
    margin: 6px 0 12px;
    font-size: 21px;
    letter-spacing: 0;
}

.panel {
    border: 1px solid rgba(226, 232, 240, 0.32) !important;
    border-radius: 8px !important;
    background: #1d293a !important;
    color: #f8fafc !important;
    box-shadow: none !important;
    padding: 14px !important;
}

.workspace-page .block:not(.profile-section),
.workspace-page .form {
    border: 0 !important;
    background: #1d293a !important;
    color: #f8fafc !important;
    box-shadow: none !important;
}

.workspace-page input,
.workspace-page textarea,
.workspace-page select {
    background: #34445a !important;
    color: #f8fafc !important;
    border: 0 !important;
}

.workspace-page label,
.workspace-page .label-wrap,
.workspace-page .prose,
.workspace-page h1,
.workspace-page h2,
.workspace-page h3 {
    color: #f8fafc !important;
}

.workspace-page [data-testid="file-upload"],
.workspace-page .upload-container,
.workspace-page .file-preview,
.workspace-page .audio-container,
.workspace-page .wrap {
    border-color: rgba(226, 232, 240, 0.32) !important;
}

.workspace-page [data-testid="file-upload"] {
    border-style: dashed !important;
    border-width: 1px !important;
    border-color: rgba(226, 232, 240, 0.72) !important;
}

.workspace-page .gradio-row,
.workspace-page .gradio-column {
    border-color: transparent !important;
}

.transcript-grid .cell-menu,
.transcript-grid .add-row-button {
    display: none !important;
}

.workspace-page .cell-menu {
    display: none !important;
}

.tight-panel {
    padding: 10px !important;
}

.primary-action button {
    min-height: 46px;
    font-weight: 700;
}

.danger-action button {
    border-color: #fecaca !important;
    color: #991b1b !important;
}

.status-box textarea,
.result-box textarea {
    font-family: Consolas, "SFMono-Regular", Menlo, monospace;
    font-size: 13px;
}

.generator-stage {
    min-height: 500px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(148, 163, 184, 0.34);
    border-radius: 8px;
    background:
        radial-gradient(circle at 50% 38%, rgba(125, 211, 252, 0.20), rgba(45, 212, 191, 0.08) 34%, transparent 58%),
        linear-gradient(180deg, #101d2f 0%, #07101f 100%);
    box-shadow: inset 0 0 120px rgba(14, 165, 233, 0.10);
    overflow: hidden;
}

.bot-scene {
    position: relative;
    width: min(500px, 86vw);
    height: 455px;
    --mood-color: #67e8f9;
    --mood-soft: rgba(103, 232, 249, 0.22);
}

.bot-shadow {
    position: absolute;
    left: 50%;
    bottom: 38px;
    width: 350px;
    height: 28px;
    transform: translateX(-50%);
    border-radius: 999px;
    background: var(--mood-soft);
    filter: blur(14px);
}

.bot-halo,
.bot-halo::before,
.bot-halo::after {
    position: absolute;
    left: 50%;
    top: 190px;
    width: 380px;
    height: 380px;
    content: "";
    transform: translate(-50%, -50%);
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.18);
}

.bot-halo::before {
    width: 300px;
    height: 300px;
}

.bot-halo::after {
    width: 220px;
    height: 220px;
}

.bot-orbit {
    position: absolute;
    left: 50%;
    top: 190px;
    width: 336px;
    height: 336px;
    transform: translate(-50%, -50%);
    border-radius: 999px;
    border: 1px solid rgba(103, 232, 249, 0.16);
    opacity: 0;
}

.bot-orbit::after {
    position: absolute;
    top: -4px;
    left: 50%;
    width: 8px;
    height: 8px;
    content: "";
    border-radius: 999px;
    background: var(--mood-color);
    box-shadow: 0 0 22px var(--mood-color);
}

.bot-body {
    position: absolute;
    left: 50%;
    bottom: 78px;
    width: 92px;
    height: 126px;
    transform: translateX(-50%);
    border-radius: 32px 32px 24px 24px;
    background: linear-gradient(145deg, #dbe7eb, #8aa2ab 58%, #4f6872);
    box-shadow: inset -10px -12px 22px rgba(0, 0, 0, 0.16);
}

.bot-head {
    position: absolute;
    left: 50%;
    top: 62px;
    width: 340px;
    height: 228px;
    transform: translateX(-50%) rotate(-1deg);
    border-radius: 92px;
    background: linear-gradient(145deg, #f3fbff 0%, #b8ccd3 44%, #627983 100%);
    box-shadow:
        inset 18px 22px 34px rgba(255, 255, 255, 0.52),
        inset -24px -28px 36px rgba(15, 23, 42, 0.18),
        0 28px 64px rgba(14, 165, 233, 0.18);
}

.bot-antenna {
    position: absolute;
    left: 50%;
    top: 12px;
    width: 12px;
    height: 62px;
    transform: translateX(-50%);
    border-radius: 999px;
    background: linear-gradient(180deg, #cbd5e1, #64748b);
}

.bot-antenna::before {
    position: absolute;
    left: 50%;
    top: -22px;
    width: 34px;
    height: 34px;
    content: "";
    transform: translateX(-50%);
    border-radius: 999px;
    background: radial-gradient(circle at 35% 30%, #fff 0 12%, var(--mood-color) 34%, #0f172a 88%);
    box-shadow: 0 0 24px var(--mood-soft);
}

.bot-face {
    position: absolute;
    left: 50%;
    top: 56px;
    width: 238px;
    height: 124px;
    transform: translateX(-50%);
    border: 1px solid rgba(103, 232, 249, 0.16);
    border-radius: 50px;
    background:
        radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.52), transparent 70%),
        linear-gradient(145deg, #111827, #020617);
    box-shadow:
        inset 8px 12px 20px rgba(255, 255, 255, 0.05),
        inset -10px -12px 18px rgba(0, 0, 0, 0.42),
        0 0 30px var(--mood-soft);
}

.bot-eye {
    position: absolute;
    top: 40px;
    width: 42px;
    height: 42px;
    border-radius: 14px;
    background: var(--mood-color);
    box-shadow: 0 0 18px var(--mood-color);
}

.bot-eye.left {
    left: 58px;
}

.bot-eye.right {
    right: 58px;
}

.bot-mouth {
    position: absolute;
    left: 50%;
    bottom: 25px;
    width: 64px;
    height: 24px;
    transform: translateX(-50%);
    border-bottom: 7px solid var(--mood-color);
    border-radius: 0 0 999px 999px;
    box-shadow: 0 12px 18px -18px var(--mood-color);
}

.bot-ear {
    position: absolute;
    top: 154px;
    width: 38px;
    height: 74px;
    border-radius: 20px;
    background: linear-gradient(145deg, #c7d7de, #6e8790);
    box-shadow: inset -8px -10px 16px rgba(0, 0, 0, 0.14);
}

.bot-ear.left {
    left: 42px;
}

.bot-ear.right {
    right: 42px;
}

.bot-arm {
    position: absolute;
    bottom: 112px;
    width: 28px;
    height: 58px;
    border-radius: 18px;
    background: linear-gradient(145deg, #bcd0d8, #5d7580);
}

.bot-arm.left {
    left: 154px;
    transform: rotate(8deg);
}

.bot-arm.right {
    right: 154px;
    transform: rotate(-8deg);
}

.bot-caption {
    position: absolute;
    left: 50%;
    bottom: 0;
    transform: translateX(-50%);
    min-width: 260px;
    padding: 12px 18px;
    border: 1px solid rgba(148, 163, 184, 0.34);
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.78);
    color: #f8fafc;
    text-align: center;
    font-weight: 700;
    letter-spacing: 0;
}

.bot-status-dots {
    position: absolute;
    left: 50%;
    bottom: 52px;
    display: flex;
    gap: 9px;
    transform: translateX(-50%);
}

.bot-status-dots span {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--mood-color);
    box-shadow: 0 0 16px var(--mood-color);
    opacity: 0.42;
    animation: dot-listen 1.4s ease-in-out infinite;
}

.bot-status-dots span:nth-child(2) { animation-delay: 0.18s; }
.bot-status-dots span:nth-child(3) { animation-delay: 0.36s; }

.voice-wave {
    position: absolute;
    left: 50%;
    bottom: 88px;
    display: flex;
    gap: 7px;
    align-items: end;
    transform: translateX(-50%);
    opacity: 0;
}

.voice-wave span {
    width: 6px;
    height: 18px;
    border-radius: 999px;
    background: var(--mood-color);
    box-shadow: 0 0 16px var(--mood-color);
    animation: voice-idle 1.8s ease-in-out infinite;
}

.voice-wave span:nth-child(2) { animation-delay: 0.12s; }
.voice-wave span:nth-child(3) { animation-delay: 0.24s; }
.voice-wave span:nth-child(4) { animation-delay: 0.36s; }
.voice-wave span:nth-child(5) { animation-delay: 0.48s; }

.bot-scene.generating .bot-orbit {
    opacity: 1;
    animation: orbit-spin 1.1s linear infinite;
}

.bot-scene.generating .bot-halo,
.bot-scene.generating .bot-halo::before,
.bot-scene.generating .bot-halo::after {
    animation: synth-pulse 1.35s ease-in-out infinite;
}

.bot-scene.generating .bot-head,
.bot-scene.talking .bot-head {
    animation: bot-talk 0.72s ease-in-out infinite;
}

.bot-scene.generating .bot-eye,
.bot-scene.talking .bot-eye {
    animation: eye-pulse 0.9s ease-in-out infinite;
}

.bot-scene.generating .voice-wave,
.bot-scene.talking .voice-wave {
    opacity: 1;
}

.bot-scene.generating .bot-status-dots,
.bot-scene.talking .bot-status-dots {
    opacity: 0;
}

.bot-scene.generating .voice-wave span {
    animation: voice-talk 0.62s ease-in-out infinite;
}

.bot-scene.talking .voice-wave span {
    animation: voice-talk 0.48s ease-in-out infinite;
}

@keyframes bot-talk {
    0%, 100% { transform: translateX(-50%) rotate(-1deg) translateY(0); }
    50% { transform: translateX(-50%) rotate(1deg) translateY(-7px); }
}

@keyframes eye-pulse {
    0%, 100% { filter: brightness(1); }
    50% { filter: brightness(1.35); }
}

@keyframes voice-idle {
    0%, 100% { height: 12px; opacity: 0.45; }
    50% { height: 24px; opacity: 0.85; }
}

@keyframes voice-talk {
    0%, 100% { height: 14px; }
    50% { height: 44px; }
}

@keyframes dot-listen {
    0%, 100% { transform: translateY(0); opacity: 0.35; }
    50% { transform: translateY(-6px); opacity: 1; }
}

@keyframes synth-pulse {
    0%, 100% { opacity: 0.24; transform: translate(-50%, -50%) scale(1); }
    50% { opacity: 0.7; transform: translate(-50%, -50%) scale(1.04); }
}

@keyframes orbit-spin {
    from { transform: translate(-50%, -50%) rotate(0deg); }
    to { transform: translate(-50%, -50%) rotate(360deg); }
}

.bot-scene[data-emotion="neutral"] {
    --mood-color: #67e8f9;
    --mood-soft: rgba(103, 232, 249, 0.22);
}

.bot-scene[data-emotion="happy"] {
    --mood-color: #5eead4;
    --mood-soft: rgba(94, 234, 212, 0.24);
}

.bot-scene[data-emotion="happy"] .bot-head {
    background: linear-gradient(145deg, #fff8d5 0%, #b8dcc9 52%, #5e8175 100%);
}

.bot-scene[data-emotion="happy"] .bot-eye {
    height: 30px;
    border-radius: 999px 999px 8px 8px;
}

.bot-scene[data-emotion="happy"] .bot-mouth {
    width: 76px;
    height: 28px;
    border-bottom-width: 8px;
}

.bot-scene[data-emotion="sad"] {
    --mood-color: #93c5fd;
    --mood-soft: rgba(147, 197, 253, 0.24);
}

.bot-scene[data-emotion="sad"] .bot-head {
    background: linear-gradient(145deg, #e7f0ff 0%, #9eb5cf 56%, #536a83 100%);
}

.bot-scene[data-emotion="sad"] .bot-eye {
    top: 48px;
    height: 18px;
    border-radius: 999px;
}

.bot-scene[data-emotion="sad"] .bot-mouth {
    bottom: 20px;
    border-top: 7px solid var(--mood-color);
    border-bottom: 0;
    border-radius: 999px 999px 0 0;
}

.bot-scene[data-emotion="angry"] {
    --mood-color: #fb7185;
    --mood-soft: rgba(251, 113, 133, 0.28);
}

.bot-scene[data-emotion="angry"] .bot-head {
    background: linear-gradient(145deg, #ffe1df 0%, #b7827d 44%, #653531 100%);
}

.bot-scene[data-emotion="angry"] .bot-face {
    background:
        radial-gradient(circle at 50% 50%, rgba(127, 29, 29, 0.58), transparent 66%),
        linear-gradient(145deg, #2a1014, #05070c);
}

.bot-scene[data-emotion="angry"] .bot-eye {
    top: 44px;
    width: 54px;
    height: 16px;
    border-radius: 999px;
}

.bot-scene[data-emotion="angry"] .bot-eye.left {
    transform: rotate(16deg);
}

.bot-scene[data-emotion="angry"] .bot-eye.right {
    transform: rotate(-16deg);
}

.bot-scene[data-emotion="angry"] .bot-mouth {
    bottom: 18px;
    width: 70px;
    height: 26px;
    border-top: 7px solid var(--mood-color);
    border-bottom: 0;
    border-radius: 999px 999px 0 0;
}

.bot-scene[data-emotion="excited"] {
    --mood-color: #c4b5fd;
    --mood-soft: rgba(196, 181, 253, 0.28);
}

.bot-scene[data-emotion="excited"] .bot-head {
    background: linear-gradient(145deg, #edfaff 0%, #a5b4fc 50%, #4f46e5 100%);
}

.bot-scene[data-emotion="excited"] .bot-eye {
    width: 36px;
    height: 36px;
    transform: rotate(45deg);
}

.bot-scene[data-emotion="excited"] .bot-mouth {
    width: 58px;
    height: 34px;
    border: 7px solid var(--mood-color);
    border-top: 0;
    border-radius: 0 0 999px 999px;
}

.orb-scene {
    position: relative;
    width: min(560px, 88vw);
    height: 430px;
    display: flex;
    align-items: center;
    justify-content: center;
    --orb-color: #67e8f9;
    --orb-secondary: #a78bfa;
    --orb-soft: rgba(103, 232, 249, 0.22);
}

.orb-scene[data-emotion="happy"] {
    --orb-color: #5eead4;
    --orb-secondary: #fef08a;
    --orb-soft: rgba(94, 234, 212, 0.24);
}

.orb-scene[data-emotion="sad"] {
    --orb-color: #93c5fd;
    --orb-secondary: #c4b5fd;
    --orb-soft: rgba(147, 197, 253, 0.24);
}

.orb-scene[data-emotion="angry"] {
    --orb-color: #fb7185;
    --orb-secondary: #f97316;
    --orb-soft: rgba(251, 113, 133, 0.30);
}

.orb-scene[data-emotion="excited"] {
    --orb-color: #c084fc;
    --orb-secondary: #22d3ee;
    --orb-soft: rgba(192, 132, 252, 0.28);
}

.orb-aura {
    position: absolute;
    left: 50%;
    top: 50%;
    width: 360px;
    height: 360px;
    transform: translate(-50%, -50%);
    border-radius: 999px;
    background:
        radial-gradient(circle, var(--orb-soft) 0 22%, transparent 62%),
        conic-gradient(from 90deg, transparent, rgba(125, 211, 252, 0.18), transparent, rgba(167, 139, 250, 0.16), transparent);
    filter: blur(2px);
    opacity: 0.8;
    animation: orb-breathe 4s ease-in-out infinite;
}

.voice-orb {
    position: relative;
    width: 230px;
    height: 230px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.42);
    background:
        radial-gradient(circle at 32% 24%, rgba(255, 255, 255, 0.84) 0 5%, transparent 18%),
        radial-gradient(circle at 50% 44%, rgba(255, 255, 255, 0.20) 0 8%, transparent 48%),
        linear-gradient(145deg, rgba(15, 23, 42, 0.24), rgba(15, 23, 42, 0.76));
    box-shadow:
        inset 0 0 32px rgba(255, 255, 255, 0.28),
        inset 0 -22px 42px rgba(15, 23, 42, 0.52),
        0 0 46px var(--orb-soft),
        0 36px 90px rgba(0, 0, 0, 0.36);
    overflow: hidden;
}

.voice-orb::before,
.voice-orb::after {
    position: absolute;
    left: -20%;
    right: -20%;
    content: "";
    border-radius: 50%;
    background: transparent;
    filter: drop-shadow(0 0 12px var(--orb-color));
}

.voice-orb::before {
    top: 72px;
    height: 96px;
    border-top: 13px solid var(--orb-color);
    transform: rotate(-7deg);
    animation: orb-wave-top 2.1s ease-in-out infinite;
}

.voice-orb::after {
    top: 96px;
    height: 84px;
    border-top: 10px solid var(--orb-secondary);
    transform: rotate(8deg);
    opacity: 0.84;
    animation: orb-wave-bottom 2.4s ease-in-out infinite;
}

.orb-core-line {
    position: absolute;
    left: 22%;
    right: 22%;
    top: 50%;
    height: 2px;
    transform: translateY(-50%);
    background: linear-gradient(90deg, transparent, var(--orb-color), #fff, var(--orb-secondary), transparent);
    box-shadow: 0 0 24px var(--orb-color);
    opacity: 0.85;
}

.orb-particles span {
    position: absolute;
    width: 3px;
    height: 3px;
    border-radius: 999px;
    background: var(--orb-color);
    box-shadow: 0 0 12px var(--orb-color);
    opacity: 0.72;
    animation: particle-float 3.4s ease-in-out infinite;
}

.orb-particles span:nth-child(1) { left: 42%; top: 32%; animation-delay: 0s; }
.orb-particles span:nth-child(2) { left: 60%; top: 44%; animation-delay: 0.4s; }
.orb-particles span:nth-child(3) { left: 50%; top: 63%; animation-delay: 0.8s; }
.orb-particles span:nth-child(4) { left: 34%; top: 54%; animation-delay: 1.2s; }
.orb-particles span:nth-child(5) { left: 66%; top: 58%; animation-delay: 1.6s; }

.orb-caption {
    position: absolute;
    left: 50%;
    bottom: 28px;
    min-width: 270px;
    transform: translateX(-50%);
    padding: 12px 18px;
    border: 1px solid rgba(148, 163, 184, 0.34);
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.78);
    color: #f8fafc;
    text-align: center;
    font-size: 15px;
    font-weight: 800;
}

.orb-scene.generating .orb-aura {
    animation: orb-spin 1.8s linear infinite, orb-breathe 1.2s ease-in-out infinite;
}

.orb-scene.generating .voice-orb,
.orb-scene.talking .voice-orb {
    animation: orb-speaking 0.9s ease-in-out infinite;
}

.orb-scene.generating .voice-orb::before,
.orb-scene.talking .voice-orb::before {
    animation: orb-wave-top 0.85s ease-in-out infinite;
}

.orb-scene.generating .voice-orb::after,
.orb-scene.talking .voice-orb::after {
    animation: orb-wave-bottom 0.75s ease-in-out infinite;
}

.speech-toggle {
    margin-top: 4px !important;
}

.speech-toggle label {
    width: 100%;
    min-height: 46px;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    padding: 10px 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 8px !important;
    background: rgba(15, 23, 42, 0.28) !important;
    cursor: pointer !important;
}

.workspace-page input[type="checkbox"] {
    appearance: auto !important;
    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;
    accent-color: #14b8a6 !important;
    cursor: pointer !important;
}

@keyframes orb-breathe {
    0%, 100% { transform: translate(-50%, -50%) scale(0.96); opacity: 0.66; }
    50% { transform: translate(-50%, -50%) scale(1.05); opacity: 1; }
}

@keyframes orb-spin {
    from { rotate: 0deg; }
    to { rotate: 360deg; }
}

@keyframes orb-speaking {
    0%, 100% { transform: scale(1); filter: brightness(1); }
    50% { transform: scale(1.035); filter: brightness(1.18); }
}

@keyframes orb-wave-top {
    0%, 100% { transform: translateY(0) rotate(-7deg) scaleX(0.94); }
    50% { transform: translateY(-12px) rotate(-2deg) scaleX(1.08); }
}

@keyframes orb-wave-bottom {
    0%, 100% { transform: translateY(0) rotate(8deg) scaleX(1.02); }
    50% { transform: translateY(10px) rotate(2deg) scaleX(0.9); }
}

@keyframes particle-float {
    0%, 100% { transform: translateY(0); opacity: 0.34; }
    50% { transform: translateY(-12px); opacity: 1; }
}

.speech-control-panel {
    min-height: 520px;
}

.gradio-container table {
    font-size: 13px;
}

@media (max-width: 760px) {
    .app-shell {
        padding: 16px;
    }

    .gradio-container {
        padding: 16px !important;
    }

    .sidebar {
        position: relative !important;
        width: auto;
        height: auto;
        margin-bottom: 16px;
        border-right: 0 !important;
        border-radius: 8px !important;
    }

    .hero {
        align-items: flex-start;
        flex-direction: column;
        padding: 20px;
    }

    .hero-badge {
        width: 100%;
        text-align: left;
    }
}
"""


