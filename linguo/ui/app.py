"""
ui/app.py — Gradio interface for Linguo

Tabs:
  1. Practice  — generate sentences, submit guesses, get hints
  2. Vocabulary — browse the word bank with mastery indicators
  3. Progress   — AI-generated session analysis
"""

from __future__ import annotations

import gradio as gr

from agents.orchestrator import Orchestrator
from config import SUPPORTED_LANGUAGES, TOPICS

orch = Orchestrator()


# ── Helper formatters ──────────────────────────────────────────────────────────

def _highlight_sentence(sentence: str, foreign_word: str) -> str:
    """Wrap the foreign word in a simple HTML span for display."""
    highlighted = sentence.replace(
        foreign_word,
        f'<span style="background:#dbeafe;color:#1e40af;padding:2px 8px;'
        f'border-radius:6px;font-weight:600;">{foreign_word}</span>',
    )
    return f'<p style="font-size:1.2rem;line-height:1.8">{highlighted}</p>'


def _vocab_html(state) -> str:
    if not state.vocab:
        return "<p style='color:gray'>No words yet — start practicing!</p>"
    rows = ""
    for word, rec in state.vocab.items():
        color = "#dcfce7" if rec.mastered else "#fef9c3" if rec.attempts > 0 else "#f1f5f9"
        label = "✓ mastered" if rec.mastered else f"{rec.attempts} attempts" if rec.attempts > 0 else "new"
        rows += (
            f'<div style="display:flex;justify-content:space-between;padding:8px 12px;'
            f'background:{color};border-radius:8px;margin-bottom:6px;">'
            f'<span><strong>{word}</strong> = {rec.meaning} <em>({rec.lang})</em></span>'
            f'<span style="font-size:0.85rem;color:#555">{label}</span>'
            f'</div>'
        )
    return rows


# ── Action handlers ────────────────────────────────────────────────────────────

def handle_generate(language: str, topic: str):
    sentence_obj, logs = orch.generate_sentence(language, topic)
    html = _highlight_sentence(sentence_obj.sentence, sentence_obj.foreign_word)
    state = orch.user_state
    log_text = "\n".join(logs)
    return (
        html,
        gr.update(visible=True),          # answer row
        gr.update(visible=False),         # feedback
        gr.update(visible=False),         # hint box
        "",                               # clear guess input
        f"Words seen: {state.total_seen}  |  Mastered: {state.mastered_count}  |  Streak: {state.streak}  |  Level: {state.level}",
        log_text,
    )


def handle_check(guess: str):
    if not guess.strip():
        return gr.update(), gr.update(), gr.update()
    result, logs = orch.check_answer(guess.strip())
    color = "#dcfce7" if result.correct else "#fee2e2"
    fb_html = f'<div style="background:{color};padding:12px;border-radius:8px">{result.feedback}</div>'
    state = orch.user_state
    stats = f"Words seen: {state.total_seen}  |  Mastered: {state.mastered_count}  |  Streak: {state.streak}  |  Level: {state.level}"
    return (
        gr.update(value=fb_html, visible=True),
        stats,
        "\n".join(logs),
    )


def handle_hint():
    hint, logs = orch.get_hint()
    hint_html = f'<div style="background:#ede9fe;padding:10px;border-radius:8px;color:#4c1d95">Hint: {hint}</div>'
    return gr.update(value=hint_html, visible=True), "\n".join(logs)


def handle_vocab():
    return _vocab_html(orch.user_state)


def handle_progress():
    analysis, logs = orch.get_progress()
    state = orch.user_state
    review = ", ".join(analysis.get("words_to_review", [])) or "none"
    adjust = analysis.get("difficulty_adjustment", "maintain")
    html = (
        f'<div style="padding:1rem">'
        f'<p style="font-size:1.1rem">{analysis.get("summary", "")}</p>'
        f'<hr style="margin:12px 0">'
        f'<p><strong>Words to review:</strong> {review}</p>'
        f'<p><strong>Difficulty recommendation:</strong> {adjust}</p>'
        f'<p><strong>Current level:</strong> {state.level}</p>'
        f'<p><strong>Mastered:</strong> {state.mastered_count} / {state.total_seen} words</p>'
        f'</div>'
    )
    return html, "\n".join(logs)


# ── Layout ─────────────────────────────────────────────────────────────────────

def launch_app():
    with gr.Blocks(title="Linguo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Linguo\n*Learn vocabulary through context — one word at a time*")

        with gr.Tabs():
            # ── Practice tab ──────────────────────────────────────────────────
            with gr.Tab("Practice"):
                with gr.Row():
                    lang_dd  = gr.Dropdown(SUPPORTED_LANGUAGES, value="Spanish", label="Language")
                    topic_dd = gr.Dropdown(TOPICS, value="everyday life", label="Topic")
                    gen_btn  = gr.Button("New sentence", variant="primary")

                stats_box = gr.Textbox(label="Session stats", interactive=False, lines=1)
                sentence_html = gr.HTML("<p style='color:gray'>Press 'New sentence' to begin.</p>")

                with gr.Row(visible=False) as answer_row:
                    guess_input = gr.Textbox(placeholder="Type the English meaning...", label="Your guess", scale=4)
                    check_btn   = gr.Button("Check", variant="primary", scale=1)
                    hint_btn    = gr.Button("Hint", scale=1)

                feedback_html = gr.HTML(visible=False)
                hint_html     = gr.HTML(visible=False)

                with gr.Accordion("Agent logs", open=False):
                    log_box = gr.Textbox(lines=6, interactive=False, label="")

                gen_btn.click(
                    handle_generate,
                    inputs=[lang_dd, topic_dd],
                    outputs=[sentence_html, answer_row, feedback_html, hint_html, guess_input, stats_box, log_box],
                )
                check_btn.click(
                    handle_check,
                    inputs=[guess_input],
                    outputs=[feedback_html, stats_box, log_box],
                )
                hint_btn.click(
                    handle_hint,
                    outputs=[hint_html, log_box],
                )

            # ── Vocabulary tab ────────────────────────────────────────────────
            with gr.Tab("Vocabulary"):
                refresh_btn = gr.Button("Refresh")
                vocab_html  = gr.HTML()
                refresh_btn.click(handle_vocab, outputs=[vocab_html])

            # ── Progress tab ──────────────────────────────────────────────────
            with gr.Tab("Progress"):
                analyze_btn    = gr.Button("Analyze my progress", variant="primary")
                progress_html  = gr.HTML()
                with gr.Accordion("Agent logs", open=False):
                    prog_log_box = gr.Textbox(lines=6, interactive=False, label="")
                analyze_btn.click(
                    handle_progress,
                    outputs=[progress_html, prog_log_box],
                )

    demo.launch()
