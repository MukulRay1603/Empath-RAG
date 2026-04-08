"""
demo/app.py
EmpathRAG — Full Gradio Demo

IG ASYNC PATTERN:
When the guardrail fires (crisis detected), the safe response must appear in <1 second.
Integrated Gradients (IG) attribution takes ~30-45 seconds on CPU.

Implementation:
1. Monkey-patch pipeline.guardrail.check to force skip_ig=True during pipeline.run()
2. First yield: safe response + loading placeholder in IG panel
3. Run real IG check (skip_ig=False) in same generator after yielding
4. Second yield: same response + populated IG panel with token highlights

This uses a generator function with conditional double-yield.
Gradio 4.21 doesn't have gr.Timer, so we use synchronous IG after first yield.
The user sees the safe response instantly; IG completes in background of same request.
"""

import sys
sys.path.insert(0, "src")

import gradio as gr
from pipeline.pipeline import EmpathRAGPipeline

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_NAMES = ["distress", "anxiety", "frustration", "neutral", "hopeful"]
LABEL_COLORS = {
    "distress":    "#e74c3c",
    "anxiety":     "#e67e22",
    "frustration": "#9b59b6",
    "neutral":     "#95a5a6",
    "hopeful":     "#27ae60",
}

# ── Global State ──────────────────────────────────────────────────────────────

print("[Demo] Initialising EmpathRAG pipeline...")
pipeline = EmpathRAGPipeline(use_real_guardrail=True, guardrail_threshold=0.5)
print("[Demo] Pipeline ready.")

emotion_history = []  # List of {turn: int, label_name: str, color: str}

# ── HTML Formatters ───────────────────────────────────────────────────────────

def format_emotion_timeline(history: list) -> str:
    """Returns HTML for emotion timeline pills."""
    if not history:
        return "<div style='color:#888;font-size:13px;padding:8px;'>No emotions detected yet.</div>"

    trajectory = pipeline.tracker.trajectory()
    trajectory_badge_colors = {
        "stable": "#95a5a6",
        "stable_positive": "#27ae60",
        "stable_negative": "#e74c3c",
        "escalating": "#c0392b",
        "de_escalating": "#16a085",
        "volatile": "#f39c12",
    }
    traj_color = trajectory_badge_colors.get(trajectory, "#95a5a6")

    header = f"""
    <div style='margin-bottom:8px;padding:6px;background:{traj_color}20;border-left:3px solid {traj_color};border-radius:4px;'>
        <span style='font-size:12px;color:{traj_color};font-weight:600;'>Session: {trajectory.replace('_', ' ').title()}</span>
    </div>
    """

    pills = []
    for entry in history:
        pill = f"""
        <span style='
            display:inline-block;
            background:{entry["color"]};
            color:white;
            font-size:12px;
            padding:4px 10px;
            margin:3px;
            border-radius:12px;
            white-space:nowrap;
        '>T{entry["turn"]}: {entry["label_name"]}</span>
        """
        pills.append(pill)

    pills_html = f"<div style='display:flex;flex-wrap:wrap;gap:4px;'>{''.join(pills)}</div>"
    return header + pills_html


def format_ig_panel(is_crisis: bool, confidence: float, ig_tokens: list, loading: bool) -> str:
    """Returns HTML for the crisis + IG attribution panel."""
    if not is_crisis:
        return "<div style='color:#888;font-size:13px;padding:8px;'>No crisis detected this session.</div>"

    if loading:
        # Loading state - IG running in background
        return f"""
        <div style='
            background:#fff3cd;
            border-left:4px solid #ffc107;
            padding:12px;
            border-radius:6px;
            margin-bottom:8px;
        '>
            <div style='font-weight:600;color:#856404;margin-bottom:6px;'>
                🚨 Crisis signal detected — confidence: {confidence:.1%}
            </div>
            <div style='font-family:monospace;font-size:12px;color:#856404;'>
                ⏳ Computing token attributions...
            </div>
        </div>
        """

    # Fully loaded - show confidence bar + token highlights
    conf_pct = int(confidence * 100)
    bar_color = "#e74c3c" if confidence >= 0.7 else "#f39c12"

    conf_bar = f"""
    <div style='margin-bottom:12px;'>
        <div style='font-size:13px;font-weight:600;color:#333;margin-bottom:4px;'>
            Crisis Confidence: {confidence:.1%}
        </div>
        <div style='background:#eee;border-radius:8px;overflow:hidden;height:20px;'>
            <div style='background:{bar_color};height:100%;width:{conf_pct}%;transition:width 0.3s;'></div>
        </div>
    </div>
    """

    if not ig_tokens:
        # IG was skipped or no tokens
        return f"""
        <div style='
            background:#f8d7da;
            border-left:4px solid #e74c3c;
            padding:12px;
            border-radius:6px;
        '>
            {conf_bar}
            <div style='font-size:12px;color:#721c24;'>
                Token attributions unavailable.
            </div>
        </div>
        """

    # Build token highlight spans
    # Filter out special tokens and compute max score for normalization
    filtered_tokens = [
        (tok, score) for tok, score in ig_tokens
        if not (tok.startswith("▁") and len(tok.strip("▁")) == 0)
    ]

    if not filtered_tokens:
        token_section = "<div style='font-size:12px;color:#721c24;'>No significant tokens.</div>"
    else:
        max_score = max(score for _, score in filtered_tokens)
        token_spans = []
        for tok, score in filtered_tokens[:10]:  # Top 10
            opacity = 0.25 + 0.75 * (score / max_score if max_score > 0 else 0)
            # Clean up token display - remove leading underscore for word pieces
            display_tok = tok.replace("▁", " ").strip()
            if not display_tok:
                continue
            span = f"""
            <span style='
                background:rgba(231,76,60,{opacity:.2f});
                padding:3px 6px;
                margin:2px;
                border-radius:4px;
                font-size:12px;
                font-family:monospace;
                display:inline-block;
            '>{display_tok}</span>
            """
            token_spans.append(span)

        token_section = f"""
        <div style='margin-top:8px;'>
            <div style='font-size:12px;font-weight:600;color:#721c24;margin-bottom:6px;'>
                Top Crisis Signals:
            </div>
            <div style='line-height:1.8;'>
                {''.join(token_spans)}
            </div>
        </div>
        """

    return f"""
    <div style='
        background:#f8d7da;
        border-left:4px solid #e74c3c;
        padding:12px;
        border-radius:6px;
    '>
        {conf_bar}
        {token_section}
    </div>
    """


# ── Core Logic ────────────────────────────────────────────────────────────────

def respond(message, chat_history):
    """
    Generator function that yields 1-2 times.

    Normal flow: single yield with all outputs.
    Crisis flow:
        - yield 1: instant safe response + IG loading panel
        - yield 2: same response + populated IG panel (after IG completes)
    """
    # Validate input
    if not message or not message.strip():
        yield (
            chat_history,
            format_emotion_timeline(emotion_history),
            pipeline.tracker.trajectory(),
            format_ig_panel(False, 0.0, [], False),
        )
        return

    # Monkey-patch guardrail to skip IG on first pass
    original_check = pipeline.guardrail.check
    def fast_check(text, threshold=0.5, skip_ig=False):
        return original_check(text, threshold=threshold, skip_ig=True)
    pipeline.guardrail.check = fast_check

    # Run pipeline with fast guardrail check
    result = pipeline.run(message)

    # Restore original guardrail
    pipeline.guardrail.check = original_check

    # Update chat history
    chat_history.append((message, result["response"]))

    # Update emotion timeline
    turn = len(emotion_history) + 1
    emotion_history.append({
        "turn": turn,
        "label_name": result["emotion_name"],
        "color": LABEL_COLORS[result["emotion_name"]],
    })

    timeline_html = format_emotion_timeline(emotion_history)
    trajectory_text = result["trajectory"]

    # Handle crisis vs non-crisis
    if result["crisis"]:
        # FIRST YIELD: Instant response with loading IG panel
        crisis_panel_loading = format_ig_panel(
            True,
            result["crisis_confidence"],
            [],
            loading=True,
        )
        yield (
            chat_history,
            timeline_html,
            trajectory_text,
            crisis_panel_loading,
        )

        # Run real IG check in foreground (blocking, but user already has response)
        # This takes ~30-45s on CPU
        _, confidence, ig_tokens = pipeline.guardrail.check(
            message,
            threshold=0.5,
            skip_ig=False,
        )

        # SECOND YIELD: Same response, populated IG panel
        crisis_panel_final = format_ig_panel(
            True,
            confidence,
            ig_tokens,
            loading=False,
        )
        yield (
            chat_history,
            timeline_html,
            trajectory_text,
            crisis_panel_final,
        )
    else:
        # Non-crisis: single yield
        crisis_panel = format_ig_panel(False, 0.0, [], False)
        yield (
            chat_history,
            timeline_html,
            trajectory_text,
            crisis_panel,
        )


def reset_session():
    """Clear session state."""
    global emotion_history
    emotion_history = []
    pipeline.reset_session()

    return (
        [],  # empty chat
        "<div style='color:#888;font-size:13px;padding:8px;'>No emotions detected yet.</div>",  # timeline
        "stable",  # trajectory
        "<div style='color:#888;font-size:13px;padding:8px;'>No crisis detected this session.</div>",  # crisis panel
    )


# ── Gradio Interface ──────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="EmpathRAG Demo") as demo:
    gr.Markdown("# 🧠 EmpathRAG — Empathetic Student Support Assistant")
    gr.Markdown("Real-time emotion detection, crisis intervention, and empathetic response generation.")

    with gr.Row():
        # Left column: Chat interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=450,
                bubble_full_width=False,
            )
            msg_box = gr.Textbox(
                placeholder="How are you feeling today?",
                label="",
                autofocus=True,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                reset_btn = gr.Button("Reset Session")

        # Right column: Dashboard
        with gr.Column(scale=1):
            gr.Markdown("## Session Dashboard")

            gr.Markdown("### 📊 Emotion Timeline")
            timeline_out = gr.HTML(
                value="<div style='color:#888;font-size:13px;padding:8px;'>No emotions detected yet.</div>"
            )

            trajectory_out = gr.Textbox(
                label="Trajectory",
                value="stable",
                interactive=False,
            )

            gr.Markdown("### 🛡️ Safety Guardrail")
            crisis_out = gr.HTML(
                value="<div style='color:#888;font-size:13px;padding:8px;'>No crisis detected this session.</div>"
            )

    # Wire up events
    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, timeline_out, trajectory_out, crisis_out],
    ).then(
        lambda: "",
        outputs=msg_box,
    )

    send_btn.click(
        respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, timeline_out, trajectory_out, crisis_out],
    ).then(
        lambda: "",
        outputs=msg_box,
    )

    reset_btn.click(
        reset_session,
        outputs=[chatbot, timeline_out, trajectory_out, crisis_out],
    )


if __name__ == "__main__":
    demo.launch(share=False)
