"""
demo/app.py
Gradio interface for EmpathRAG - Empathetic Student Support System
"""

import sys
sys.path.insert(0, "src")

import gradio as gr
import json
import uuid
import datetime
import os
from pipeline.pipeline import EmpathRAGPipeline

# Constants
LABEL_NAMES = ["distress", "anxiety", "frustration", "neutral", "hopeful"]
LABEL_COLORS = {
    "distress":    "#e74c3c",
    "anxiety":     "#e67e22",
    "frustration": "#9b59b6",
    "neutral":     "#95a5a6",
    "hopeful":     "#27ae60",
}
LOG_PATH = "eval/human_eval_log.jsonl"

# Initialize pipeline (runs once at module load)
print("[Demo] Initialising EmpathRAG pipeline...")
pipeline = EmpathRAGPipeline(use_real_guardrail=True, guardrail_threshold=0.5)
print("[Demo] Pipeline ready.")

# Module-level state (not using gr.State)
emotion_history = []
session_id = ""


def new_session_id() -> str:
    """Generate 6-character alphanumeric session ID"""
    return uuid.uuid4().hex[:6].upper()


# Initialize session ID
session_id = new_session_id()


def log_turn(session_id, turn, user_message, result):
    """Append turn to human evaluation log (JSONL format)"""
    try:
        log_entry = {
            "session_id": session_id,
            "turn": turn,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "user_message": user_message,
            "response": result["response"],
            "emotion_label": result["emotion"],
            "emotion_name": result["emotion_name"],
            "trajectory": result["trajectory"],
            "crisis_fired": result["crisis"],
            "crisis_confidence": result["crisis_confidence"]
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[Warning] Failed to log turn: {e}")


def format_emotion_timeline(history, trajectory) -> str:
    """Format emotion timeline as HTML"""
    if not history:
        return "<div style='color:#888;font-size:13px;padding:8px;'>No emotions detected yet.</div>"

    trajectory_badge_colors = {
        "stable": "#95a5a6",
        "stable_positive": "#27ae60",
        "stable_negative": "#e74c3c",
        "escalating": "#c0392b",
        "de_escalating": "#16a085",
        "volatile": "#f39c12"
    }

    traj_color = trajectory_badge_colors.get(trajectory, "#95a5a6")
    html = f"<div style='margin-bottom:10px;padding:6px 10px;background:{traj_color};color:white;border-radius:4px;font-size:12px;font-weight:600;'>Session: {trajectory}</div>"
    html += "<div style='display:flex;flex-wrap:wrap;gap:6px;'>"

    for item in history:
        html += f"<span style='padding:4px 8px;background:{item['color']};color:white;border-radius:3px;font-size:11px;'>T{item['turn']}: {item['label_name']}</span>"

    html += "</div>"
    return html


def format_ig_panel(is_crisis, confidence, ig_tokens, loading) -> str:
    """Format Integrated Gradients crisis panel as HTML"""
    if not is_crisis:
        return "<div style='color:#888;font-size:13px;padding:8px;'>No crisis detected this session.</div>"

    if loading:
        return f"<div style='background:#fff3cd;border:1px solid #ffc107;padding:10px;border-radius:4px;'><div style='font-weight:600;color:#856404;margin-bottom:4px;'>Crisis signal detected - confidence: {confidence:.1%}</div><div style='color:#856404;font-size:12px;'>Computing token attributions...</div></div>"

    # Not loading, show full IG panel
    conf_pct = int(confidence * 100)
    html = f"<div style='background:#f8d7da;border:1px solid #e74c3c;padding:10px;border-radius:4px;'>"
    html += f"<div style='font-weight:600;color:#721c24;margin-bottom:8px;'>Crisis Confidence: {confidence:.1%}</div>"
    html += f"<div style='background:#fff;height:8px;border-radius:4px;overflow:hidden;margin-bottom:10px;'><div style='background:#e74c3c;height:100%;width:{conf_pct}%;'></div></div>"

    if ig_tokens:
        # Filter out empty/whitespace tokens
        valid_tokens = [(tok, score) for tok, score in ig_tokens if tok.strip()]
        if valid_tokens:
            max_score = max(score for _, score in valid_tokens)
            html += "<div style='font-size:11px;color:#721c24;margin-bottom:4px;font-weight:600;'>Top Crisis Signals:</div>"
            html += "<div style='display:flex;flex-wrap:wrap;gap:4px;'>"
            for tok, score in valid_tokens[:10]:
                opacity = score / max_score if max_score > 0 else 0.5
                bg_color = f"rgba(231,76,60,{opacity:.2f})"
                html += f"<span style='padding:2px 6px;background:{bg_color};border:1px solid #e74c3c;border-radius:3px;font-size:10px;'>{tok}</span>"
            html += "</div>"

    html += "</div>"
    return html


def respond(message, chat_history):
    """
    Generator function - yields UI state after each update.
    Yields tuple of 5 values: (chatbot, timeline_html, trajectory, crisis_html, session_id)
    """
    global emotion_history, session_id

    # Validate input
    if not message.strip():
        yield (chat_history,
               format_emotion_timeline(emotion_history, pipeline.tracker.trajectory()),
               pipeline.tracker.trajectory(),
               format_ig_panel(False, 0.0, [], False),
               session_id)
        return

    # Fast first pass - skip IG computation
    original_check = pipeline.guardrail.check
    def fast_check(text, threshold=0.5, skip_ig=False):
        return original_check(text, threshold=threshold, skip_ig=True)
    pipeline.guardrail.check = fast_check

    result = pipeline.run(message)

    # Restore original guardrail check immediately
    pipeline.guardrail.check = original_check

    # Update chat history
    chat_history.append((message, result["response"]))

    # Update emotion history
    emotion_history.append({
        "turn": len(emotion_history) + 1,
        "label_name": result["emotion_name"],
        "color": LABEL_COLORS[result["emotion_name"]]
    })

    # Log turn
    log_turn(session_id, len(emotion_history), message, result)

    # Format timeline
    timeline_html = format_emotion_timeline(emotion_history, result["trajectory"])

    if result["crisis"]:
        # First yield: show loading state
        yield (chat_history,
               timeline_html,
               result["trajectory"],
               format_ig_panel(True, result["crisis_confidence"], [], loading=True),
               session_id)

        # Compute real IG
        _, confidence, ig_tokens = pipeline.guardrail.check(message, threshold=0.5, skip_ig=False)

        # Second yield: show full IG panel
        yield (chat_history,
               timeline_html,
               result["trajectory"],
               format_ig_panel(True, confidence, ig_tokens, loading=False),
               session_id)
    else:
        # Single yield for non-crisis
        yield (chat_history,
               timeline_html,
               result["trajectory"],
               format_ig_panel(False, 0.0, [], False),
               session_id)


def reset_session_handler():
    """Reset session - returns 5 values matching respond() outputs"""
    global emotion_history, session_id

    emotion_history = []
    pipeline.reset_session()
    session_id = new_session_id()

    placeholder_timeline = "<div style='color:#888;font-size:13px;padding:8px;'>No emotions detected yet.</div>"
    placeholder_crisis = "<div style='color:#888;font-size:13px;padding:8px;'>No crisis detected this session.</div>"

    return ([], placeholder_timeline, "stable", placeholder_crisis, session_id)


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), title="EmpathRAG Demo") as demo:
    gr.Markdown("""
    # EmpathRAG - Empathetic Student Support
    Emotion-aware conversational support system for graduate students
    """)

    session_id_box = gr.Textbox(
        label="Session ID (use this in the feedback form)",
        interactive=False,
        value=session_id
    )

    with gr.Row():
        # Left column - chat interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=420, bubble_full_width=False)
            msg_box = gr.Textbox(
                placeholder="How are you feeling today?",
                label="",
                autofocus=True
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                reset_btn = gr.Button("Reset Session")

        # Right column - emotion tracking and crisis panel
        with gr.Column(scale=1):
            gr.Markdown("### Emotion Timeline")
            timeline_out = gr.HTML(value="<div style='color:#888;font-size:13px;padding:8px;'>No emotions detected yet.</div>")
            trajectory_out = gr.Textbox(label="Trajectory", value="stable", interactive=False)

            gr.Markdown("### Safety Guardrail")
            crisis_out = gr.HTML(value="<div style='color:#888;font-size:13px;padding:8px;'>No crisis detected this session.</div>")

    # Wire up interactions
    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, timeline_out, trajectory_out, crisis_out, session_id_box]
    ).then(
        lambda: "",
        outputs=msg_box
    )

    send_btn.click(
        respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, timeline_out, trajectory_out, crisis_out, session_id_box]
    ).then(
        lambda: "",
        outputs=msg_box
    )

    reset_btn.click(
        reset_session_handler,
        outputs=[chatbot, timeline_out, trajectory_out, crisis_out, session_id_box]
    )


if __name__ == "__main__":
    os.makedirs("eval", exist_ok=True)
    demo.launch(share=True)
