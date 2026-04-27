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
import threading
from html import escape
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
LOG_TURNS = os.getenv("EMPATHRAG_LOG_TURNS") == "1"
SHARE_DEMO = os.getenv("EMPATHRAG_SHARE") == "1"
RETRIEVAL_CORPUS = os.getenv("EMPATHRAG_RETRIEVAL_CORPUS", "auto")

# Initialize pipeline (runs once at module load)
print("[Demo] Initialising EmpathRAG pipeline...")
pipeline = EmpathRAGPipeline(
    use_real_guardrail=True,
    guardrail_threshold=0.5,
    retrieval_corpus=RETRIEVAL_CORPUS,
)
pipeline_lock = threading.Lock()
print("[Demo] Pipeline ready.")


def new_session_id() -> str:
    """Generate 6-character alphanumeric session ID"""
    return uuid.uuid4().hex[:6].upper()


def new_session_state() -> dict:
    return {
        "session_id": new_session_id(),
        "emotion_history": [],
        "tracker_history": [],
        "conv_history": [],
    }


def log_turn(session_id, turn, user_message, result):
    """Append turn to human evaluation log (JSONL format)"""
    if not LOG_TURNS:
        return
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


def format_retrieval_panel(result=None) -> str:
    """Format retrieval corpus and source metadata for the demo side panel."""
    if not result:
        return "<div style='color:#888;font-size:13px;padding:8px;'>No retrieval yet.</div>"

    safety_level = escape(str(result.get("safety_level", "unknown")))
    safety_reason = escape(str(result.get("safety_reason", "")))
    corpus = escape(str(result.get("retrieval_corpus", "unknown")))
    html = (
        "<div style='font-size:12px;line-height:1.35;'>"
        f"<div><strong>Corpus:</strong> {corpus}</div>"
        f"<div><strong>Safety:</strong> {safety_level}</div>"
        f"<div><strong>Reason:</strong> {safety_reason}</div>"
    )

    sources = result.get("retrieved_sources", [])
    if not sources:
        html += "<div style='color:#888;margin-top:8px;'>No sources retrieved.</div></div>"
        return html

    html += "<div style='margin-top:10px;font-weight:600;'>Top Sources</div>"
    for source in sources[:3]:
        title = escape(str(source.get("title", "") or "Untitled source"))
        source_name = escape(str(source.get("source_name", "") or "Unknown source"))
        topic = escape(str(source.get("topic", "") or ""))
        risk = escape(str(source.get("risk_level", "") or ""))
        url = escape(str(source.get("url", "") or ""))
        html += (
            "<div style='border-top:1px solid #ddd;padding-top:6px;margin-top:6px;'>"
            f"<div><strong>{title}</strong></div>"
            f"<div>{source_name}</div>"
            f"<div style='color:#666;'>topic={topic} · risk={risk}</div>"
        )
        if url:
            html += f"<div><a href='{url}' target='_blank'>source link</a></div>"
        html += "</div>"
    html += "</div>"
    return html


def respond(message, chat_history, session_state):
    """
    Generator function - yields UI state after each update.
    Yields chatbot, emotion timeline, trajectory, safety panel, retrieval panel,
    session ID, and per-user session state.
    """
    if not session_state:
        session_state = new_session_state()

    emotion_history = session_state["emotion_history"]
    session_id = session_state["session_id"]

    # Validate input
    if not message.strip():
        yield (chat_history,
               format_emotion_timeline(emotion_history, pipeline.tracker.trajectory()),
               pipeline.tracker.trajectory(),
               format_ig_panel(False, 0.0, [], False),
               format_retrieval_panel(),
               session_id,
               session_state)
        return

    with pipeline_lock:
        pipeline.tracker.reset()
        for label in session_state.get("tracker_history", []):
            pipeline.tracker.update(label, token_count=5)
        pipeline.conv_history = list(session_state.get("conv_history", []))

        # Fast first pass - skip IG computation
        original_check = pipeline.guardrail.check
        def fast_check(text, threshold=0.5, skip_ig=False):
            return original_check(text, threshold=threshold, skip_ig=True)
        pipeline.guardrail.check = fast_check

        result = pipeline.run(message)

        # Restore original guardrail check immediately
        pipeline.guardrail.check = original_check
        session_state["tracker_history"] = pipeline.tracker.history()
        session_state["conv_history"] = list(pipeline.conv_history)

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
               format_retrieval_panel(result),
               session_id,
               session_state)

        # Compute real IG
        with pipeline_lock:
            _, confidence, ig_tokens = pipeline.guardrail.check(message, threshold=0.5, skip_ig=False)

        # Second yield: show full IG panel
        yield (chat_history,
               timeline_html,
               result["trajectory"],
               format_ig_panel(True, confidence, ig_tokens, loading=False),
               format_retrieval_panel(result),
               session_id,
               session_state)
    else:
        # Single yield for non-crisis
        yield (chat_history,
               timeline_html,
               result["trajectory"],
               format_ig_panel(False, 0.0, [], False),
               format_retrieval_panel(result),
               session_id,
               session_state)


def reset_session_handler():
    """Reset session - returns 5 values matching respond() outputs"""
    session_state = new_session_state()

    placeholder_timeline = "<div style='color:#888;font-size:13px;padding:8px;'>No emotions detected yet.</div>"
    placeholder_crisis = "<div style='color:#888;font-size:13px;padding:8px;'>No crisis detected this session.</div>"
    placeholder_retrieval = "<div style='color:#888;font-size:13px;padding:8px;'>No retrieval yet.</div>"

    return ([], placeholder_timeline, "stable", placeholder_crisis, placeholder_retrieval, session_state["session_id"], session_state)


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), title="EmpathRAG Demo") as demo:
    initial_state = new_session_state()
    session_state = gr.State(value=initial_state)
    gr.Markdown("""
    # EmpathRAG - Empathetic Student Support
    Emotion-aware conversational support system for graduate students
    """)

    session_id_box = gr.Textbox(
        label="Session ID (use this in the feedback form)",
        interactive=False,
        value=initial_state["session_id"]
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
            gr.Markdown("### Retrieval Sources")
            retrieval_out = gr.HTML(value="<div style='color:#888;font-size:13px;padding:8px;'>No retrieval yet.</div>")

    # Wire up interactions
    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot, session_state],
        outputs=[chatbot, timeline_out, trajectory_out, crisis_out, retrieval_out, session_id_box, session_state]
    ).then(
        lambda: "",
        outputs=msg_box
    )

    send_btn.click(
        respond,
        inputs=[msg_box, chatbot, session_state],
        outputs=[chatbot, timeline_out, trajectory_out, crisis_out, retrieval_out, session_id_box, session_state]
    ).then(
        lambda: "",
        outputs=msg_box
    )

    reset_btn.click(
        reset_session_handler,
        outputs=[chatbot, timeline_out, trajectory_out, crisis_out, retrieval_out, session_id_box, session_state]
    )


if __name__ == "__main__":
    os.makedirs("eval", exist_ok=True)
    demo.launch(share=SHARE_DEMO)
