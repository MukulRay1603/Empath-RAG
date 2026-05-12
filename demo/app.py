"""
demo/app.py
Gradio interface for EmpathRAG V2.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
import sys
import threading
import uuid
from html import escape
from pathlib import Path

import gradio as gr

# Load .env (GROQ_API_KEY, ANTHROPIC_API_KEY, etc.) before any provider
# checks os.getenv. Soft import so the app still runs without python-dotenv.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

sys.path.insert(0, "src")

from pipeline.safety_policy import SafetyLevel, SafetyTriagePolicy
from pipeline.core import EmpathRAGCore
from pipeline.output_guard import validate_output
from pipeline.service_graph import match_services
from pipeline.v2_schema import (
    SafetyTier,
    SupportRoute,
    classify_route,
    map_safety_level,
)


LABEL_COLORS = {
    "distress": "#fb7185",
    "anxiety": "#f59e0b",
    "frustration": "#a78bfa",
    "neutral": "#94a3b8",
    "hopeful": "#34d399",
}

LOG_PATH = "eval/human_eval_log.jsonl"
LOG_TURNS = os.getenv("EMPATHRAG_LOG_TURNS") == "1"
SHARE_DEMO = os.getenv("EMPATHRAG_SHARE") == "1"
RETRIEVAL_CORPUS = os.getenv("EMPATHRAG_RETRIEVAL_CORPUS", "auto")
DEMO_TOP_K = int(os.getenv("EMPATHRAG_TOP_K", "5"))
DEMO_MAX_TOKENS = int(os.getenv("EMPATHRAG_MAX_TOKENS", "140"))
DEMO_BACKEND = os.getenv("EMPATHRAG_DEMO_BACKEND", "fast").strip().lower()
CURATED_DB_PATH = Path(os.getenv("EMPATHRAG_CURATED_DB", "data/curated/indexes/metadata_curated.db"))

APP_CSS = """
:root {
  --bg: #0a0c10;
  --bg-soft: #0d1017;
  --surface: #11151c;
  --surface-2: #161c25;
  --surface-3: #1d2531;
  --surface-glass: rgba(17,21,28,0.72);
  --border: rgba(255,255,255,0.06);
  --border-mid: rgba(255,255,255,0.10);
  --border-strong: rgba(255,255,255,0.16);
  --accent: #5eead4;
  --accent-dim: #2dd4bf;
  --accent-soft: rgba(94,234,212,0.10);
  --accent-line: rgba(94,234,212,0.22);
  --accent-glow: rgba(94,234,212,0.20);
  --accent-aurora: rgba(94,234,212,0.28);
  --text: #e7ecf2;
  --text-muted: #8a93a3;
  --text-dim: #5a6373;
  --text-faint: #424b5a;
  --warm: #f5b669;
  --warm-soft: rgba(245,182,105,0.10);
  --warm-line: rgba(245,182,105,0.22);
  --danger: #f87171;
  --danger-soft: rgba(248,113,113,0.10);
  --danger-line: rgba(248,113,113,0.22);
  --indigo: #818cf8;
  --indigo-soft: rgba(129,140,248,0.10);
  --indigo-line: rgba(129,140,248,0.22);
  --radius-sm: 8px;
  --radius: 12px;
  --radius-lg: 16px;
  --radius-xl: 22px;
}

* { box-sizing: border-box; }

html, body {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: "Inter", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  letter-spacing: -0.005em;
}

body::before {
  content: "";
  position: fixed; inset: 0;
  pointer-events: none; z-index: 0;
  background:
    radial-gradient(1100px 520px at 18% -10%, rgba(94,234,212,0.07), transparent 70%),
    radial-gradient(800px 400px at 100% 20%, rgba(129,140,248,0.045), transparent 70%),
    radial-gradient(720px 380px at 80% 110%, rgba(94,234,212,0.04), transparent 70%);
}

/* Natural Gradio flow. Container holds everything at its document-flow
   height; the page is allowed to be exactly as tall as its content. */
.gradio-container {
  position: relative; z-index: 1;
  background: transparent !important;
  max-width: 1320px !important;
  margin: 0 auto !important;
  padding: 0 32px 24px !important;
  color: var(--text) !important;
}

.gradio-container * { border-color: var(--border); }
.gradio-container label, .gradio-container .label-wrap {
  color: var(--text-muted) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
}
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .tabs,
.gradio-container .tabitem {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* TOP BAR — uses position:relative (not sticky) so it renders inside the
   HF Spaces iframe, where there is no scroll container for sticky to attach
   to. Solid background so brand text is always legible. */
.er-topbar {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 16px !important;
  padding: 14px 0 12px !important;
  margin: 0 0 16px !important;
  border-bottom: 1px solid var(--border) !important;
  flex-wrap: nowrap !important;
  position: relative;
  z-index: 100;
  background: var(--bg);
  height: 64px;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0;
  box-sizing: border-box;
}
.er-topbar > * { flex: none !important; }
.er-topbar > .er-mode-wrap { flex: 1 1 auto !important; display: flex; justify-content: center; }

.er-brand {
  display: flex; align-items: center; gap: 12px;
  font-weight: 600; font-size: 16px; letter-spacing: -0.012em;
  color: var(--text);
}
.er-brand-dot {
  width: 9px; height: 9px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 18px var(--accent-glow);
  animation: er-pulse 2.4s ease-in-out infinite;
  position: relative;
}
.er-brand-dot::after {
  content: "";
  position: absolute;
  inset: -4px;
  border-radius: 50%;
  border: 1px solid var(--accent);
  opacity: 0;
  animation: er-ripple 2.6s ease-out infinite;
}
@keyframes er-ripple {
  0%   { transform: scale(0.85); opacity: 0.55; }
  100% { transform: scale(2.5);  opacity: 0; }
}
.er-brand-meta {
  color: var(--text-dim); font-size: 12.5px; font-weight: 400;
}
@keyframes er-pulse {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.55; }
}

/* MODE BAR — ablation toggle row between topbar and studio */
.gradio-container .er-modebar {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 16px !important;
  padding: 10px 14px !important;
  margin: 0 0 14px !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  flex-wrap: wrap !important;
}
.er-modebar-label {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.er-modebar-title {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--accent);
}
.er-modebar-help {
  font-size: 11.5px;
  color: var(--text-dim);
  line-height: 1.4;
}
.gradio-container .er-rephrase-toggle { flex: 0 0 auto !important; }
.gradio-container .er-rephrase-toggle fieldset,
.gradio-container .er-rephrase-toggle .wrap-inner {
  background: var(--bg-soft) !important;
}
.gradio-container .er-rephrase-toggle label:has(input:checked) {
  background: var(--accent-soft) !important;
  color: var(--accent) !important;
}

/* SEGMENTED MODE TOGGLE */
.gradio-container .er-mode-wrap { padding: 0 !important; }
.gradio-container .er-mode-wrap > .wrap,
.gradio-container .er-mode-wrap > .form { background: transparent !important; }
.gradio-container .er-mode-wrap fieldset,
.gradio-container .er-mode-wrap .wrap-inner {
  display: inline-flex !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 999px !important;
  padding: 3px !important;
  gap: 0 !important;
}
.gradio-container .er-mode-wrap label {
  padding: 7px 18px !important;
  border-radius: 999px !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  cursor: pointer;
  transition: color 180ms ease, background 180ms ease;
  color: var(--text-muted) !important;
  background: transparent !important;
  border: none !important;
  margin: 0 !important;
  display: inline-flex !important;
  align-items: center;
}
.gradio-container .er-mode-wrap label:has(input:checked) {
  background: var(--accent-soft) !important;
  color: var(--accent) !important;
}
.gradio-container .er-mode-wrap input { display: none !important; }

/* TOPBAR RIGHT (reset). Explicitly visible, never clipped */
.gradio-container .er-reset-btn {
  flex: 0 0 auto !important;
  flex-shrink: 0 !important;
  min-width: 0 !important;
  visibility: visible !important;
  display: inline-flex !important;
}
.gradio-container .er-reset-btn button {
  background: var(--surface) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text) !important;
  padding: 7px 16px !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  border-radius: 999px !important;
  min-width: 0 !important;
  transition: all 180ms ease;
  box-shadow: none !important;
  white-space: nowrap !important;
  display: inline-flex !important;
  align-items: center !important;
  gap: 6px !important;
}
.gradio-container .er-reset-btn button:hover {
  border-color: var(--accent-line) !important;
  color: var(--accent) !important;
  background: var(--surface-2) !important;
}
/* Export button mirrors the reset button styling — sibling secondary action. */
.gradio-container .er-export-btn { flex: 0 0 auto !important; min-width: 0 !important; }
.gradio-container .er-export-btn button {
  background: var(--surface) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text) !important;
  padding: 7px 16px !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  border-radius: 999px !important;
  min-width: 0 !important;
  transition: all 180ms ease;
  box-shadow: none !important;
  white-space: nowrap !important;
  display: inline-flex !important;
}
.gradio-container .er-export-btn button:hover {
  border-color: var(--accent-line) !important;
  color: var(--accent) !important;
  background: var(--surface-2) !important;
}
.gradio-container .er-topbar-actions { flex: 0 0 auto !important; padding: 0 !important; gap: 8px; }
.gradio-container .er-support-plan-file { margin-top: 8px; }
/* Voice toggle — small low-weight link-button under the composer. Hidden
   the voice row by default; clickers expand it on demand. */
.gradio-container .er-voice-toggle {
  margin-top: 6px !important;
}
.gradio-container .er-voice-toggle button {
  background: transparent !important;
  border: none !important;
  color: var(--text-dim) !important;
  font-size: 12px !important;
  padding: 4px 6px !important;
  font-weight: 400 !important;
  text-align: left !important;
  width: auto !important;
  min-width: 0 !important;
  box-shadow: none !important;
}
.gradio-container .er-voice-toggle button:hover {
  color: var(--accent) !important;
  background: transparent !important;
}

/* Voice row sits below the composer. Compact, secondary affordance. We let
   Gradio render its native audio component (record button → waveform/timer
   while recording → auto-transcribe on stop) and just contain the size. */
.gradio-container .er-voice-row {
  margin-top: 8px;
  gap: 12px !important;
  align-items: center !important;
}
.gradio-container .er-mic {
  flex: 0 0 auto !important;
  max-width: 280px !important;
}
.gradio-container .er-mic .audio-container {
  background: var(--surface) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: 10px !important;
  padding: 4px 8px !important;
}
.er-voice-status-wrap { flex: 1 1 auto; min-width: 0; }
.er-voice-status {
  font-size: 11.5px;
  color: var(--text-dim);
  line-height: 1.5;
  padding: 0 4px;
}
.er-voice-status.er-voice-ok { color: var(--accent); }
.er-voice-status.er-voice-error { color: #ef4444; }

/* Document section per source card — F-1 / ISSS official documents the
   student is encouraged to read directly. Compact list + optional iframe. */
.er-source-docs {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed var(--border-mid);
}
.er-source-docs-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: var(--text-dim);
  margin-bottom: 6px;
}
.er-doc {
  font-size: 12.5px;
  margin-bottom: 6px;
  line-height: 1.5;
}
.er-doc-type {
  display: inline-block;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--surface-2);
  color: var(--text-dim);
  margin-right: 6px;
}
.er-doc-meta {
  font-size: 11px;
  color: var(--text-dim);
  font-style: italic;
}
.er-doc-embed { margin-top: 4px; }
.er-doc-embed summary {
  cursor: pointer;
  font-size: 11.5px;
  color: var(--accent);
}
.er-doc-embed iframe {
  width: 100%;
  height: 360px;
  border: 1px solid var(--border-mid);
  border-radius: 8px;
  margin-top: 6px;
  background: white;
}

.er-topbar > * { flex-shrink: 0 !important; }
.er-topbar { overflow: visible !important; flex: 0 0 auto !important; }
.gradio-container .er-modebar { flex: 0 0 auto !important; }

/* STUDIO 2-COLUMN LAYOUT — natural Gradio flow.

   The chatbot has its height controlled in Python via gr.Chatbot(height=N)
   which is the framework's official sizing API. Around it, hero/chips/dock
   flow naturally below. The right column is sticky so it stays visible as
   the user scrolls. We do NOT lock the page to 100vh; Gradio expects pages
   to be as tall as their content. */
.gradio-container .er-studio {
  display: grid !important;
  grid-template-columns: minmax(0, 1fr) 360px !important;
  gap: 32px !important;
  align-items: start !important;
  width: 100% !important;
  max-width: 100% !important;
}

.gradio-container .er-chat-col,
.gradio-container .er-context-col {
  min-width: 0 !important;
  max-width: 100% !important;
  background: transparent !important;
  padding: 0 !important;
  box-sizing: border-box !important;
}

/* CONTEXT COLUMN — natural document flow so it renders correctly inside
   the HF Spaces iframe. Sticky positioning needs a scroll-container
   ancestor that the iframe document does not provide. */
.gradio-container .er-context-col {
  align-self: start !important;
  padding: 0 6px 16px 24px !important;
  border-left: 1px solid var(--border) !important;
}

/* HERO (empty state). Compact so it fits the viewport */
.er-hero {
  text-align: left;
  padding: 8px 4px 6px;
}
.er-hero h1 {
  font-size: 26px;
  font-weight: 500;
  letter-spacing: -0.024em;
  margin: 0 0 8px;
  line-height: 1.2;
  /* Solid color instead of gradient text-clip — the latter renders fully
     transparent in HF Spaces' iframe sandbox in some browsers. */
  color: #f3f7fc;
}
.er-hero p {
  color: var(--text-muted);
  font-size: 13.5px;
  margin: 0;
  max-width: 580px;
  line-height: 1.6;
}
.er-hero-meta {
  margin-top: 10px;
  color: var(--text-dim);
  font-size: 10.5px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

/* CHIPS. Outlined pills, no fill. Sits directly above the input. */
.er-chips {
  display: flex !important;
  gap: 6px !important;
  flex-wrap: wrap !important;
  margin: 0 !important;
  padding: 0 !important;
}
.gradio-container .er-chip-btn { min-width: 0 !important; flex: 0 0 auto !important; }
.gradio-container .er-chip-btn button {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-muted) !important;
  padding: 9px 14px !important;
  font-size: 12.5px !important;
  font-weight: 400 !important;
  border-radius: 999px !important;
  transition: all 200ms ease;
  text-align: left !important;
  min-width: 0 !important;
  box-shadow: none !important;
  white-space: nowrap;
}
.gradio-container .er-chip-btn button:hover {
  border-color: var(--accent-line) !important;
  background: var(--surface-2) !important;
  color: var(--text) !important;
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(94,234,212,0.10) !important;
}

/* CHAT */
.gradio-container .er-chat {
  background: transparent !important;
  border: none !important;
  margin-top: 10px;
}
.gradio-container .er-chat > .wrap,
.gradio-container .er-chat > div {
  background: transparent !important;
  border: none !important;
}
.gradio-container .er-chat .message-wrap { gap: 6px !important; }
.gradio-container .er-chat .message {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  font-size: 15.5px !important;
  line-height: 1.72 !important;
  padding: 18px 0 !important;
  color: var(--text) !important;
  max-width: 100% !important;
  animation: er-msg-in 280ms cubic-bezier(0.22, 0.61, 0.36, 1) both;
}
@keyframes er-msg-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
.gradio-container .er-chat .message.user,
.gradio-container .er-chat .user {
  background: var(--accent-soft) !important;
  color: var(--text) !important;
  border-radius: 18px 18px 4px 18px !important;
  padding: 14px 18px !important;
  max-width: 92% !important;
  margin-left: auto !important;
  border: 1px solid var(--accent-line) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 1px 2px rgba(0,0,0,0.10) !important;
}
.gradio-container .er-chat .message.bot,
.gradio-container .er-chat .bot {
  padding-left: 0 !important;
  background: transparent !important;
  border: none !important;
  max-width: 100% !important;
}
.gradio-container .er-chat .message p { margin: 0 0 12px !important; }
.gradio-container .er-chat .message p:last-child { margin: 0 !important; }
.gradio-container .er-chat .avatar-container { display: none !important; }

/* TYPING DOTS */
.er-typing {
  display: inline-flex;
  gap: 5px;
  align-items: center;
  height: 1.4em;
  padding: 4px 0;
}
.er-typing > span {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--text-dim);
  animation: er-blink 1.4s infinite both;
  display: inline-block;
}
.er-typing > span:nth-child(2) { animation-delay: 0.18s; }
.er-typing > span:nth-child(3) { animation-delay: 0.36s; }
@keyframes er-blink {
  0%, 80%, 100% { opacity: 0.25; transform: scale(0.85); }
  40% { opacity: 1; transform: scale(1); background: var(--accent); }
}

/* COMPOSER */
/* BOTTOM DOCK: divider · chips · composer · footnote.
   Sits below the scrollable chatbot, anchored to the bottom of the chat
   column. Provides clear visual separation from chat history above. */
.gradio-container .er-dock {
  flex: 0 0 auto !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 10px !important;
  padding: 14px 0 0 !important;
  margin-top: 6px !important;
}
.er-dock-divider {
  height: 1px;
  width: 100%;
  background: var(--border);
  margin: 0;
}

/* COMPOSER. Flat surface, single muted border, no shadows / gradients. */
.er-composer-wrap {
  background: #1a1f2e !important;
  border: 1px solid var(--border) !important;
  border-radius: 11px !important;
  padding: 0 !important;
  position: relative;
  transition: border-color 160ms ease;
}
.er-composer-wrap:focus-within {
  border-color: var(--border-strong);
  box-shadow: 0 0 0 2px rgba(94,234,212,0.10);
}
.gradio-container .er-composer-wrap textarea {
  background: transparent !important;
  border: none !important;
  resize: none !important;
  color: var(--text) !important;
  font-size: 14.5px !important;
  line-height: 1.5 !important;
  padding: 12px 56px 12px 14px !important;
  min-height: 44px !important;
  max-height: 132px !important;
  outline: none !important;
  box-shadow: none !important;
  font-family: inherit !important;
  width: 100% !important;
}
.gradio-container .er-composer-wrap textarea::placeholder { color: var(--text-dim) !important; }

/* SEND BUTTON. Muted icon by default, teal on hover only. */
.gradio-container .er-send-btn {
  position: absolute !important;
  right: 6px !important;
  bottom: 6px !important;
  min-width: 0 !important;
  z-index: 6;
}
.gradio-container .er-send-btn button {
  background: transparent !important;
  color: var(--text-dim) !important;
  border: none !important;
  width: 32px !important;
  height: 32px !important;
  min-width: 32px !important;
  border-radius: 8px !important;
  padding: 0 !important;
  font-size: 16px !important;
  font-weight: 500 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: color 160ms ease, background 160ms ease;
  box-shadow: none !important;
}
.gradio-container .er-send-btn button:hover {
  color: #5eead4 !important;
  background: rgba(94,234,212,0.08) !important;
}
.gradio-container .er-send-btn button:active { color: #2dd4bf !important; }

/* FOOTNOTE */
.er-footnote {
  margin-top: 6px;
  color: var(--text-dim);
  font-size: 10.5px;
  letter-spacing: 0.01em;
  text-align: center;
}

/* =============================================
   LIVE CONTEXT PANEL (right column)
   ============================================= */
.er-context {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  position: relative;
  /* overflow MUST be visible so the column's scroll can show all content.
     Earlier `overflow: hidden` was clipping the resources/things-to-try
     list inside the card while the column thought everything fit. */
  overflow: visible;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  box-sizing: border-box;
}
/* Top gradient line via inset border-image trick so we don't need
   overflow:hidden to clip a ::before pseudo-element. */
.er-context {
  border-top: 1px solid transparent;
  background-clip: padding-box;
}
.er-context::before {
  content: "";
  position: absolute;
  top: -1px; left: 18px; right: 18px;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-aurora), transparent);
  opacity: 0.6;
  pointer-events: none;
}

.er-ctx-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.er-ctx-title {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: 0.01em;
}
.er-ctx-status {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.10em;
  color: var(--text-dim);
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.er-ctx-status::before {
  content: "";
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--text-faint);
}
.er-ctx-status.active::before { background: var(--accent); box-shadow: 0 0 8px var(--accent-glow); }
.er-ctx-mode {
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-dim);
  padding: 3px 9px;
  border-radius: 999px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  font-weight: 500;
}
.er-ctx-mode.active {
  color: var(--accent);
  background: var(--accent-soft);
  border-color: var(--accent-line);
}
.er-ctx-mode.warm {
  color: var(--warm);
  background: var(--warm-soft);
  border-color: var(--warm-line);
}
.er-ctx-mode.fallback-warn {
  /* Distinct from intentional warm: subtle pulse so the user notices the
     swap from a working LLM to deterministic-fallback. */
  cursor: help;
  animation: er-fallback-pulse 2.4s ease-in-out infinite;
}

/* Safety pipeline visualization — 6 chips showing each layer's state for
   the current turn. Hover for tooltip with the layer's reason. */
.er-safety-pipeline {
  margin: 10px 0 4px 0;
  padding: 8px 10px;
  background: rgba(94, 234, 212, 0.04);
  border: 1px solid var(--border);
  border-radius: 8px;
}
.er-safety-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-dim);
  margin-bottom: 5px;
  font-weight: 500;
}
.er-safety-row {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.er-safety-chip {
  font-size: 10.5px;
  font-weight: 600;
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid var(--border-mid);
  cursor: help;
  transition: transform 120ms ease;
  letter-spacing: 0.02em;
  min-width: 26px;
  text-align: center;
}
.er-safety-chip:hover { transform: translateY(-1px); }
.er-safety-on {
  background: rgba(94, 234, 212, 0.14);
  border-color: rgba(94, 234, 212, 0.36);
  color: var(--accent);
}
.er-safety-hit {
  background: rgba(248, 113, 113, 0.16);
  border-color: rgba(248, 113, 113, 0.42);
  color: #fda4a4;
}
.er-safety-skip {
  background: rgba(255,255,255,0.04);
  border-color: var(--border);
  color: var(--text-dim);
}
.er-safety-off {
  background: transparent;
  border-color: var(--border);
  color: rgba(255,255,255,0.18);
}

/* Resource card foot row — Open ↗ link + last-verified date as a small
   trust signal. Clinicians look for this. */
.er-source-foot {
  margin-top: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
}
.er-source-verified {
  font-size: 10.5px;
  color: var(--text-dim);
  background: var(--surface);
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  cursor: help;
}
@keyframes er-fallback-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(251, 146, 60, 0.0); }
  50%      { box-shadow: 0 0 0 4px rgba(251, 146, 60, 0.18); }
}
.er-ctx-status.warm::before { background: var(--warm); box-shadow: 0 0 8px rgba(245,182,105,0.36); }
.er-ctx-status.danger::before { background: var(--danger); box-shadow: 0 0 8px rgba(248,113,113,0.36); }

/* Conversation arc */
.er-arc {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.er-arc-text {
  font-size: 13.5px;
  color: var(--text);
  line-height: 1.55;
  font-weight: 500;
}
.er-arc-sub {
  font-size: 11.5px;
  color: var(--text-muted);
  line-height: 1.6;
}
.er-arc-meter {
  height: 3px;
  background: rgba(255,255,255,0.05);
  border-radius: 999px;
  overflow: hidden;
  margin-top: 4px;
}
.er-arc-meter > div {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-dim), var(--accent));
  transition: width 480ms cubic-bezier(0.22, 0.61, 0.36, 1);
  box-shadow: 0 0 12px var(--accent-glow);
}

/* Signal pills. Wrap freely, never push container width. Long labels
   (e.g. "F-1 / international context") allowed to break to next line. */
.er-signals {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  max-width: 100%;
  min-width: 0;
}
.er-signal {
  font-size: 11px;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-muted);
  border: 1px solid var(--border);
  letter-spacing: 0.02em;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-weight: 500;
  max-width: 100%;
  white-space: normal;
  word-break: break-word;
  line-height: 1.35;
}
.er-signal.route { background: var(--accent-soft); color: var(--accent); border-color: var(--accent-line); }
.er-signal.stage { background: var(--indigo-soft); color: var(--indigo); border-color: var(--indigo-line); }
.er-signal.tier-warm { background: var(--warm-soft); color: var(--warm); border-color: var(--warm-line); }
.er-signal.tier-danger { background: var(--danger-soft); color: var(--danger); border-color: var(--danger-line); }
.er-signal.intl {
  background: var(--warm-soft);
  color: var(--warm);
  border-color: var(--warm-line);
  animation: er-signal-in 360ms cubic-bezier(0.22, 0.61, 0.36, 1) both;
}
.er-signal.intl::before { content: "✦"; font-size: 9px; }
@keyframes er-signal-in {
  from { opacity: 0; transform: translateY(-3px) scale(0.95); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* Section heading */
.er-ctx-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.er-ctx-section-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}
.er-section-title {
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin: 0;
}
.er-count-pill {
  font-size: 10.5px;
  color: var(--text-faint);
  font-feature-settings: "tnum";
}

/* Resource cards. Uniform min-height, column flex so the "Open ↗" link
   bottom-aligns regardless of how long the title or reason is. */
.er-resources { display: flex; flex-direction: column; gap: 10px; }
.er-rsrc {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  min-height: 84px;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  box-sizing: border-box;
  transition: border-color 180ms ease, background 180ms ease;
  animation: er-rsrc-in 320ms cubic-bezier(0.22, 0.61, 0.36, 1) both;
}
@keyframes er-rsrc-in {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.er-rsrc:hover {
  border-color: var(--accent-line);
  background: var(--surface-3);
}
.er-rsrc.featured {
  border-color: var(--warm-line);
  background: linear-gradient(135deg, var(--warm-soft), transparent 60%), var(--surface-2);
}
.er-rsrc.featured:hover { background: linear-gradient(135deg, var(--warm-soft), transparent 50%), var(--surface-3); }
.er-rsrc-title {
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
  margin-bottom: 4px;
  line-height: 1.4;
  display: flex;
  align-items: baseline;
  gap: 6px;
  word-break: break-word;
}
.er-rsrc-title::before {
  content: "◇";
  color: var(--accent);
  font-size: 9px;
  opacity: 0.7;
  flex: 0 0 auto;
}
.er-rsrc.featured .er-rsrc-title::before { content: "✦"; color: var(--warm); opacity: 0.9; }
.er-rsrc.crisis .er-rsrc-title::before { content: "✦"; color: var(--danger); opacity: 0.9; }
.er-rsrc-why {
  font-size: 11px;
  color: var(--text-muted);
  line-height: 1.5;
  margin-bottom: 10px;
  word-break: break-word;
}
.er-rsrc a {
  color: var(--accent);
  font-size: 11.5px;
  text-decoration: none;
  border-bottom: 1px solid var(--accent-line);
  transition: border-color 180ms ease, color 180ms ease;
  align-self: flex-start;
  margin-top: auto;  /* push link to bottom of card */
}
.er-rsrc a:hover { border-bottom-color: var(--accent); }

/* Things-to-try chips */
.er-actions { display: flex; flex-direction: column; gap: 8px; }
.er-action {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 11px 12px;
  font-size: 12.5px;
  color: var(--text);
  line-height: 1.5;
  position: relative;
  animation: er-rsrc-in 320ms cubic-bezier(0.22, 0.61, 0.36, 1) both;
}
.er-action::before {
  content: "→";
  color: var(--accent);
  margin-right: 8px;
  font-weight: 600;
  opacity: 0.7;
}

/* Empty section state */
.er-empty {
  color: var(--text-faint);
  font-size: 12px;
  padding: 14px;
  text-align: center;
  background: var(--surface-2);
  border: 1px dashed var(--border);
  border-radius: var(--radius-sm);
  font-style: italic;
}

/* Diagnostics expander (Accordion) */
.gradio-container .er-diag-acc {
  margin-top: 16px;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--surface) !important;
  overflow: hidden;
}
.gradio-container .er-diag-acc .label-wrap {
  padding: 12px 16px !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  color: var(--text-dim) !important;
  background: transparent !important;
}
.gradio-container .er-diag-acc .label-wrap:hover { color: var(--text-muted) !important; }
.gradio-container .er-diag-acc > .wrap > .open { padding: 0 16px 16px !important; }

/* Diag grid (inside accordion) */
.er-diag-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.er-diag {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  min-width: 0;
}
.er-diag .k {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-dim);
  margin-bottom: 4px;
  font-weight: 500;
}
.er-diag .v {
  font-size: 12.5px;
  color: var(--text);
  font-weight: 500;
  line-height: 1.4;
  word-break: break-word;
  font-feature-settings: "tnum";
}
.er-diag.warn { border-color: var(--warm-line); }
.er-diag.warn .v { color: var(--warm); }
.er-diag.danger { border-color: var(--danger-line); }
.er-diag.danger .v { color: var(--danger); }
.er-diag.accent { border-color: var(--accent-line); }
.er-diag.accent .v { color: var(--accent); }

/* IG tokens (in diagnostics) */
.er-ig-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.er-ig {
  font-size: 10.5px;
  padding: 3px 9px;
  border-radius: 999px;
  background: var(--danger-soft);
  color: var(--danger);
  border: 1px solid var(--danger-line);
}

/* HIDE GRADIO CRUFT */
.gradio-container footer { display: none !important; }
.gradio-container .progress-text { color: var(--text-dim) !important; }
.gradio-container .icon-button-wrapper { background: transparent !important; }

/* SCROLLBAR */
.gradio-container ::-webkit-scrollbar { width: 8px; height: 8px; }
.gradio-container ::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.06); border-radius: 999px;
}
.gradio-container ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.12); }
.gradio-container ::-webkit-scrollbar-track { background: transparent; }

/* RESPONSIVE — collapse to single column on narrow viewports. */
@media (max-width: 1100px) {
  .gradio-container { padding: 0 24px 40px !important; }
  .gradio-container .er-studio {
    grid-template-columns: 1fr !important;
    gap: 24px !important;
  }
  .gradio-container .er-context-col {
    position: static !important;
    top: auto !important;
    max-height: none !important;
    overflow: visible !important;
    border-left: none !important;
    border-top: 1px solid var(--border) !important;
    padding: 18px 0 0 !important;
  }
}
@media (max-width: 700px) {
  .gradio-container { padding: 0 16px 32px !important; }
  .er-topbar { flex-wrap: wrap !important; gap: 10px !important; }
  .er-topbar > .er-mode-wrap { order: 3; flex-basis: 100% !important; justify-content: center; }
  .er-hero h1 { font-size: 26px; }
  .er-diag-grid { grid-template-columns: 1fr; }
  .gradio-container .er-chat .message.user { max-width: 90% !important; }
  /* Make the 4-button action area in the topbar stay tappable on phones */
  .gradio-container .er-export-btn,
  .gradio-container .er-reset-btn { flex: 1 1 auto !important; }
  .gradio-container .er-export-btn button,
  .gradio-container .er-reset-btn button { padding: 8px 12px !important; font-size: 12px !important; }
  /* Safety pipeline chips wrap to multiple lines on phones; keep them
     readable rather than squished. */
  .er-safety-row { row-gap: 5px !important; }
  .er-safety-chip { font-size: 10px !important; padding: 3px 7px !important; }
  /* Hero gets cramped at narrow widths */
  .er-hero { padding: 18px 14px !important; }
  .er-hero p { font-size: 14px !important; }
  /* Composer right-padding accounts for one button at narrow widths */
  .gradio-container .er-composer-wrap textarea { padding-right: 52px !important; }
  /* Source cards already full-width; tighten internal padding */
  .er-source { padding: 12px 14px !important; }
  /* Voice row stays compact */
  .er-voice-row { flex-direction: column !important; align-items: stretch !important; gap: 6px !important; }
  .er-voice-row > * { width: 100% !important; }
}

/* Extra-small phones (iPhone SE width ~375px) */
@media (max-width: 420px) {
  .gradio-container { padding: 0 12px 24px !important; }
  .er-brand { gap: 8px !important; font-size: 14px !important; }
  .er-hero h1 { font-size: 22px !important; }
  .er-hero p { font-size: 13px !important; }
  .er-modebar { flex-wrap: wrap !important; }
  .er-chip-btn { font-size: 11.5px !important; padding: 6px 10px !important; }
  /* Send button stays anchored bottom-right but shrinks slightly */
  .gradio-container .er-send-btn button { padding: 8px 12px !important; }
}
"""


class FastDemoPipeline:
    """Presentation backend backed by EmpathRAG Core without heavyweight LLM loading."""

    def __init__(self, db_path: Path, retrieval_corpus: str, top_k: int):
        self.db_path = db_path
        self.retrieval_corpus = "curated_support" if db_path.exists() else retrieval_corpus
        self.top_k = top_k
        self.safety_policy = SafetyTriagePolicy()
        self.core = EmpathRAGCore(
            curated_db_path=db_path,
            retrieval_corpus=self.retrieval_corpus,
            top_k=top_k,
        )
        self._turn = 0
        self._tier_history: list[str] = []
        self._crisis_locked = False
        self._last_escalation_reason = ""

    def run(self, user_message: str, audience_mode: str = "student") -> dict:
        core_result = self.core.run_turn(
            message=user_message,
            session_id="demo",
            audience_mode=audience_mode,
            resource_profile="umd",
            backend_mode="hybrid_ml",
        ).to_dict()
        return self._enrich_result(core_result)

    def run_streaming(self, user_message: str, audience_mode: str = "student"):
        """Generator wrapping ``EmpathRAGCore.run_turn_streaming``.

        Yields ``("token", text)`` for each streamed chunk and ``("done",
        enriched_result_dict)`` exactly once at the end.
        """
        for event in self.core.run_turn_streaming(
            message=user_message,
            session_id="demo",
            audience_mode=audience_mode,
            resource_profile="umd",
            backend_mode="hybrid_ml",
        ):
            kind = event[0]
            if kind == "token":
                yield ("token", event[1])
            elif kind == "done":
                core_result = event[1].to_dict()
                yield ("done", self._enrich_result(core_result))

    def _enrich_result(self, core_result: dict) -> dict:
        emotion_name = core_result.get("emotion_name", "neutral")
        emotion_label = ["distress", "anxiety", "frustration", "neutral", "hopeful"].index(
            emotion_name if emotion_name in {"distress", "anxiety", "frustration", "neutral", "hopeful"} else "neutral"
        )
        core_result.update(
            {
                "emotion": emotion_label,
                "trajectory": core_result.get("trajectory_state", "active"),
                "crisis_confidence": 1.0 if core_result.get("crisis") else 0.0,
                "safety_level": core_result.get("safety_tier", ""),
            }
        )
        return core_result

    def _legacy_run(self, user_message: str, audience_mode: str = "student") -> dict:
        self._turn += 1
        emotion_name = self._emotion_name(user_message)
        emotion_label = ["distress", "anxiety", "frustration", "neutral", "hopeful"].index(emotion_name)
        safety_decision = self.safety_policy.classify(
            user_message,
            confidence=0.0,
            model_flag=False,
        )
        if safety_decision.level == SafetyLevel.PASS and self._wellbeing_request(user_message):
            safety_level = SafetyLevel.WELLBEING_SUPPORT
            safety_reason = "wellbeing_or_grounding_request"
        else:
            safety_level = safety_decision.level
            safety_reason = safety_decision.reason

        safety_tier = map_safety_level(safety_level, wellbeing_request=self._wellbeing_request(user_message))
        normalized_message = user_message.lower()
        dependency_or_secrecy = any(
            phrase in normalized_message
            for phrase in (
                "you are the only one",
                "only one i can talk to",
                "don't tell anyone",
                "do not tell anyone",
                "keep this secret",
                "no one can help",
            )
        )
        peer_context = audience_mode == "helping_friend" or any(
            phrase in normalized_message
            for phrase in ("my friend", "my roommate", "my labmate", "my teammate", "someone i know")
        )
        peer_imminent = peer_context and (
            "goodbye" in normalized_message
            and any(phrase in normalized_message for phrase in ("locked", "will not answer", "won't answer", "not answering"))
        )
        if peer_imminent:
            safety_tier = SafetyTier.IMMINENT_SAFETY
            safety_reason = "peer_goodbye_unreachable"
        elif dependency_or_secrecy and safety_tier == SafetyTier.SUPPORT_NAVIGATION:
            safety_tier = SafetyTier.HIGH_DISTRESS
            safety_reason = "dependency_or_secrecy_redirect"
        route_decision = classify_route(user_message, safety_tier, audience_mode=audience_mode)
        escalation_reason = self._update_trajectory_lock(user_message, safety_tier, route_decision.route)

        if safety_decision.should_intercept or self._crisis_locked or safety_tier == SafetyTier.IMMINENT_SAFETY:
            retrieved = self._retrieve(
                user_message,
                SafetyLevel.CRISIS,
                route=route_decision.route.value,
                safety_tier=SafetyTier.IMMINENT_SAFETY.value,
                audience_mode=audience_mode,
            )
            if route_decision.route == SupportRoute.PEER_HELPER:
                response = (
                    "I am concerned this could be an immediate safety situation for your friend. "
                    "Do not handle this alone. Contact emergency/crisis support now, and involve a trusted nearby person, RA, supervisor, or campus support while you try to reach them."
                )
            else:
                response = safety_decision.response or (
                    "I am really concerned about your immediate safety. Please call or text 988 now, "
                    "or call emergency services if you may be in immediate danger."
                )
            return self._result(
                response=response,
                emotion_label=emotion_label,
                emotion_name=emotion_name,
                safety_level=safety_decision.level,
                safety_reason=safety_decision.reason,
                crisis=True,
                retrieved=retrieved,
                latency={"demo_backend_ms": 8},
                route_label=route_decision.route.value,
                safety_tier=SafetyTier.IMMINENT_SAFETY.value,
                recommended_action=self._recommended_action(route_decision.route.value),
                escalation_reason=escalation_reason,
                output_guard={"allowed": True, "reason": "crisis_template", "flags": []},
            )

        retrieved = self._retrieve(
            user_message,
            safety_level,
            route=route_decision.route.value,
            safety_tier=safety_tier.value,
            audience_mode=audience_mode,
        )
        route_label = route_decision.route.value
        response = self._response_for(user_message, retrieved, safety_level, route_label, audience_mode)
        guard = validate_output(
            response=response,
            retrieved_sources=self._source_summaries(retrieved),
            safety_tier=safety_tier.value,
            route=route_label,
            conversation_history=[],
        )
        if guard.fallback_required and guard.corrected_response:
            response = guard.corrected_response
        return self._result(
            response=response,
            emotion_label=emotion_label,
            emotion_name=emotion_name,
            safety_level=safety_level,
            safety_reason=safety_reason,
            crisis=False,
            retrieved=retrieved,
            latency={"demo_backend_ms": 8},
            route_label=route_label,
            recommended_action=self._recommended_action(route_label),
            safety_tier=safety_tier.value,
            escalation_reason=escalation_reason,
            output_guard={"allowed": guard.allowed, "reason": guard.reason, "flags": guard.flags},
        )

    def tracker_trajectory(self) -> str:
        return "stable"

    def reset_session(self) -> None:
        self._turn = 0
        self._tier_history = []
        self._crisis_locked = False
        self._last_escalation_reason = ""
        self.core.reset_session("demo")

    def _result(
        self,
        response: str,
        emotion_label: int,
        emotion_name: str,
        safety_level: SafetyLevel,
        safety_reason: str,
        crisis: bool,
        retrieved: list[dict],
        latency: dict,
        route_label: str,
        recommended_action: str,
        safety_tier: str,
        escalation_reason: str,
        output_guard: dict,
    ) -> dict:
        return {
            "response": response,
            "emotion": emotion_label,
            "emotion_name": emotion_name,
            "trajectory": "stable",
            "crisis": crisis,
            "crisis_confidence": 1.0 if crisis else 0.0,
            "safety_level": safety_level.value,
            "safety_tier": safety_tier,
            "safety_reason": safety_reason,
            "escalation_reason": escalation_reason,
            "ig_highlights": [],
            "retrieved_chunks": [row["text"] for row in retrieved],
            "retrieved_sources": self._source_summaries(retrieved),
            "retrieval_corpus": self.retrieval_corpus,
            "latency_ms": latency,
            "route_label": route_label,
            "recommended_action": recommended_action,
            "output_guard": output_guard,
        }

    def _retrieve(
        self,
        message: str,
        safety_level: SafetyLevel,
        route: str | None = None,
        safety_tier: str | None = None,
        audience_mode: str = "student",
    ) -> list[dict]:
        if not self.db_path.exists():
            return [node.as_source("resource registry fallback") for node in match_services(route or "", safety_tier or "", audience_mode, limit=self.top_k)]
        topics, source_names = self._targets(message, safety_level, route=route)
        usage_modes = self._usage_modes(safety_level)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, resource_id, text, source_id, source_name, source_type,
                   title, url, topic, audience, risk_level, usage_mode, summary,
                   last_checked, notes
            FROM chunks
            WHERE usage_mode IN ({})
            """.format(",".join("?" * len(usage_modes))),
            tuple(usage_modes),
        ).fetchall()
        conn.close()

        scored = []
        query = message.lower()
        for row in rows:
            score = 0
            reasons = []
            title = row["title"].lower()
            if row["topic"] in topics:
                score += 8
                reasons.append(f"topic match: {row['topic']}")
            if row["source_name"] in source_names:
                score += 7
                reasons.append(f"preferred source: {row['source_name']}")
            if "workshop" in title and any(token in query for token in ("stress", "anxious", "panic", "grades", "exam")):
                score += 6
                reasons.append("student workshop fit")
            if "ptsd" in title and not any(token in query for token in ("ptsd", "trauma", "traumatic", "flashback")):
                score -= 12
            if "eating disorder" in title and not any(token in query for token in ("eating", "food", "body", "weight", "diet")):
                score -= 12
            if "funding" in title and not any(token in query for token in ("funding", "financial", "money", "tuition", "assistantship")):
                score -= 8
            if "admission" in title and not any(token in query for token in ("admission", "admissions", "apply", "application", "admitted")):
                score -= 12
            if "traumatic" in title and not any(token in query for token in ("trauma", "traumatic", "ptsd", "assault", "violence")):
                score -= 8
            haystack = f"{row['title']} {row['summary']} {row['text']}".lower()
            keyword_hits = []
            for token in self._keywords(query):
                if token in haystack:
                    score += 1
                    keyword_hits.append(token)
            if keyword_hits:
                reasons.append("keyword overlap: " + ", ".join(keyword_hits[:3]))
            row_dict = dict(row)
            row_dict["why_retrieved"] = "; ".join(reasons[:2]) if reasons else "semantic support match"
            scored.append((score, row_dict))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        source_counts: dict[str, int] = {}
        seen_cards: set[tuple[str, str]] = set()
        for score, row in scored:
            if score <= 0 and selected:
                continue
            card_key = (row["source_name"], row["title"])
            if card_key in seen_cards:
                continue
            source = row["source_name"]
            if source_counts.get(source, 0) >= 2:
                continue
            selected.append(row)
            seen_cards.add(card_key)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(selected) == self.top_k:
                break
        if route and safety_tier:
            seen_source_titles = {(row.get("source_name", ""), row.get("title", "")) for row in selected}
            graph_rows = []
            for node in match_services(route, safety_tier, audience_mode, limit=self.top_k):
                source_row = node.as_source("resource registry route match")
                key = (source_row.get("source_name", ""), source_row.get("title", ""))
                if key in seen_source_titles:
                    continue
                if source_row.get("usage_mode") not in usage_modes:
                    continue
                graph_rows.append(source_row)
                seen_source_titles.add(key)
            selected = (graph_rows + selected)[: self.top_k]
        return selected

    def _targets(self, message: str, safety_level: SafetyLevel, route: str | None = None) -> tuple[set[str], set[str]]:
        text = message.lower()
        if safety_level in {SafetyLevel.CRISIS, SafetyLevel.EMERGENCY}:
            return (
                {"crisis_immediate_help", "emergency_services"},
                {"988 Suicide & Crisis Lifeline", "UMD Counseling Center"},
            )
        if route == SupportRoute.PEER_HELPER.value:
            return (
                {"crisis_immediate_help", "help_seeking_script", "counseling_services"},
                {"988 Suicide & Crisis Lifeline", "UMD Counseling Center", "JED Foundation"},
            )
        if route == SupportRoute.BASIC_NEEDS.value:
            return (
                {"help_seeking_script", "campus_navigation", "graduate_student_support"},
                {"UMD Dean of Students", "UMD Graduate School", "UMD Counseling Center"},
            )
        if route == SupportRoute.ACCESSIBILITY_ADS.value:
            return (
                {"accessibility_disability", "campus_navigation"},
                {"UMD Accessibility & Disability Service"},
            )
        if route == SupportRoute.ADVISOR_CONFLICT.value:
            return (
                {"advisor_conflict", "graduate_student_support"},
                {"UMD Graduate School Ombuds", "UMD Graduate School"},
            )
        if "accommodation" in text or "disability" in text or "ads" in text:
            return (
                {"accessibility_disability"},
                {"UMD Accessibility & Disability Service"},
            )
        if "advisor" in text or "ombuds" in text or "neutral" in text:
            return (
                {"advisor_conflict", "graduate_student_support"},
                {"UMD Graduate School Ombuds", "UMD Counseling Center"},
            )
        if "ground" in text or "panic" in text or "panicking" in text:
            return (
                {"grounding_exercise", "anxiety_stress", "counseling_services"},
                {"UMD Counseling Center", "NAMI", "NIMH"},
            )
        if any(word in text for word in ("stress", "stressful", "stressed", "overwhelmed", "too much", "spiral")):
            return (
                {"anxiety_stress", "academic_burnout", "counseling_services", "grounding_exercise"},
                {"UMD Counseling Center", "NIMH"},
            )
        if any(word in text for word in ("failed", "fail", "exam", "grades", "grade", "doomed", "class", "course", "semester")):
            return (
                {"academic_burnout", "anxiety_stress", "counseling_services", "graduate_student_support"},
                {"UMD Counseling Center", "UMD Graduate School", "NIMH"},
            )
        if any(word in text for word in ("depressing", "depressed", "depression", "low mood")):
            return (
                {"depression_support", "counseling_services", "anxiety_stress"},
                {"UMD Counseling Center", "NIMH", "NAMI"},
            )
        if any(word in text for word in ("grade", "grades", "doomed", "failing", "failed", "class", "course", "semester")):
            return (
                {"academic_burnout", "anxiety_stress", "counseling_services", "graduate_student_support"},
                {"UMD Counseling Center", "UMD Graduate School", "NIMH"},
            )
        if "counsel" in text or "therapy" in text or "start" in text:
            return (
                {"counseling_services", "campus_navigation", "therapy_expectations"},
                {"UMD Counseling Center"},
            )
        if "isolated" in text or "lonely" in text:
            return (
                {"isolation_loneliness", "counseling_services"},
                {"UMD Counseling Center", "NAMI"},
            )
        return (
            {"anxiety_stress", "counseling_services", "academic_burnout"},
            {"UMD Counseling Center", "NIMH"},
        )

    def _usage_modes(self, safety_level: SafetyLevel) -> tuple[str, ...]:
        if safety_level in {SafetyLevel.CRISIS, SafetyLevel.EMERGENCY}:
            return ("crisis_only",)
        if safety_level == SafetyLevel.WELLBEING_SUPPORT:
            return ("retrieval", "wellbeing_only")
        return ("retrieval",)

    def _keywords(self, query: str) -> list[str]:
        return [token for token in query.replace("?", " ").replace(".", " ").split() if len(token) > 4]

    def _source_summaries(self, rows: list[dict]) -> list[dict]:
        return [
            {
                "title": row.get("title", ""),
                "source_name": row.get("source_name", ""),
                "url": row.get("url", ""),
                "topic": row.get("topic", ""),
                "risk_level": row.get("risk_level", ""),
                "usage_mode": row.get("usage_mode", ""),
                "source_type": row.get("source_type", ""),
                "why_retrieved": row.get("why_retrieved", ""),
            }
            for row in rows
        ]

    def _emotion_name(self, message: str) -> str:
        text = message.lower()
        if any(word in text for word in ("safe tonight", "hurt myself", "hopeless", "die", "suicide")):
            return "distress"
        if any(word in text for word in ("depressing", "depressed", "depression", "failed my exam")):
            return "distress"
        if any(word in text for word in ("anxious", "panic", "panicking", "overwhelmed", "exam", "grades", "grade", "doomed", "failing", "stress", "stressful", "stressed")):
            return "anxiety"
        if any(word in text for word in ("advisor", "dismiss", "angry", "rejected")):
            return "frustration"
        if any(word in text for word in ("finished", "better", "proud", "hopeful")):
            return "hopeful"
        return "neutral"

    def _wellbeing_request(self, message: str) -> bool:
        text = message.lower()
        return any(word in text for word in ("grounding", "ground", "panic", "breathing", "cope"))

    def _response_for(
        self,
        message: str,
        rows: list[dict],
        safety_level: SafetyLevel,
        route_label: str,
        audience_mode: str,
    ) -> str:
        source = rows[0]["source_name"] if rows else "a student-support resource"
        topic = rows[0]["topic"].replace("_", " ") if rows else "student support"
        source_line = self._source_line(rows)
        if route_label == SupportRoute.PEER_HELPER.value:
            return (
                "Route detected: peer-helper support. This is not something your friend should have to handle alone, and it is not something you should handle alone either.\n\n"
                "Recommended next action: if there may be immediate danger, contact emergency/crisis support now and involve a trusted nearby person, RA, supervisor, or campus support. Do not promise secrecy when safety may be at risk.\n\n"
                f"Sources matched: {source_line}\n\n"
                "A safer thing to say: I care about you, and I am worried enough that we need to get another person involved right now."
            )
        if route_label == SupportRoute.BASIC_NEEDS.value:
            return (
                "Route detected: basic needs / student support. Food, housing, and money stress are not motivation problems; they are support-navigation problems.\n\n"
                "Recommended next action: contact a campus student-support office or Dean of Students-style support path and say plainly what you need help with today. I will not invent Pantry or Thrive details unless they are in the verified corpus.\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.ACADEMIC_SETBACK.value:
            return (
                "Route detected: academic setback with distress. Failing an exam can feel catastrophic, but this is exactly the kind of moment where the next step matters more than the spiral.\n\n"
                "Recommended next action: send a short office-hours note instead of trying to solve the whole semester tonight.\n\n"
                "Email script: Hi Professor/TA [Name], I am trying to understand what went wrong on [exam/assignment] and what I can do differently before the next assessment. Could I come to office hours or schedule a short meeting to review my mistakes?\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.LOW_MOOD.value:
            return (
                "Route detected: low mood / depression support. I am not reading this as an emergency from the wording alone, but it is serious enough to deserve support instead of being minimized.\n\n"
                f"Recommended next action: tell one trusted person what is going on, then use a campus counseling starting point. If this shifts into not feeling safe, use crisis support immediately.\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.EXAM_STRESS.value:
            return (
                "That sounds like the kind of grade panic that can make everything feel bigger and more permanent than it actually is.\n\n"
                f"Recommended next action: choose one academic action for the next 24 hours: office hours, TA email, syllabus policy check, or advisor check-in. I found {topic} resources anchored around {source}.\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.ANXIETY_PANIC.value:
            return (
                "That sounds like stress has moved from background noise into something that is taking over the whole room.\n\n"
                f"Recommended next action: first do one short grounding step, then choose whether you need a campus support path or a simple next-step plan. I found {topic} resources anchored around {source}.\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.ACCESSIBILITY_ADS.value:
            return (
                "Route detected: accessibility / accommodations support. This is a practical support path, not something you have to improvise alone.\n\n"
                f"Recommended next action: identify the class or exam barrier, then use the official ADS student process so the request is traceable.\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.ADVISOR_CONFLICT.value:
            return (
                "Route detected: advisor conflict / graduate support. The safest next step is to keep the record factual and use a neutral campus channel before the situation escalates.\n\n"
                f"Recommended next action: write down the specific concern, separate urgent academic deadlines from relationship issues, and consider Ombuds or graduate support resources.\n\n"
                f"Sources matched: {source_line}"
            )
        if safety_level == SafetyLevel.WELLBEING_SUPPORT:
            return (
                f"That sounds like a sharp spike of student stress, and it makes sense to want something steadying rather than another wall of advice.\n\n"
                f"Recommended next action: take one short grounding step, then decide whether you need who to contact or what to expect next. I found {topic} resources anchored around {source}."
            )
        return (
            f"That sounds like a real student-support concern, and you should not have to untangle it from scratch.\n\n"
            f"Recommended next action: pick one concrete support path before trying to solve the whole situation. I found {topic} resources anchored around {source}. What would help most to focus on first: next steps, who to contact, or what to expect?\n\n"
            f"Sources matched: {source_line}"
        )

    def _need_label(self, message: str, safety_level: SafetyLevel) -> str:
        text = message.lower()
        if safety_level in {SafetyLevel.CRISIS, SafetyLevel.EMERGENCY}:
            return "immediate safety"
        if "accommodation" in text or "disability" in text or "ads" in text:
            return "accessibility"
        if "advisor" in text or "neutral" in text or "ombuds" in text:
            return "advisor conflict"
        if any(word in text for word in ("failed", "failed my exam", "fail", "exam")):
            return "academic setback"
        if any(word in text for word in ("depressing", "depressed", "depression", "low mood")):
            return "low mood"
        if "counsel" in text or "therapy" in text:
            return "counseling navigation"
        if "panic" in text or "ground" in text:
            return "anxiety"
        if any(word in text for word in ("stress", "stressful", "stressed", "overwhelmed", "too much", "spiral")):
            return "stress overload"
        if any(word in text for word in ("grade", "grades", "doomed", "failing", "class", "course", "semester")):
            return "academic stress"
        return "student-support"

    def _source_line(self, rows: list[dict]) -> str:
        if not rows:
            return "no source cards available"
        labels = []
        seen = set()
        for row in rows[:3]:
            label = f"{row['source_name']} - {row['title']}"
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
        return "; ".join(labels)

    def _recommended_action(self, route_label: str) -> str:
        actions = {
            SupportRoute.CRISIS_IMMEDIATE.value: "Stop normal advice. Show 988, emergency, and campus crisis options first.",
            SupportRoute.PEER_HELPER.value: "Do not ask the peer to handle risk alone. Escalate to a trusted person, campus support, or crisis help when safety may be at risk.",
            SupportRoute.ACADEMIC_SETBACK.value: "Send a short office-hours note and identify the next academic policy/support step.",
            SupportRoute.LOW_MOOD.value: "Tell one trusted person and use a campus counseling starting point; escalate if safety changes.",
            SupportRoute.EXAM_STRESS.value: "Choose one academic action for the next 24 hours: office hours, TA email, syllabus policy check, or advisor check-in.",
            SupportRoute.ANXIETY_PANIC.value: "Start with one grounding step, then choose a support path if symptoms keep interfering.",
            SupportRoute.ACCESSIBILITY_ADS.value: "Route to the official ADS process and keep the accommodations request traceable.",
            SupportRoute.ADVISOR_CONFLICT.value: "Keep the record factual and consider Ombuds or graduate support before escalating the conflict.",
            SupportRoute.COUNSELING_NAVIGATION.value: "Explain how to start with UMD Counseling and what to expect from first contact.",
            SupportRoute.BASIC_NEEDS.value: "Route to verified campus student-support resources without inventing Pantry/Thrive details.",
        }
        return actions.get(route_label, "Keep the answer practical, source-grounded, and student-support oriented.")

    def _update_trajectory_lock(self, message: str, safety_tier: SafetyTier, route: SupportRoute) -> str:
        self._tier_history.append(safety_tier.value)
        self._tier_history = self._tier_history[-3:]
        text = message.lower()
        reason = ""
        if len(self._tier_history) == 3 and all(tier in {"imminent_safety", "high_distress"} for tier in self._tier_history):
            self._crisis_locked = True
            reason = "three_consecutive_high_risk_turns"
        dependency_or_secrecy = any(
            phrase in text
            for phrase in (
                "you are the only one",
                "only one i can talk to",
                "don't tell anyone",
                "do not tell anyone",
                "keep this secret",
                "no one can help",
            )
        )
        if dependency_or_secrecy:
            reason = reason or "dependency_or_secrecy_redirect"
            if safety_tier == SafetyTier.IMMINENT_SAFETY:
                self._crisis_locked = True
        if self._crisis_locked and not reason:
            reason = "crisis_locked"
        self._last_escalation_reason = reason
        return reason


pipeline_lock = threading.Lock()
pipeline = None


def get_pipeline() -> EmpathRAGPipeline:
    global pipeline
    if pipeline is None:
        if DEMO_BACKEND == "real":
            print("[Demo] Initialising full EmpathRAG pipeline...", flush=True)
            from pipeline.pipeline import EmpathRAGPipeline

            pipeline = EmpathRAGPipeline(
                use_real_guardrail=True,
                guardrail_threshold=0.5,
                retrieval_corpus=RETRIEVAL_CORPUS,
                top_k=DEMO_TOP_K,
                generation_max_tokens=DEMO_MAX_TOKENS,
            )
            print("[Demo] Full pipeline ready.", flush=True)
        else:
            print("[Demo] Initialising fast presentation backend.", flush=True)
            pipeline = FastDemoPipeline(
                db_path=CURATED_DB_PATH,
                retrieval_corpus=RETRIEVAL_CORPUS,
                top_k=DEMO_TOP_K,
            )
    return pipeline


def new_session_id() -> str:
    return uuid.uuid4().hex[:6].upper()


def new_session_state() -> dict:
    return {
        "session_id": new_session_id(),
        "emotion_history": [],
        "tracker_history": [],
        "conv_history": [],
        "turn_log": [],
        "started_at": datetime.datetime.utcnow().isoformat(),
    }


def log_turn(session_id, turn, user_message, result):
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
            "crisis_confidence": result["crisis_confidence"],
            "retrieval_corpus": result.get("retrieval_corpus", ""),
            "safety_level": result.get("safety_level", ""),
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[Warning] Failed to log turn: {e}")


def format_emotion_timeline(history, trajectory) -> str:
    if not history:
        return (
            "<div class='er-card'><div class='er-mini-title'>Session feel</div>"
            "<div class='er-empty'>No turns yet.</div></div>"
        )
    pretty_traj = escape(str(trajectory).replace("_", " ").title())
    html = "<div class='er-card'><div class='er-mini-title'>Session feel</div>"
    html += "<div class='er-plan-rows'>"
    html += f"<div class='er-plan-row'><span class='k'>Trajectory</span><span class='v'>{pretty_traj}</span></div>"
    html += "</div>"
    html += "<div class='er-timeline-row' style='margin-top:10px;'>"
    for item in history[-12:]:
        label = escape(str(item['label_name']))
        turn = escape(str(item['turn']))
        html += f"<span class='er-time-pill'>T{turn} · {label}</span>"
    html += "</div></div>"
    return html


def format_ig_panel(is_crisis, confidence, ig_tokens, loading, explanation_reason="") -> str:
    if not is_crisis:
        return (
            "<div class='er-card'><div class='er-mini-title'>Safety guardrail</div>"
            "<div class='er-empty'>No safety intercept on this turn.</div></div>"
        )
    conf_pct = max(2, min(100, int(confidence * 100)))
    html = "<div class='er-card'>"
    html += "<div class='er-mini-title'>Safety guardrail</div>"
    html += "<div class='er-plan-rows'>"
    html += (
        f"<div class='er-plan-row'><span class='k'>Crisis signal</span>"
        f"<span class='v' style='color:var(--danger);'>{confidence:.1%}</span></div>"
    )
    html += "</div>"
    html += f"<div class='er-meter'><div style='width:{conf_pct}%; background:var(--danger);'></div></div>"
    if loading:
        html += "<div class='er-empty' style='margin-top:12px;'>Computing token attributions…</div>"
    elif ig_tokens:
        valid = [(t, s) for t, s in ig_tokens if t.strip()]
        if valid:
            html += "<div class='er-mini-title' style='margin-top:14px;'>Top crisis signals</div>"
            html += "<div class='er-ig-row'>"
            for tok, _score in valid[:10]:
                html += f"<span class='er-ig'>{escape(tok)}</span>"
            html += "</div>"
    elif explanation_reason:
        html += (
            f"<div class='er-source-why' style='margin-top:10px;'>"
            f"{escape(str(explanation_reason))}</div>"
        )
    html += "</div>"
    return html


def format_decision_trace(result=None) -> str:
    """Support card. What kind of support, what's next, which resources."""
    if not result:
        return (
            "<div class='er-card'><div class='er-mini-title'>Support card</div>"
            "<div class='er-empty'>Send a message to see the support path and resources.</div></div>"
        )
    route_label = str(result.get("route_label", "unknown"))
    safety_tier = str(result.get("safety_tier", "unknown"))
    should_intercept = bool(result.get("crisis") or result.get("should_intercept"))
    recommended_action = escape(str(result.get("recommended_action", "")))
    route_text = escape(_pretty_route(route_label))
    tier_text = escape(_pretty_tier(safety_tier))
    sources = result.get("retrieved_sources", []) or []

    path_class = "" if should_intercept else "accent"

    html = "<div class='er-card'>"
    html += "<div class='er-mini-title'>Support card</div>"
    html += "<div class='er-plan-rows'>"
    html += f"<div class='er-plan-row {path_class}'><span class='k'>Path</span><span class='v'>{route_text}</span></div>"
    html += f"<div class='er-plan-row'><span class='k'>Tier</span><span class='v'>{tier_text}</span></div>"
    if recommended_action:
        html += f"<div class='er-plan-row'><span class='k'>Next move</span><span class='v'>{recommended_action}</span></div>"
    html += "</div>"

    if sources:
        html += "<div class='er-mini-title' style='margin-top:18px;'>Resources</div>"
        html += "<div class='er-sources'>"
        for src in sources[:4]:
            title = escape(str(src.get("title") or src.get("source_name") or "Resource"))
            sname = escape(str(src.get("source_name") or ""))
            topic = escape(str(src.get("topic") or ""))
            risk = str(src.get("risk_level") or "")
            why = str(src.get("why_retrieved") or "matched prompt intent")
            url = escape(str(src.get("url") or ""))
            risk_cls = "crisis" if "crisis" in risk else ""
            html += "<div class='er-source'>"
            html += f"<div class='er-source-title'>{title}</div>"
            if sname and sname != title:
                html += f"<div class='er-source-name'>{sname}</div>"
            html += "<div class='er-source-tags'>"
            if topic: html += f"<span class='er-tag'>{escape(topic)}</span>"
            if risk: html += f"<span class='er-tag {risk_cls}'>{escape(risk)}</span>"
            html += "</div>"
            html += f"<div class='er-source-why'>{escape(_pretty_reason(why))}</div>"
            last_verified = str(src.get("last_verified") or "").strip()
            if url:
                html += "<div class='er-source-foot'>"
                html += f"<a href='{url}' target='_blank' rel='noopener'>Open ↗</a>"
                if last_verified:
                    html += f"<span class='er-source-verified' title='URL last verified on this date'>Verified {escape(last_verified)}</span>"
                html += "</div>"
            documents = src.get("documents") or []
            if documents:
                html += "<div class='er-source-docs'>"
                html += "<div class='er-source-docs-label'>Read directly</div>"
                for doc in documents:
                    d_title = escape(str(doc.get("title") or "Document"))
                    d_url = escape(str(doc.get("url") or ""))
                    d_type = escape(str(doc.get("document_type") or "guide"))
                    if not d_url:
                        continue
                    embeddable = bool(doc.get("embeddable")) and not bool(doc.get("requires_login"))
                    html += f"<div class='er-doc'><span class='er-doc-type'>{d_type}</span> "
                    html += f"<a href='{d_url}' target='_blank' rel='noopener'>{d_title} ↗</a>"
                    if embeddable:
                        # Inline iframe behind a <details> so the card stays compact
                        # by default but the doc is one click away.
                        html += (
                            f"<details class='er-doc-embed'><summary>Preview inline</summary>"
                            f"<iframe src='{d_url}' loading='lazy' "
                            f"sandbox='allow-same-origin allow-scripts allow-popups' "
                            f"title='{d_title}'></iframe></details>"
                        )
                    if doc.get("requires_login"):
                        html += " <span class='er-doc-meta'>(terpconnect login)</span>"
                    html += "</div>"
                html += "</div>"
            html += "</div>"
        html += "</div>"
    else:
        html += "<div class='er-mini-title' style='margin-top:18px;'>Resources</div>"
        html += "<div class='er-empty'>No external resource needed for this turn.</div>"
    html += "</div>"
    return html


def format_retrieval_panel(result=None) -> str:
    """Diagnostics. Pipeline internals for class & eval review."""
    if not result:
        return (
            "<div class='er-card'><div class='er-mini-title'>Diagnostics</div>"
            "<div class='er-empty'>Pipeline metadata appears here once a turn runs.</div></div>"
        )
    safety_tier = _pretty_tier(str(result.get("safety_tier", "unknown")))
    safety_reason = _pretty_reason(str(result.get("safety_reason", "")))
    corpus = str(result.get("retrieval_corpus", "unknown"))
    output_guard = result.get("output_guard", {}) or {}
    output_guard_reason = _pretty_reason(str(output_guard.get("reason", "not_checked")))
    guard_flags = output_guard.get("flags", []) or []
    safety_precheck = result.get("safety_precheck", {}) or {}
    precheck_reason = _pretty_reason(str(safety_precheck.get("reason", "not_recorded")))
    precheck_level = _pretty_precheck(
        str(safety_precheck.get("level", "unknown")),
        bool(result.get("crisis")),
    )
    classifier = result.get("classifier_confidence", {}) or {}
    route_conf = float(classifier.get("route", 0.0) or 0.0)
    tier_conf = float(classifier.get("tier", 0.0) or 0.0)
    classifier_label = "learned" if classifier.get("used_ml") else "fallback"
    retrieval_mode = _pretty_retrieval_mode(str(result.get("retrieval_mode", "")))
    latency = result.get("latency_ms", {}) or {}
    total_latency = float(latency.get("total_ms", 0.0) or 0.0)
    should_intercept = bool(result.get("crisis"))
    safety_cls = "danger" if should_intercept else ""
    guard_cls = "warn" if guard_flags else ""

    html = "<div class='er-card'>"
    html += "<div class='er-mini-title'>Diagnostics</div>"
    html += "<div class='er-diag-grid'>"
    html += f"<div class='er-diag {safety_cls}'><div class='k'>Safety check</div><div class='v'>{escape(precheck_level)}</div></div>"
    html += f"<div class='er-diag'><div class='k'>Tier</div><div class='v'>{escape(safety_tier)}</div></div>"
    html += f"<div class='er-diag'><div class='k'>Classifier</div><div class='v'>{classifier_label} · r {route_conf:.2f} / t {tier_conf:.2f}</div></div>"
    html += f"<div class='er-diag'><div class='k'>Retrieval</div><div class='v'>{escape(retrieval_mode or '—')}</div></div>"
    html += f"<div class='er-diag {guard_cls}'><div class='k'>Response check</div><div class='v'>{escape(output_guard_reason)}</div></div>"
    html += f"<div class='er-diag'><div class='k'>Speed</div><div class='v'>{total_latency:.0f} ms</div></div>"
    html += f"<div class='er-diag'><div class='k'>Corpus</div><div class='v'>{escape(corpus)}</div></div>"
    html += f"<div class='er-diag'><div class='k'>Safety reason</div><div class='v'>{escape(safety_reason or '—')}</div></div>"
    # Surface cross-cutting NLP flags for the grad-course audience.
    intl_flag = "yes" if result.get("international_concern") else "no"
    intl_cls = "warn" if result.get("international_concern") else ""
    stage_label = str(result.get("conversation_stage") or "—")
    html += f"<div class='er-diag {intl_cls}'><div class='k'>International concern</div><div class='v'>{escape(intl_flag)}</div></div>"
    html += f"<div class='er-diag'><div class='k'>Conversation stage</div><div class='v'>{escape(stage_label)}</div></div>"
    html += "</div>"

    notes = []
    if precheck_reason and precheck_reason not in {"—", "Not recorded"}:
        notes.append(f"Safety precheck: {escape(precheck_reason)}")
    escalation_reason = str(result.get("escalation_reason", ""))
    if escalation_reason:
        notes.append(f"Escalation: {escape(escalation_reason)}")
    if guard_flags:
        flag_text = ", ".join(escape(str(f)) for f in guard_flags)
        notes.append(f"Guard flags: {flag_text}")
    if notes:
        html += "<div class='er-source-why' style='margin-top:14px;line-height:1.7;'>" + "<br>".join(notes) + "</div>"
    html += "</div>"
    return html


def _stage_arc_text(stage: str, has_message: bool) -> tuple[str, str, int]:
    """Return (headline, sub, percent) for the conversation arc panel."""
    if not has_message:
        return (
            "Waiting for your first message",
            "Tell me what's on your mind. I'll listen first.",
            0,
        )
    if stage == "listen":
        return (
            "Listening to what's coming up",
            "Sitting with this before suggesting anything. You stay in charge of when to pivot.",
            28,
        )
    if stage == "permission":
        return (
            "Reflecting and quietly offering options",
            "I have a couple of places that could help, but only when you want them.",
            58,
        )
    if stage == "offer":
        return (
            "Working through this together",
            "Naming concrete next steps, with grounded UMD resources alongside.",
            88,
        )
    return ("Ready", "—", 0)


def _action_items_for(result: dict | None) -> list[str]:
    """Extract concrete \"things to try\" from the planner output."""
    if not result:
        return []
    items: list[str] = []
    rec = (result.get("recommended_action") or "").strip()
    if rec:
        items.append(rec)
    # Crisis path adds an emergency reminder
    if result.get("crisis"):
        items.append("Call or text 988 now. If immediate danger, call emergency services.")
    return items


def _render_safety_pipeline(result: dict | None) -> str:
    """Six-badge row visualizing each safety layer's status for this turn.

    Order mirrors the pipeline: Stage-1 -> Route -> Registry -> Stage ->
    Rephrase -> Safety verify -> Output guard. Each badge state:
      - on   (green) : layer ran and did what it should
      - hit  (red)   : layer intercepted / blocked something
      - skip (gray)  : layer intentionally skipped (e.g. listening stage)
      - off  (gray-dim): layer disabled or N/A
    """
    if not result:
        # Empty-state: show the layer names ghosted so the architecture is
        # legible even before the first message.
        slots = [
            ("S1", "Stage-1 safety", "off"),
            ("Route", "Route classifier", "off"),
            ("Reg", "Resource registry", "off"),
            ("Stage", "Conversation stage", "off"),
            ("Reph", "Rephraser", "off"),
            ("Guard", "Output guard", "off"),
        ]
    else:
        # Stage-1 lexical precheck
        precheck = result.get("safety_precheck", {}) or {}
        if precheck.get("should_intercept"):
            s1 = ("S1", f"Stage-1 INTERCEPTED: {precheck.get('reason','crisis')}", "hit")
        elif precheck.get("level") in ("wellbeing_support",):
            s1 = ("S1", f"Stage-1 flagged wellbeing: {precheck.get('reason','')}", "on")
        else:
            s1 = ("S1", f"Stage-1 pass: {precheck.get('reason','no_match')}", "on")

        # Route classifier
        route_label = result.get("route_label", "")
        classifier = result.get("classifier_confidence", {}) or {}
        route_conf = float(classifier.get("route", 0.0) or 0.0)
        used_ml = classifier.get("used_ml")
        route_state = "hit" if route_label == "crisis_immediate" else "on"
        route = ("Route",
                 f"Route: {route_label} (conf {route_conf:.2f}, {'ML' if used_ml else 'rule'})",
                 route_state)

        # Resource registry filter — count of sources surfaced
        sources = result.get("retrieved_sources", []) or []
        if sources:
            reg = ("Reg", f"{len(sources)} verified UMD/national resource(s) surfaced", "on")
        else:
            reg = ("Reg", "No resources surfaced (route doesn't need them)", "skip")

        # Conversation stage
        stage_val = result.get("conversation_stage", "—")
        stage_glyph = {"listen": "L", "permission": "P", "offer": "O", "clarify": "C", "offer (crisis)": "X"}.get(
            stage_val, stage_val[:1].upper() if stage_val else "?"
        )
        if result.get("crisis"):
            stage_state = "hit"
            stage_glyph = "X"
            stage_label = f"CRISIS — LLM bypassed, deterministic crisis template"
        elif stage_val == "clarify":
            stage_state = "skip"
            stage_label = "Clarify: short open-ended (output guard skipped)"
        elif stage_val == "offer":
            stage_state = "on"
            stage_label = "Offer: full plan + named resources"
        else:
            stage_state = "on"
            stage_label = f"{stage_val.title()}: listening / inviting"
        stage = (stage_glyph, stage_label, stage_state)

        # Rephraser
        provider = result.get("rephraser_provider", "deterministic")
        used_llm = bool(result.get("rephraser_used_llm"))
        rephraser_err = result.get("rephraser_last_error", "")
        if used_llm:
            reph = ("Reph", f"Paraphrased via {provider}", "on")
        elif provider == "deterministic_fallback":
            reph = ("Reph", f"FALLBACK to deterministic — {rephraser_err or 'unknown error'}", "hit")
        elif provider == "deterministic":
            reph = ("Reph", "Deterministic templates (rephraser off)", "skip")
        else:
            reph = ("Reph", f"Provider: {provider}", "on")

        # Output guard
        guard = result.get("output_guard", {}) or {}
        guard_flags = guard.get("flags", []) or []
        guard_reason = guard.get("reason", "")
        if guard_flags:
            grd = ("Guard", f"Guard flags: {', '.join(guard_flags)}", "hit")
        elif "disabled" in guard_reason:
            grd = ("Guard", "Output guard disabled (ablation)", "off")
        elif "listening_stage" in guard_reason or "minimal_response_clarify" == guard_reason:
            grd = ("Guard", f"Skipped at {stage_val} stage by design", "skip")
        elif guard_reason == "crisis_template":
            grd = ("Guard", "Crisis template (no guard needed)", "skip")
        else:
            grd = ("Guard", "Output guard passed", "on")

        slots = [s1, route, reg, stage, reph, grd]

    html = "<div class='er-safety-pipeline'>"
    html += "<div class='er-safety-label'>Safety pipeline</div>"
    html += "<div class='er-safety-row'>"
    for label, tooltip, state in slots:
        html += (
            f"<div class='er-safety-chip er-safety-{escape(state)}' "
            f"title='{escape(tooltip)}'>{escape(label)}</div>"
        )
    html += "</div></div>"
    return html


def format_live_context(result: dict | None = None, turn_index: int = 0) -> str:
    """Right-panel: arc + signals + resources + things-to-try (one HTML string)."""
    has_msg = bool(result)
    stage = (result or {}).get("conversation_stage", "")
    arc_head, arc_sub, arc_pct = _stage_arc_text(stage, has_msg)

    # Status pill in panel header
    if not has_msg:
        status_text = "ready"
        status_cls = ""
    elif result.get("crisis"):
        status_text = "safety intercept"
        status_cls = "danger"
    elif result.get("international_concern"):
        status_text = "intl context"
        status_cls = "warm"
    else:
        status_text = "active"
        status_cls = "active"

    # Active generation mode badge — visible without opening Diagnostics so
    # the user always knows which mode answered.
    rephraser_provider = (result or {}).get("rephraser_provider", "")
    used_llm = bool((result or {}).get("rephraser_used_llm"))
    rephraser_err = str((result or {}).get("rephraser_last_error", "")).strip()
    mode_title = ""
    if rephraser_provider:
        if used_llm:
            mode_label = rephraser_provider.split(":")[0]  # 'groq' / 'anthropic'
            mode_text = f"via {mode_label}"
            mode_cls = "active"
        elif rephraser_provider == "deterministic_fallback":
            # Make the fallback condition clearly visible: warning glyph + a
            # tooltip carrying the actual provider error so the user can tell
            # whether this is intentional (deterministic mode) or a failure.
            mode_text = "deterministic (fallback) ⚠"
            mode_cls = "warm fallback-warn"
            mode_title = (
                f"Live LLM rephrasing was unavailable for this turn — falling back to the deterministic template. "
                f"Last provider error: {rephraser_err or 'unknown'}"
            )
        else:
            mode_text = "deterministic"
            mode_cls = ""
    else:
        mode_text = ""
        mode_cls = ""

    parts: list[str] = []
    parts.append("<div class='er-context'>")
    parts.append(
        "<div class='er-ctx-head'>"
        "<div class='er-ctx-title'>Live thread</div>"
        + (
            (
                f"<div class='er-ctx-mode {mode_cls}' title='{escape(mode_title)}'>{escape(mode_text)}</div>"
                if mode_text else ""
            )
        )
        + f"<div class='er-ctx-status {status_cls}'>{escape(status_text)}</div>"
        "</div>"
    )

    # Safety-pipeline visualization: 6 layer badges showing what fired on this
    # turn. The point is to make the "defense in depth" story visible during
    # the demo without forcing the viewer to open Diagnostics. Each badge has
    # a tooltip with the layer's reason / status.
    parts.append(_render_safety_pipeline(result))

    # Arc
    parts.append(
        "<div class='er-arc'>"
        f"<div class='er-arc-text'>{escape(arc_head)}</div>"
        f"<div class='er-arc-sub'>{escape(arc_sub)}</div>"
        f"<div class='er-arc-meter'><div style='width:{arc_pct}%'></div></div>"
        "</div>"
    )

    # Signals
    signals: list[str] = []
    if has_msg:
        route = result.get("route_label", "")
        tier = result.get("safety_tier", "")
        if route and route != "general_student_support":
            signals.append(f"<span class='er-signal route'>{escape(_pretty_route(route))}</span>")
        if stage:
            signals.append(f"<span class='er-signal stage'>{escape(stage.title())}</span>")
        if tier in {"high_distress", "imminent_safety"}:
            tier_cls = "tier-danger" if tier == "imminent_safety" else "tier-warm"
            signals.append(f"<span class='er-signal {tier_cls}'>{escape(_pretty_tier(tier))}</span>")
        if result.get("international_concern"):
            signals.append("<span class='er-signal intl'>F-1 / international context</span>")
    if signals:
        parts.append("<div class='er-signals'>" + "".join(signals) + "</div>")

    # Resources
    sources = (result or {}).get("retrieved_sources", []) or []
    parts.append("<div class='er-ctx-section'>")
    parts.append(
        "<div class='er-ctx-section-head'>"
        "<h4 class='er-section-title'>Resources building</h4>"
        f"<span class='er-count-pill'>{len(sources)} found</span>"
        "</div>"
    )
    if sources:
        parts.append("<div class='er-resources'>")
        for i, src in enumerate(sources[:5]):
            title = escape(str(src.get("source_name") or src.get("title") or "Resource"))
            why = escape(_pretty_reason(str(src.get("why_retrieved") or "matched the prompt")))
            url = escape(str(src.get("url") or ""))
            risk = str(src.get("risk_level") or "")
            cls = "er-rsrc"
            if "international" in title.lower() or "isss" in title.lower():
                cls += " featured"
            elif "crisis" in risk:
                cls += " crisis"
            inner = (
                f"<div class='{cls}'>"
                f"<div class='er-rsrc-title'>{title}</div>"
                f"<div class='er-rsrc-why'>{why}</div>"
            )
            if url:
                inner += f"<a href='{url}' target='_blank' rel='noopener'>Open ↗</a>"
            inner += "</div>"
            parts.append(inner)
        parts.append("</div>")
    else:
        if has_msg and stage == "listen":
            parts.append("<div class='er-empty'>Resources stay quiet while we're still listening.</div>")
        elif has_msg:
            parts.append("<div class='er-empty'>No external resource needed for this turn.</div>")
        else:
            parts.append("<div class='er-empty'>Resources will appear here as we talk.</div>")
    parts.append("</div>")

    # Things to try
    actions = _action_items_for(result)
    parts.append("<div class='er-ctx-section'>")
    parts.append("<h4 class='er-section-title'>Things to try</h4>")
    if actions:
        parts.append("<div class='er-actions'>")
        for action in actions[:3]:
            parts.append(f"<div class='er-action'>{escape(action)}</div>")
        parts.append("</div>")
    elif has_msg and stage == "listen":
        parts.append("<div class='er-empty'>Suggestions will appear when you're ready.</div>")
    else:
        parts.append("<div class='er-empty'>None yet.</div>")
    parts.append("</div>")

    parts.append("</div>")  # close er-context
    return "".join(parts)


def format_studio_diagnostics(result: dict | None = None) -> str:
    """Diagnostics accordion content for the grad-NLP audience."""
    if not result:
        return (
            "<div class='er-empty'>Pipeline metadata appears here once a turn runs.</div>"
        )

    safety_tier = _pretty_tier(str(result.get("safety_tier", "unknown")))
    safety_reason = _pretty_reason(str(result.get("safety_reason", "")))
    corpus = str(result.get("retrieval_corpus", "unknown"))
    output_guard = result.get("output_guard", {}) or {}
    output_guard_reason = _pretty_reason(str(output_guard.get("reason", "not_checked")))
    guard_flags = output_guard.get("flags", []) or []
    safety_precheck = result.get("safety_precheck", {}) or {}
    precheck_level = _pretty_precheck(
        str(safety_precheck.get("level", "unknown")),
        bool(result.get("crisis")),
    )
    classifier = result.get("classifier_confidence", {}) or {}
    route_conf = float(classifier.get("route", 0.0) or 0.0)
    tier_conf = float(classifier.get("tier", 0.0) or 0.0)
    classifier_label = "learned" if classifier.get("used_ml") else "fallback"
    retrieval_mode = _pretty_retrieval_mode(str(result.get("retrieval_mode", "")))
    latency = result.get("latency_ms", {}) or {}
    total_latency = float(latency.get("total_ms", 0.0) or 0.0)
    intl = "yes" if result.get("international_concern") else "no"
    intl_cls = "warn" if result.get("international_concern") else ""
    stage = str(result.get("conversation_stage") or "—")
    turn_idx = int(result.get("turn_index") or 0)
    safety_cls = "danger" if result.get("crisis") else ""
    guard_cls = "warn" if guard_flags else ""

    rephraser_provider = str(result.get("rephraser_provider") or "deterministic")
    rephraser_used_llm = bool(result.get("rephraser_used_llm"))
    rephraser_latency = float(result.get("rephraser_latency_ms") or 0.0)
    rephraser_cls = "accent" if rephraser_used_llm else ""

    rows = [
        f"<div class='er-diag {safety_cls}'><div class='k'>Safety check</div><div class='v'>{escape(precheck_level)}</div></div>",
        f"<div class='er-diag'><div class='k'>Tier</div><div class='v'>{escape(safety_tier)}</div></div>",
        f"<div class='er-diag accent'><div class='k'>Stage · turn</div><div class='v'>{escape(stage)} · t{turn_idx}</div></div>",
        f"<div class='er-diag {intl_cls}'><div class='k'>F-1 / intl signal</div><div class='v'>{escape(intl)}</div></div>",
        f"<div class='er-diag {rephraser_cls}'><div class='k'>Rephraser</div><div class='v'>{escape(rephraser_provider)}</div></div>",
        f"<div class='er-diag'><div class='k'>Rephrase latency</div><div class='v'>{rephraser_latency:.0f} ms</div></div>",
        f"<div class='er-diag'><div class='k'>Classifier</div><div class='v'>{classifier_label} · r {route_conf:.2f} / t {tier_conf:.2f}</div></div>",
        f"<div class='er-diag'><div class='k'>Retrieval</div><div class='v'>{escape(retrieval_mode or '—')}</div></div>",
        f"<div class='er-diag {guard_cls}'><div class='k'>Output guard</div><div class='v'>{escape(output_guard_reason)}</div></div>",
        f"<div class='er-diag'><div class='k'>Total latency</div><div class='v'>{total_latency:.0f} ms</div></div>",
        f"<div class='er-diag'><div class='k'>Corpus</div><div class='v'>{escape(corpus)}</div></div>",
        f"<div class='er-diag'><div class='k'>Safety reason</div><div class='v'>{escape(safety_reason or '—')}</div></div>",
    ]

    html = "<div class='er-diag-grid'>" + "".join(rows) + "</div>"

    # IG tokens if available
    safety_explanation = result.get("safety_explanation", {}) or {}
    ig_tokens = safety_explanation.get("ig_tokens") or []
    if result.get("crisis") and ig_tokens:
        valid = [(t, s) for t, s in ig_tokens if t.strip()][:10]
        if valid:
            html += "<div style='margin-top:14px;'><div class='k' style='font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-dim);font-weight:500;margin-bottom:6px;'>Top crisis signals (Integrated Gradients)</div>"
            html += "<div class='er-ig-row'>"
            for tok, _score in valid:
                html += f"<span class='er-ig'>{escape(tok)}</span>"
            html += "</div></div>"

    # Notes
    notes = []
    rephraser_error = str(result.get("rephraser_last_error") or "").strip()
    if rephraser_error:
        notes.append(f"Rephraser error: {escape(rephraser_error[:240])}")
    if guard_flags:
        flag_text = ", ".join(escape(str(f)) for f in guard_flags)
        notes.append(f"Guard flags: {flag_text}")
    escalation_reason = str(result.get("escalation_reason", ""))
    if escalation_reason:
        notes.append(f"Escalation: {escape(escalation_reason)}")
    if notes:
        html += (
            "<div style='margin-top:14px;font-size:11.5px;color:var(--text-dim);line-height:1.7;'>"
            + "<br>".join(notes) + "</div>"
        )

    return html


TYPING_HTML = "<span class='er-typing'><span></span><span></span><span></span></span>"
STREAM_ENABLED = os.getenv("EMPATHRAG_STREAM", "1") != "0"
STREAM_WORDS_PER_CHUNK = int(os.getenv("EMPATHRAG_STREAM_WORDS", "2"))
STREAM_CHUNK_DELAY_MS = int(os.getenv("EMPATHRAG_STREAM_DELAY_MS", "75"))
TYPING_DELAY_MS = int(os.getenv("EMPATHRAG_TYPING_DELAY_MS", "650"))


def _stream_chunks(full_text: str):
    if not STREAM_ENABLED or not full_text:
        yield full_text
        return
    words = full_text.split(" ")
    if len(words) <= STREAM_WORDS_PER_CHUNK:
        yield full_text
        return
    import time as _t
    cursor = STREAM_WORDS_PER_CHUNK
    while cursor < len(words):
        yield " ".join(words[:cursor])
        _t.sleep(STREAM_CHUNK_DELAY_MS / 1000.0)
        cursor += STREAM_WORDS_PER_CHUNK
    yield full_text


def respond(message, chat_history, session_state, audience_mode, rephrase_mode="deterministic"):
    """Yield (chatbot, context_html, diag_html, session_id, session_state)."""
    if not session_state:
        session_state = new_session_state()

    # Per-turn override of the rephraser. The UI toggle picks "deterministic"
    # or "llm"; we set the env var the rephraser reads. Done before pipeline.run.
    os.environ["EMPATHRAG_REPHRASER_ENABLED"] = "1" if rephrase_mode == "llm" else "0"

    emotion_history = session_state["emotion_history"]
    session_id = session_state["session_id"]
    turn_count = len(chat_history)

    if not message.strip():
        yield (
            chat_history,
            format_live_context(None, turn_count),
            format_studio_diagnostics(None),
            session_id,
            session_state,
        )
        return

    # Show user msg + typing dots immediately, with provisional "thinking" arc.
    chat_history = list(chat_history) + [(message, TYPING_HTML)]
    provisional = {
        "conversation_stage": "listen",
        "route_label": "",
        "safety_tier": "",
        "international_concern": False,
        "retrieved_sources": [],
        "recommended_action": "",
        "crisis": False,
    }
    yield (
        chat_history,
        format_live_context(provisional, turn_count + 1),
        format_studio_diagnostics(None),
        session_id,
        session_state,
    )

    # When LLM rephrasing is enabled and the active pipeline supports real
    # token streaming, we consume the streaming generator and update the chat
    # bubble per arrived token. Otherwise we fall back to the synchronous
    # path with the legacy fake word-chunk reveal.
    use_real_streaming = (
        rephrase_mode == "llm"
        and STREAM_ENABLED
        and hasattr(get_pipeline(), "run_streaming")
        and not hasattr(get_pipeline(), "tracker")
    )

    # Typing-dot delay is purely cosmetic. In deterministic mode the compute
    # is instantaneous so the dots have to stand in as "thinking" time. In
    # real-streaming mode the LLM's own first-token latency (~300-500ms) is
    # the thinking time — adding 650ms more before the stream starts just
    # delays first-token visibility for no UX gain.
    if STREAM_ENABLED and TYPING_DELAY_MS > 0 and not use_real_streaming:
        import time as _t
        _t.sleep(TYPING_DELAY_MS / 1000.0)

    provisional_context = format_live_context(provisional, turn_count + 1)

    with pipeline_lock:
        active_pipeline = get_pipeline()
        if hasattr(active_pipeline, "tracker"):
            active_pipeline.tracker.reset()
            for label in session_state.get("tracker_history", []):
                active_pipeline.tracker.update(label, token_count=5)
            active_pipeline.conv_history = list(session_state.get("conv_history", []))
            original_check = active_pipeline.guardrail.check
            def fast_check(text, threshold=0.5, skip_ig=False):
                return original_check(text, threshold=threshold, skip_ig=True)
            active_pipeline.guardrail.check = fast_check
            result = active_pipeline.run(message)
            active_pipeline.guardrail.check = original_check
            session_state["tracker_history"] = active_pipeline.tracker.history()
            session_state["conv_history"] = list(active_pipeline.conv_history)
        elif use_real_streaming:
            result = None
            for ev in active_pipeline.run_streaming(message, audience_mode=audience_mode or "student"):
                if ev[0] == "token":
                    chat_history[-1] = (message, ev[1])
                    yield (
                        chat_history,
                        provisional_context,
                        format_studio_diagnostics(None),
                        session_id,
                        session_state,
                    )
                elif ev[0] == "done":
                    result = ev[1]
            session_state["tracker_history"] = session_state.get("tracker_history", []) + [result["emotion"]]
            session_state["conv_history"] = session_state.get("conv_history", [])
        else:
            result = active_pipeline.run(message, audience_mode=audience_mode or "student")
            session_state["tracker_history"] = session_state.get("tracker_history", []) + [result["emotion"]]
            session_state["conv_history"] = session_state.get("conv_history", [])

    full_response = result["response"]
    emotion_history.append(
        {
            "turn": len(emotion_history) + 1,
            "label_name": result["emotion_name"],
            "color": LABEL_COLORS[result["emotion_name"]],
        }
    )
    log_turn(session_id, len(emotion_history), message, result)

    # Per-turn record for the Support Plan export. Stored only in the user's
    # browser-side gr.State; never persisted server-side.
    session_state.setdefault("turn_log", []).append(
        {
            "turn_index": len(emotion_history),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "user_message": message,
            "route_label": result.get("route_label", ""),
            "safety_tier": result.get("safety_tier", ""),
            "conversation_stage": result.get("conversation_stage", ""),
            "recommended_action": result.get("recommended_action", ""),
            "international_concern": bool(result.get("international_concern")),
            "intl_topic": result.get("intl_topic", ""),
            "retrieved_sources": result.get("retrieved_sources", []) or [],
        }
    )

    context_html = format_live_context(result, turn_count + 1)
    diag_html = format_studio_diagnostics(result)

    if use_real_streaming:
        # Tokens were already streamed into the bubble. Yield once more with
        # the final body (in case the output guard corrected it post-stream)
        # plus the now-populated context + diagnostics.
        chat_history[-1] = (message, full_response)
        yield (
            chat_history,
            context_html,
            diag_html,
            session_id,
            session_state,
        )
    else:
        for partial in _stream_chunks(full_response):
            chat_history[-1] = (message, partial)
            yield (
                chat_history,
                context_html,
                diag_html,
                session_id,
                session_state,
            )

    # Optional crisis IG re-yield
    is_crisis = bool(result.get("crisis"))
    safety_explanation = result.get("safety_explanation", {}) or {}
    explanation_available = bool(safety_explanation.get("available"))
    if is_crisis and hasattr(get_pipeline(), "guardrail") and not explanation_available:
        with pipeline_lock:
            active_pipeline = get_pipeline()
            if hasattr(active_pipeline, "guardrail"):
                _, confidence, ig_tokens = active_pipeline.guardrail.check(
                    message, threshold=0.5, skip_ig=False
                )
                if "safety_explanation" not in result:
                    result["safety_explanation"] = {}
                result["safety_explanation"]["ig_tokens"] = ig_tokens
                result["crisis_confidence"] = confidence
        yield (
            chat_history,
            format_live_context(result, turn_count + 1),
            format_studio_diagnostics(result),
            session_id,
            session_state,
        )


def transcribe_voice(audio_path, current_msg):
    """Transcribe a recorded clip via Groq Whisper, drop into the composer.

    Returns updates for (msg_box, voice_status). The clip is NOT auto-sent —
    the user reviews the transcript and clicks send themselves. If the
    composer already has text, the transcript is appended on a new line so a
    user mixing typing and dictation doesn't lose what they had.
    """
    from pipeline.voice import GroqWhisperTranscriber

    if not audio_path:
        return gr.update(), gr.update(value=(
            "<div class='er-voice-status'>No audio recorded. Tap the mic to try again.</div>"
        ))
    transcriber = GroqWhisperTranscriber()
    if not transcriber.available():
        return gr.update(), gr.update(value=(
            "<div class='er-voice-status er-voice-error'>Voice transcription unavailable: GROQ_API_KEY is not set.</div>"
        ))
    result = transcriber.transcribe(audio_path)
    if not result.ok():
        return gr.update(), gr.update(value=(
            f"<div class='er-voice-status er-voice-error'>Transcription failed ({result.error}). Try again or type instead.</div>"
        ))
    new_text = result.text
    if current_msg and current_msg.strip():
        new_text = current_msg.rstrip() + "\n" + new_text
    return (
        gr.update(value=new_text),
        gr.update(value=(
            f"<div class='er-voice-status er-voice-ok'>Transcribed in {int(result.latency_ms)}ms via Groq Whisper. Review and send when ready.</div>"
        )),
    )


def _support_plan_started_at(session_state):
    import datetime as _dt
    started_iso = (session_state or {}).get("started_at")
    try:
        return _dt.datetime.fromisoformat(started_iso) if started_iso else None
    except (TypeError, ValueError):
        return None


def export_support_plan_md(session_state):
    """Markdown support plan — for internal review / our dev use."""
    from pipeline.support_plan import build_support_plan_markdown
    import datetime as _dt
    import tempfile

    turn_log = (session_state or {}).get("turn_log", [])
    md = build_support_plan_markdown(turn_log, started_at=_support_plan_started_at(session_state))
    sid_short = (session_state or {}).get("session_id", "session")[:8]
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fd, path = tempfile.mkstemp(prefix=f"empathrag_support_plan_{sid_short}_{stamp}_", suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(md)
    return gr.update(value=path, visible=True)


def export_support_plan_pdf(session_state):
    """PDF support plan — counselor-friendly format. Falls through to the
    Markdown export if fpdf2 isn't installed, so the path doesn't go dark."""
    import datetime as _dt
    import tempfile

    turn_log = (session_state or {}).get("turn_log", [])
    sid_short = (session_state or {}).get("session_id", "session")[:8]
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    try:
        from pipeline.support_plan import build_support_plan_pdf
        fd, path = tempfile.mkstemp(prefix=f"empathrag_support_plan_{sid_short}_{stamp}_", suffix=".pdf")
        os.close(fd)
        build_support_plan_pdf(turn_log, path, started_at=_support_plan_started_at(session_state))
        return gr.update(value=path, visible=True)
    except ImportError:
        return export_support_plan_md(session_state)


def reset_session_handler():
    session_state = new_session_state()
    return (
        [],
        format_live_context(None, 0),
        format_studio_diagnostics(None),
        session_state["session_id"],
        session_state,
    )


def set_prompt(prompt: str) -> str:
    return prompt


def _pretty_route(route: str) -> str:
    return {
        "academic_setback": "Academic setback",
        "exam_stress": "Test or exam stress",
        "accessibility_ads": "Accessibility accommodations",
        "advisor_conflict": "Advisor or graduate conflict",
        "counseling_navigation": "Counseling navigation",
        "basic_needs": "Basic needs support",
        "care_violence_confidential": "Confidential CARE support",
        "peer_helper": "Helping someone else",
        "loneliness_isolation": "Loneliness or isolation",
        "anxiety_panic": "Anxiety or panic",
        "low_mood": "Low mood support",
        "crisis_immediate": "Immediate safety handoff",
        "general_student_support": "General student support",
        "out_of_scope": "Outside support scope",
    }.get(route, route.replace("_", " ").title())


def _pretty_tier(tier: str) -> str:
    return {
        "imminent_safety": "Immediate safety",
        "high_distress": "High distress",
        "support_navigation": "Support navigation",
        "wellbeing": "Wellbeing",
        "pass": "No urgent safety flag",
        "crisis": "Immediate safety",
        "emergency": "Emergency safety",
    }.get(tier, tier.replace("_", " ").title())


def _pretty_precheck(level: str, should_intercept: bool) -> str:
    if should_intercept:
        return "Human support now"
    return {
        "pass": "No urgent safety flag",
        "wellbeing_support": "Supportive check",
        "crisis": "Human support now",
        "emergency": "Emergency handoff",
    }.get(level, _pretty_tier(level))


def _pretty_reason(reason: str) -> str:
    if not reason:
        return "Ready"
    return {
        "below_support_threshold": "No urgent safety signal detected",
        "passed_output_guard": "Response passed safety check",
        "disabled": "Off for fast demo",
        "not_checked": "Not checked",
        "not_recorded": "Not recorded",
        "resource registry route match": "Matched this support path",
        "curated retrieval match": "Matched the prompt",
        "exam_stress_language": "Test or exam stress language",
        "high_distress_language": "Distress language increased the tier",
        "wellbeing_support_language": "Low-risk coping support",
        "dependency_or_secrecy_redirect": "Dependency or secrecy needs human support",
        "stage1_intercept": "Handled by the safety precheck",
    }.get(reason, reason.replace("_", " "))


def _pretty_retrieval_mode(mode: str) -> str:
    if "crisis_only" in mode:
        return "crisis-only"
    if "registry_filtered" in mode:
        return "resource-filtered"
    return mode.replace("_", " ")


theme = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="teal",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0a0c10",
    body_background_fill_dark="#0a0c10",
    body_text_color="#e7ecf2",
    background_fill_primary="#0a0c10",
    background_fill_secondary="#11151c",
    border_color_primary="rgba(255,255,255,0.06)",
    button_primary_background_fill="#5eead4",
    button_primary_background_fill_hover="#5eead4",
    button_primary_text_color="#061a16",
    button_secondary_background_fill="transparent",
    button_secondary_text_color="#8a93a3",
    input_background_fill="#11151c",
    input_border_color="rgba(255,255,255,0.06)",
    block_background_fill="transparent",
    block_border_color="rgba(255,255,255,0.06)",
    block_label_background_fill="transparent",
    block_label_text_color="#8a93a3",
)


# Force-scroll the chat surface to bottom whenever its content mutates.
# Gradio's gr.Chatbot doesn't reliably auto-scroll during streaming updates
# (especially with many small token yields); a MutationObserver makes the
# behavior reliable across versions.
_CHATBOT_AUTOSCROLL_JS = """
() => {
  const tryAttach = () => {
    const containers = document.querySelectorAll('.er-chat .wrap, .er-chat .bubble-wrap, .er-chat > div, .er-chat');
    let target = null;
    for (const c of containers) {
      if (c && c.scrollHeight > c.clientHeight) { target = c; break; }
    }
    if (!target) {
      const chat = document.querySelector('.er-chat');
      if (chat) {
        target = chat.querySelector('[role="log"]') || chat;
      }
    }
    if (!target) { setTimeout(tryAttach, 400); return; }
    const scroll = () => { target.scrollTop = target.scrollHeight; };
    const obs = new MutationObserver(() => { requestAnimationFrame(scroll); });
    obs.observe(target, { childList: true, subtree: true, characterData: true });
    scroll();
  };
  tryAttach();
}
"""

with gr.Blocks(theme=theme, title="EmpathRAG Studio", css=APP_CSS, js=_CHATBOT_AUTOSCROLL_JS) as demo:
    initial_state = new_session_state()
    session_state = gr.State(value=initial_state)

    # ---- Top bar (sticky) ----
    with gr.Row(elem_classes=["er-topbar"]):
        gr.HTML(
            """
            <div class="er-brand">
              <span class="er-brand-dot"></span>
              EmpathRAG
              <span class="er-brand-meta">Studio</span>
            </div>
            """
        )
        audience_mode_box = gr.Radio(
            choices=[("Student", "student"), ("Helping a friend", "helping_friend")],
            value="student",
            show_label=False,
            container=False,
            elem_classes=["er-mode-wrap"],
        )
        export_pdf_btn = gr.Button("⬇  PDF (for counselor)", elem_classes=["er-export-btn"])
        export_md_btn = gr.Button("⬇  Markdown", elem_classes=["er-export-btn"])
        reset_btn = gr.Button("↺  New conversation", elem_classes=["er-reset-btn"])

    # File component for the generated support plan; appears after Save click.
    support_plan_file = gr.File(
        label="Your support plan",
        visible=False,
        interactive=False,
        elem_classes=["er-support-plan-file"],
    )

    # ---- Mode bar: ablation toggle for the demo ----
    with gr.Row(elem_classes=["er-modebar"]):
        gr.HTML(
            '<div class="er-modebar-label">'
            '<span class="er-modebar-title">Generation</span>'
            '<span class="er-modebar-help">Flip to compare. Diagnostics tracks which provider answered.</span>'
            '</div>'
        )
        rephraser_mode_box = gr.Radio(
            choices=[
                ("Deterministic templates", "deterministic"),
                ("LLM-rephrased (Groq)", "llm"),
            ],
            value="llm" if os.getenv("EMPATHRAG_REPHRASER_ENABLED", "0") != "0" else "deterministic",
            show_label=False,
            container=False,
            elem_classes=["er-mode-wrap", "er-rephrase-toggle"],
        )

    # ---- Studio: strict CSS grid (chat 1fr · context 360px) ----
    with gr.Row(elem_classes=["er-studio"], equal_height=True):
        with gr.Column(elem_classes=["er-chat-col"]):
            hero_block = gr.HTML(
                """
                <div class="er-hero">
                  <h1>How are you doing today?</h1>
                  <p>I listen first. About academic stress, mental health, advisor pressure, F-1 / visa worry, or anything weighing on you. When you're ready, I'll point you toward specific UMD resources. Not before.</p>
                  <div class="er-hero-meta">Conversations are not logged · Not therapy or emergency care</div>
                </div>
                """,
                visible=True,
            )

            chatbot = gr.Chatbot(
                elem_classes=["er-chat"],
                show_label=False,
                bubble_full_width=False,
                avatar_images=None,
                show_share_button=False,
                show_copy_button=True,
                sanitize_html=False,
                # Pixel height — Gradio's native sizing parameter. The chatbot
                # scrolls internally past this height. Composer/dock flow
                # naturally below in the document.
                height=520,
            )

            # ---- Bottom dock: divider · chips · input ----
            with gr.Column(elem_classes=["er-dock"]):
                gr.HTML("<div class='er-dock-divider'></div>")

                with gr.Row(elem_classes=["er-chips"], visible=True) as chip_row:
                    chip_counseling = gr.Button("Thinking about counseling", elem_classes=["er-chip-btn"])
                    chip_ads = gr.Button("ADS accommodations", elem_classes=["er-chip-btn"])
                    chip_advisor = gr.Button("Advisor conflict", elem_classes=["er-chip-btn"])
                    chip_intl = gr.Button("F-1 visa & academic worry", elem_classes=["er-chip-btn"])
                    chip_grounding = gr.Button("Pre-exam grounding", elem_classes=["er-chip-btn"])

                with gr.Group(elem_classes=["er-composer-wrap"]):
                    msg_box = gr.Textbox(
                        placeholder="Tell me what's on your mind...",
                        show_label=False,
                        container=False,
                        lines=1,
                        max_lines=4,
                        autofocus=True,
                    )
                    send_btn = gr.Button("→", elem_classes=["er-send-btn"], variant="primary")

                # Voice input is provisional — typers don't see it; clickers
                # toggle it open. Default hidden so the composer stays clean.
                voice_toggle_btn = gr.Button(
                    "🎤  Use voice instead",
                    elem_classes=["er-voice-toggle"],
                )
                with gr.Row(elem_classes=["er-voice-row"], visible=False) as voice_row:
                    voice_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        show_label=False,
                        container=False,
                        elem_classes=["er-mic"],
                        format="wav",
                    )
                    voice_status = gr.HTML(
                        "<div class='er-voice-status'>Tap the mic, record, then stop. Transcript lands in the composer.</div>",
                        elem_classes=["er-voice-status-wrap"],
                    )

                gr.HTML(
                    "<div class='er-footnote'>If you are in immediate danger, call or text 988.</div>"
                )

        with gr.Column(elem_classes=["er-context-col"]):
            context_block = gr.HTML(value=format_live_context(None, 0))
            with gr.Accordion(
                "Diagnostics · NLP signals",
                open=False,
                elem_classes=["er-diag-acc"],
            ):
                diag_block = gr.HTML(value=format_studio_diagnostics(None))

    # Hidden state surfaces (kept to preserve respond() output contract)
    session_id_box = gr.Textbox(value=initial_state["session_id"], visible=False)

    # ---- Wiring ----
    # Only the hero (welcome state) is hidden after the first message.
    # Chips remain available throughout the conversation as quick prompts.
    submit_outputs = [
        chatbot,
        context_block,
        diag_block,
        session_id_box,
        session_state,
        hero_block,
    ]

    def respond_with_chrome(message, chat_history, session_state, audience_mode, rephrase_mode):
        hide_hero = bool(message and message.strip())
        hero_chrome = (gr.update(visible=not hide_hero),)
        for tup in respond(message, chat_history, session_state, audience_mode, rephrase_mode):
            yield tup + hero_chrome

    # gr.update(value="") explicitly preserves the placeholder attribute on
    # the underlying textarea, which a bare lambda: "" can lose between turns.
    def _clear_input():
        return gr.update(value="", placeholder="Tell me what's on your mind...")

    msg_box.submit(
        respond_with_chrome,
        inputs=[msg_box, chatbot, session_state, audience_mode_box, rephraser_mode_box],
        outputs=submit_outputs,
    ).then(_clear_input, outputs=msg_box)

    send_btn.click(
        respond_with_chrome,
        inputs=[msg_box, chatbot, session_state, audience_mode_box, rephraser_mode_box],
        outputs=submit_outputs,
    ).then(_clear_input, outputs=msg_box)

    def reset_with_chrome():
        base = reset_session_handler()
        return base + (gr.update(visible=True),)

    reset_btn.click(reset_with_chrome, outputs=submit_outputs)

    export_pdf_btn.click(export_support_plan_pdf, inputs=[session_state], outputs=[support_plan_file])
    export_md_btn.click(export_support_plan_md, inputs=[session_state], outputs=[support_plan_file])

    # Voice input is provisional. Toggle reveals/hides the recorder row.
    voice_toggle_state = gr.State(value=False)

    def toggle_voice(currently_visible: bool):
        new_visible = not currently_visible
        return (
            gr.update(visible=new_visible),
            gr.update(value="🎤  Hide voice input" if new_visible else "🎤  Use voice instead"),
            new_visible,
        )

    voice_toggle_btn.click(
        toggle_voice,
        inputs=[voice_toggle_state],
        outputs=[voice_row, voice_toggle_btn, voice_toggle_state],
    )

    # When the user finishes recording, auto-transcribe and drop the text into
    # the composer (not auto-sent — user reviews).
    voice_input.stop_recording(
        transcribe_voice,
        inputs=[voice_input, msg_box],
        outputs=[msg_box, voice_status],
    )

    chip_counseling.click(
        lambda: set_prompt("I think I need counseling at UMD, but I don't know how to start."),
        outputs=msg_box,
    )
    chip_ads.click(
        lambda: set_prompt("I need disability accommodations for an upcoming exam at UMD."),
        outputs=msg_box,
    )
    chip_advisor.click(
        lambda: set_prompt("My advisor keeps dismissing my concerns and I need someone neutral to talk to."),
        outputs=msg_box,
    )
    chip_grounding.click(
        lambda: set_prompt("I am panicking before my exam. Can you help me with a grounding exercise?"),
        outputs=msg_box,
    )
    chip_intl.click(
        lambda: set_prompt(
            "I'm an F-1 student and I think I'm going to fail my final tomorrow. "
            "I'm scared about what this means for my visa status."
        ),
        outputs=msg_box,
    )



if __name__ == "__main__":
    os.makedirs("eval", exist_ok=True)
    # Print provider availability so it's obvious whether GROQ_API_KEY /
    # ANTHROPIC_API_KEY actually loaded.
    try:
        from pipeline.rephraser import ResponseRephraser as _RR
        _r = _RR()
        print("[rephraser] provider availability:")
        for _p in _r.providers:
            print(f"  - {_p.name}: {'available' if _p.available() else 'unavailable'}")
    except Exception as _e:
        print(f"[rephraser] probe failed: {_e}")
    demo.launch(share=SHARE_DEMO)
