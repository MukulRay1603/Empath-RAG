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
  --border: rgba(255,255,255,0.06);
  --border-mid: rgba(255,255,255,0.10);
  --border-strong: rgba(255,255,255,0.16);
  --accent: #5eead4;
  --accent-dim: #2dd4bf;
  --accent-soft: rgba(94,234,212,0.10);
  --accent-line: rgba(94,234,212,0.22);
  --accent-glow: rgba(94,234,212,0.20);
  --text: #e7ecf2;
  --text-muted: #8a93a3;
  --text-dim: #5a6373;
  --warm: #f5b669;
  --warm-soft: rgba(245,182,105,0.10);
  --danger: #f87171;
  --radius-sm: 8px;
  --radius: 12px;
  --radius-lg: 16px;
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
    radial-gradient(900px 480px at 50% -20%, rgba(94,234,212,0.07), transparent 70%),
    radial-gradient(600px 340px at 90% 110%, rgba(94,234,212,0.04), transparent 70%);
}

.gradio-container {
  position: relative; z-index: 1;
  background: transparent !important;
  max-width: 880px !important;
  margin: 0 auto !important;
  padding: 0 24px 48px !important;
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

/* TOP BAR */
.er-topbar {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 16px !important;
  padding: 22px 0 18px !important;
  margin: 0 0 8px !important;
  border-bottom: 1px solid var(--border) !important;
  flex-wrap: nowrap !important;
}
.er-topbar > * { flex: none !important; }
.er-topbar > .er-mode-wrap { flex: 1 1 auto !important; display: flex; justify-content: center; }

.er-brand {
  display: flex; align-items: center; gap: 10px;
  font-weight: 600; font-size: 15.5px; letter-spacing: -0.01em;
  color: var(--text);
}
.er-brand-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 14px var(--accent-glow);
  animation: er-pulse 2.4s ease-in-out infinite;
}
.er-brand-meta {
  color: var(--text-dim); font-size: 12.5px; font-weight: 400; margin-left: 4px;
}
@keyframes er-pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 14px var(--accent-glow); }
  50%      { opacity: 0.6; box-shadow: 0 0 6px var(--accent-glow); }
}

/* SEGMENTED MODE TOGGLE (Radio) */
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
  padding: 7px 16px !important;
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

/* INSPECT BUTTON */
.gradio-container .er-inspect-btn { min-width: 0 !important; }
.gradio-container .er-inspect-btn button {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-muted) !important;
  padding: 8px 16px !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  border-radius: 999px !important;
  min-width: 0 !important;
  transition: border-color 180ms ease, color 180ms ease, background 180ms ease;
  box-shadow: none !important;
}
.gradio-container .er-inspect-btn button:hover {
  border-color: var(--border-strong) !important;
  color: var(--text) !important;
  background: var(--surface) !important;
}

/* HERO (empty state) */
.er-hero {
  text-align: center;
  padding: 84px 12px 28px;
}
.er-hero h1 {
  font-size: 30px;
  font-weight: 500;
  letter-spacing: -0.025em;
  margin: 0 0 12px;
  color: var(--text);
  line-height: 1.2;
}
.er-hero p {
  color: var(--text-muted);
  font-size: 14px;
  margin: 0 auto;
  max-width: 480px;
  line-height: 1.6;
}

/* SUGGESTION CHIPS */
.er-chips {
  display: flex !important;
  gap: 8px !important;
  flex-wrap: wrap !important;
  justify-content: center !important;
  margin: 28px 0 0 !important;
  padding: 0 8px !important;
}
.gradio-container .er-chip-btn { min-width: 0 !important; flex: 0 0 auto !important; }
.gradio-container .er-chip-btn button {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-muted) !important;
  padding: 9px 14px !important;
  font-size: 13px !important;
  font-weight: 400 !important;
  border-radius: 10px !important;
  transition: border-color 180ms ease, background 180ms ease, color 180ms ease, transform 180ms ease;
  text-align: left !important;
  min-width: 0 !important;
  box-shadow: none !important;
}
.gradio-container .er-chip-btn button:hover {
  border-color: var(--accent-line) !important;
  background: var(--surface-2) !important;
  color: var(--text) !important;
  transform: translateY(-1px);
}

/* CHAT */
.gradio-container .er-chat {
  background: transparent !important;
  border: none !important;
  margin-top: 8px;
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
  padding: 16px 0 !important;
  color: var(--text) !important;
  max-width: 100% !important;
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
  font-size: 15px !important;
  line-height: 1.6 !important;
}
.gradio-container .er-chat .message.bot,
.gradio-container .er-chat .bot {
  padding-left: 0 !important;
  background: transparent !important;
  border: none !important;
  max-width: 100% !important;
}
.gradio-container .er-chat .message p { margin: 0 0 10px !important; }
.gradio-container .er-chat .message p:last-child { margin: 0 !important; }
.gradio-container .er-chat .avatar-container { display: none !important; }

/* COMPOSER */
.er-composer-wrap {
  position: sticky;
  bottom: 16px;
  z-index: 5;
  margin-top: 24px;
  background: var(--surface);
  border: 1px solid var(--border-mid);
  border-radius: 16px;
  padding: 0;
  transition: border-color 180ms ease, box-shadow 180ms ease;
  position: relative;
  box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}
.er-composer-wrap:focus-within {
  border-color: var(--accent-line);
  box-shadow: 0 0 0 3px var(--accent-soft), 0 8px 32px rgba(0,0,0,0.22);
}
.gradio-container .er-composer-wrap textarea {
  background: transparent !important;
  border: none !important;
  resize: none !important;
  color: var(--text) !important;
  font-size: 15px !important;
  line-height: 1.55 !important;
  padding: 16px 64px 16px 18px !important;
  min-height: 56px !important;
  outline: none !important;
  box-shadow: none !important;
  font-family: inherit !important;
  width: 100% !important;
}
.gradio-container .er-composer-wrap textarea::placeholder {
  color: var(--text-dim) !important;
}
.gradio-container .er-send-btn {
  position: absolute !important;
  right: 8px !important;
  bottom: 8px !important;
  min-width: 0 !important;
  z-index: 6;
}
.gradio-container .er-send-btn button {
  background: var(--accent) !important;
  color: #061a16 !important;
  border: none !important;
  width: 38px !important;
  height: 38px !important;
  min-width: 38px !important;
  border-radius: 10px !important;
  padding: 0 !important;
  font-size: 16px !important;
  font-weight: 600 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: filter 180ms ease, transform 120ms ease, box-shadow 180ms ease;
  box-shadow: 0 0 24px rgba(94,234,212,0.16);
}
.gradio-container .er-send-btn button:hover {
  filter: brightness(1.06);
  box-shadow: 0 0 32px rgba(94,234,212,0.28);
}
.gradio-container .er-send-btn button:active {
  transform: scale(0.96);
}

/* RESET */
.er-toolrow {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  margin-top: 14px !important;
  gap: 12px !important;
}
.er-footnote {
  color: var(--text-dim);
  font-size: 11.5px;
  letter-spacing: 0.01em;
}
.gradio-container .er-reset-btn { min-width: 0 !important; flex: 0 0 auto !important; }
.gradio-container .er-reset-btn button {
  background: transparent !important;
  border: none !important;
  color: var(--text-dim) !important;
  font-size: 12px !important;
  font-weight: 400 !important;
  padding: 6px 10px !important;
  min-width: 0 !important;
  transition: color 180ms ease;
  box-shadow: none !important;
}
.gradio-container .er-reset-btn button:hover { color: var(--text-muted) !important; }

/* INSPECT DRAWER */
.er-inspect {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 22px 22px 18px !important;
  margin-top: 28px !important;
  animation: er-fade-in 220ms ease both;
}
@keyframes er-fade-in {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
.er-inspect-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 18px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--border);
}
.er-inspect-title { font-size: 13.5px; font-weight: 600; color: var(--text); letter-spacing: 0.01em; }
.er-inspect-sub { font-size: 11.5px; color: var(--text-dim); }

/* TABS inside drawer */
.gradio-container .er-tabs > div[role="tablist"],
.gradio-container .er-tabs .tab-nav {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--border) !important;
  margin-bottom: 18px !important;
  padding: 0 !important;
}
.gradio-container .er-tabs button {
  background: transparent !important;
  border: none !important;
  color: var(--text-muted) !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  padding: 10px 0 !important;
  margin-right: 24px !important;
  border-bottom: 1.5px solid transparent !important;
  border-radius: 0 !important;
  transition: color 180ms ease, border-color 180ms ease;
  min-width: 0 !important;
  box-shadow: none !important;
}
.gradio-container .er-tabs button.selected,
.gradio-container .er-tabs button[aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}

/* INSPECT CARDS */
.er-card { padding: 0; margin-bottom: 18px; }
.er-card:last-child { margin-bottom: 0; }
.er-mini-title {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 12px;
}
.er-empty {
  color: var(--text-dim);
  font-size: 13px;
  padding: 12px 14px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}

/* PLAN ROWS */
.er-plan-rows { display: flex; flex-direction: column; gap: 8px; }
.er-plan-row {
  display: flex; justify-content: space-between; align-items: flex-start;
  gap: 14px;
  padding: 12px 14px;
  background: var(--surface-2);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}
.er-plan-row .k {
  color: var(--text-muted); font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.06em;
  font-weight: 500;
  flex: 0 0 auto;
  padding-top: 1px;
}
.er-plan-row .v {
  color: var(--text); font-size: 13.5px; font-weight: 500;
  text-align: right; line-height: 1.5;
}
.er-plan-row.accent { border-color: var(--accent-line); background: var(--accent-soft); }
.er-plan-row.accent .v { color: var(--accent); }

/* SOURCE CARDS */
.er-sources { display: flex; flex-direction: column; gap: 8px; }
.er-source {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 13px 14px;
  transition: border-color 180ms ease, background 180ms ease;
}
.er-source:hover { border-color: var(--accent-line); background: var(--surface-3); }
.er-source-title { font-size: 13.5px; font-weight: 500; color: var(--text); margin-bottom: 4px; line-height: 1.4; }
.er-source-name { font-size: 11.5px; color: var(--text-muted); margin-bottom: 8px; }
.er-source-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
.er-tag {
  display: inline-block;
  font-size: 10.5px;
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.04);
  color: var(--text-muted);
  border: 1px solid var(--border);
  letter-spacing: 0.02em;
}
.er-tag.crisis { background: rgba(248,113,113,0.10); color: var(--danger); border-color: rgba(248,113,113,0.22); }
.er-tag.accent { background: var(--accent-soft); color: var(--accent); border-color: var(--accent-line); }
.er-source-why { font-size: 11.5px; color: var(--text-dim); line-height: 1.55; }
.er-source a {
  color: var(--accent); font-size: 12px; text-decoration: none;
  border-bottom: 1px solid var(--accent-line);
  transition: border-color 180ms ease;
}
.er-source a:hover { border-bottom-color: var(--accent); }

/* DIAGNOSTICS GRID */
.er-diag-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.er-diag {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 11px 12px;
  min-width: 0;
}
.er-diag .k {
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-dim);
  margin-bottom: 5px;
  font-weight: 500;
}
.er-diag .v {
  font-size: 13px; color: var(--text); font-weight: 500; line-height: 1.4;
  word-break: break-word;
}
.er-diag.warn { border-color: rgba(245,182,105,0.22); }
.er-diag.warn .v { color: var(--warm); }
.er-diag.danger { border-color: rgba(248,113,113,0.22); }
.er-diag.danger .v { color: var(--danger); }

/* TIMELINE */
.er-timeline-row { display: flex; flex-wrap: wrap; gap: 6px; }
.er-time-pill {
  font-size: 10.5px; padding: 4px 10px; border-radius: 999px;
  background: var(--surface-2); color: var(--text-muted);
  border: 1px solid var(--border);
  letter-spacing: 0.02em;
}

/* METER */
.er-meter {
  height: 4px; border-radius: 999px; overflow: hidden;
  background: rgba(255,255,255,0.06);
  margin-top: 10px;
}
.er-meter > div { height: 100%; background: var(--accent); transition: width 320ms ease; }

/* IG TOKENS */
.er-ig-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.er-ig {
  font-size: 10.5px; padding: 3px 9px; border-radius: 999px;
  background: rgba(248,113,113,0.10);
  color: var(--danger);
  border: 1px solid rgba(248,113,113,0.22);
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
.gradio-container ::-webkit-scrollbar-thumb:hover {
  background: rgba(255,255,255,0.12);
}
.gradio-container ::-webkit-scrollbar-track { background: transparent; }

/* RESPONSIVE */
/* TYPING INDICATOR (3 dots) */
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

/* HERO GRADIENT TEXT */
.er-hero h1 {
  background: linear-gradient(180deg, #f3f7fc 0%, #b6c2d2 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  color: transparent;
}

/* MESSAGE ENTRANCE FADE */
.gradio-container .er-chat .message {
  animation: er-msg-in 260ms cubic-bezier(0.22, 0.61, 0.36, 1) both;
}
@keyframes er-msg-in {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* DRAWER SLIDE-IN (replace plain fade) */
.er-inspect {
  animation: er-slide-in 320ms cubic-bezier(0.22, 0.61, 0.36, 1) both !important;
}
@keyframes er-slide-in {
  from { opacity: 0; transform: translateY(14px) scale(0.995); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* BRAND-DOT RIPPLE */
.er-brand-dot {
  position: relative;
}
.er-brand-dot::after {
  content: "";
  position: absolute;
  inset: -3px;
  border-radius: 50%;
  border: 1px solid var(--accent);
  opacity: 0;
  animation: er-ripple 2.6s ease-out infinite;
}
@keyframes er-ripple {
  0%   { transform: scale(0.85); opacity: 0.55; }
  100% { transform: scale(2.4);  opacity: 0; }
}

/* USER PILL DEPTH */
.gradio-container .er-chat .message.user,
.gradio-container .er-chat .user {
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 1px 2px rgba(0,0,0,0.10) !important;
  border-radius: 16px 16px 4px 16px !important;
}

/* SEND BUTTON GRADIENT + KICK */
.gradio-container .er-send-btn button {
  background: linear-gradient(135deg, #5eead4 0%, #7ff0d9 100%) !important;
}
.gradio-container .er-send-btn button:active {
  transform: scale(0.94);
  filter: brightness(0.96);
}

/* SOURCE-TYPE GLYPH */
.er-source-title::before {
  content: "◇";
  color: var(--accent);
  margin-right: 8px;
  font-size: 11px;
  opacity: 0.7;
}
.er-source[data-kind="crisis"] .er-source-title::before { content: "✦"; color: var(--danger); opacity: 0.8; }
.er-source[data-kind="university"] .er-source-title::before { content: "◆"; }

/* CHIP HOVER LIFT (already partial; smooth) */
.gradio-container .er-chip-btn button {
  transition: border-color 200ms ease, background 200ms ease, color 200ms ease, transform 200ms ease, box-shadow 200ms ease !important;
}
.gradio-container .er-chip-btn button:hover {
  box-shadow: 0 6px 20px rgba(94,234,212,0.10) !important;
}

/* COMPOSER FOCUS GLOW (animated border) */
.er-composer-wrap {
  transition: border-color 240ms ease, box-shadow 240ms ease !important;
}

/* HERO PARAGRAPH SUBTLE EMPHASIS */
.er-hero p {
  font-style: normal;
}
.er-hero p::first-letter {
  color: var(--text);
}
.er-hero-meta {
  margin-top: 22px;
  color: var(--text-dim);
  font-size: 11.5px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

/* International concern soft tag (used in body if needed) */
.er-intl-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  padding: 3px 9px;
  border-radius: 999px;
  background: rgba(245,182,105,0.10);
  color: var(--warm);
  border: 1px solid rgba(245,182,105,0.22);
  letter-spacing: 0.02em;
}

/* Tighten chip text wrap on smaller widths */
.gradio-container .er-chip-btn button { white-space: nowrap; }

@media (max-width: 700px) {
  .gradio-container { padding: 0 16px 32px !important; }
  .er-topbar { flex-wrap: wrap !important; gap: 10px !important; }
  .er-topbar > .er-mode-wrap { order: 3; flex-basis: 100% !important; justify-content: center; }
  .er-hero { padding: 56px 8px 18px; }
  .er-hero h1 { font-size: 24px; }
  .er-diag-grid { grid-template-columns: 1fr; }
  .gradio-container .er-chat .message.user { max-width: 88% !important; }
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
    """Support card — what kind of support, what's next, which resources."""
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
            if url:
                html += f"<div style='margin-top:8px;'><a href='{url}' target='_blank' rel='noopener'>Open ↗</a></div>"
            html += "</div>"
        html += "</div>"
    else:
        html += "<div class='er-mini-title' style='margin-top:18px;'>Resources</div>"
        html += "<div class='er-empty'>No external resource needed for this turn.</div>"
    html += "</div>"
    return html


def format_retrieval_panel(result=None) -> str:
    """Diagnostics — pipeline internals for class & eval review."""
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


TYPING_HTML = "<span class='er-typing'><span></span><span></span><span></span></span>"
STREAM_ENABLED = os.getenv("EMPATHRAG_STREAM", "1") != "0"
STREAM_WORDS_PER_CHUNK = int(os.getenv("EMPATHRAG_STREAM_WORDS", "2"))
STREAM_CHUNK_DELAY_MS = int(os.getenv("EMPATHRAG_STREAM_DELAY_MS", "75"))
TYPING_DELAY_MS = int(os.getenv("EMPATHRAG_TYPING_DELAY_MS", "650"))


def _stream_chunks(full_text: str):
    """Yield growing partial strings to simulate streaming."""
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


def respond(message, chat_history, session_state, audience_mode):
    if not session_state:
        session_state = new_session_state()

    emotion_history = session_state["emotion_history"]
    session_id = session_state["session_id"]

    if not message.strip():
        yield (
            chat_history,
            format_decision_trace(),
            format_emotion_timeline(emotion_history, "stable"),
            "stable",
            format_ig_panel(False, 0.0, [], False),
            format_retrieval_panel(),
            session_id,
            session_state,
        )
        return

    # Show user message + typing indicator immediately for liveness.
    chat_history = list(chat_history) + [(message, TYPING_HTML)]
    yield (
        chat_history,
        format_decision_trace(),
        format_emotion_timeline(emotion_history, "stable"),
        "stable",
        format_ig_panel(False, 0.0, [], False),
        format_retrieval_panel(),
        session_id,
        session_state,
    )

    if STREAM_ENABLED and TYPING_DELAY_MS > 0:
        import time as _t
        _t.sleep(TYPING_DELAY_MS / 1000.0)

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
    timeline_html = format_emotion_timeline(emotion_history, result["trajectory"])
    decision_html = format_decision_trace(result)
    retrieval_html = format_retrieval_panel(result)

    # Stream the response into the last chat slot.
    is_crisis = bool(result.get("crisis"))
    safety_explanation = result.get("safety_explanation", {}) or {}
    ig_tokens = safety_explanation.get("ig_tokens") or []
    explanation_reason = safety_explanation.get("reason", "")
    explanation_available = bool(safety_explanation.get("available"))
    ig_panel_html = format_ig_panel(
        is_crisis,
        result.get("crisis_confidence", 0.0),
        ig_tokens,
        loading=is_crisis and hasattr(get_pipeline(), "guardrail") and not explanation_available,
        explanation_reason=explanation_reason,
    )

    for partial in _stream_chunks(full_response):
        chat_history[-1] = (message, partial)
        yield (
            chat_history,
            decision_html,
            timeline_html,
            result["trajectory"],
            ig_panel_html,
            retrieval_html,
            session_id,
            session_state,
        )

    # Crisis: optionally compute IG attributions after the message lands.
    if is_crisis and hasattr(get_pipeline(), "guardrail") and not explanation_available:
        with pipeline_lock:
            active_pipeline = get_pipeline()
            if hasattr(active_pipeline, "guardrail"):
                _, confidence, ig_tokens = active_pipeline.guardrail.check(
                    message, threshold=0.5, skip_ig=False
                )
            else:
                confidence, ig_tokens = result["crisis_confidence"], []
        yield (
            chat_history,
            decision_html,
            timeline_html,
            result["trajectory"],
            format_ig_panel(True, confidence, ig_tokens, loading=False),
            retrieval_html,
            session_id,
            session_state,
        )


def reset_session_handler():
    session_state = new_session_state()
    return (
        [],
        format_decision_trace(),
        format_emotion_timeline([], "stable"),
        "stable",
        format_ig_panel(False, 0.0, [], False),
        format_retrieval_panel(),
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


with gr.Blocks(theme=theme, title="EmpathRAG", css=APP_CSS) as demo:
    initial_state = new_session_state()
    session_state = gr.State(value=initial_state)
    inspect_open = gr.State(value=False)

    # ---- Top bar ----
    with gr.Row(elem_classes=["er-topbar"]):
        gr.HTML(
            """
            <div class="er-brand">
              <span class="er-brand-dot"></span>
              EmpathRAG
              <span class="er-brand-meta">· support navigator</span>
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
        inspect_btn = gr.Button("Inspect", elem_classes=["er-inspect-btn"])

    # ---- Hero (empty state) ----
    hero_block = gr.HTML(
        """
        <div class="er-hero">
          <h1>How are you doing today?</h1>
          <p>I'm here to listen first — about academic stress, mental health, advisor pressure, or anything weighing on you. When you're ready, I can also help you find specific UMD resources, including ones for international and F-1 students.</p>
          <div class="er-hero-meta">Conversations are not logged. Not therapy or emergency care.</div>
        </div>
        """,
        visible=True,
    )

    with gr.Row(elem_classes=["er-chips"], visible=True) as chip_row:
        chip_counseling = gr.Button("I'm thinking about counseling", elem_classes=["er-chip-btn"])
        chip_ads = gr.Button("ADS accommodations", elem_classes=["er-chip-btn"])
        chip_advisor = gr.Button("Advisor conflict", elem_classes=["er-chip-btn"])
        chip_intl = gr.Button("F-1 visa & academic worry", elem_classes=["er-chip-btn"])
        chip_grounding = gr.Button("Pre-exam grounding", elem_classes=["er-chip-btn"])

    # ---- Chat ----
    chatbot = gr.Chatbot(
        elem_classes=["er-chat"],
        show_label=False,
        height=520,
        bubble_full_width=False,
        avatar_images=None,
        show_share_button=False,
        show_copy_button=True,
        sanitize_html=False,
    )

    # ---- Composer ----
    with gr.Group(elem_classes=["er-composer-wrap"]):
        msg_box = gr.Textbox(
            placeholder="Tell me what's on your mind…",
            show_label=False,
            container=False,
            lines=1,
            max_lines=8,
            autofocus=True,
        )
        send_btn = gr.Button("→", elem_classes=["er-send-btn"], variant="primary")

    with gr.Row(elem_classes=["er-toolrow"]):
        gr.HTML(
            "<div class='er-footnote'>Conversations are not logged by default. If you are in immediate danger, call or text 988.</div>"
        )
        reset_btn = gr.Button("Clear conversation", elem_classes=["er-reset-btn"])

    # Hidden state surfaces (kept to preserve respond() output contract)
    session_id_box = gr.Textbox(value=initial_state["session_id"], visible=False)
    trajectory_out = gr.Textbox(value="stable", visible=False)

    # ---- Inspect drawer ----
    with gr.Column(visible=False, elem_classes=["er-inspect"]) as inspect_drawer:
        gr.HTML(
            "<div class='er-inspect-head'>"
            "<div class='er-inspect-title'>Behind the answer</div>"
            "<div class='er-inspect-sub'>Pipeline view · class & eval</div>"
            "</div>"
        )
        with gr.Tabs(elem_classes=["er-tabs"]):
            with gr.Tab("Support card"):
                decision_out = gr.HTML(value=format_decision_trace())
            with gr.Tab("Diagnostics"):
                retrieval_out = gr.HTML(value=format_retrieval_panel())
                timeline_out = gr.HTML(value=format_emotion_timeline([], "stable"))
                crisis_out = gr.HTML(value=format_ig_panel(False, 0.0, [], False))

    # ---- Wiring ----
    submit_outputs = [
        chatbot,
        decision_out,
        timeline_out,
        trajectory_out,
        crisis_out,
        retrieval_out,
        session_id_box,
        session_state,
        hero_block,
        chip_row,
    ]

    def respond_with_chrome(message, chat_history, session_state, audience_mode):
        hide = bool(message and message.strip())
        chrome = (gr.update(visible=not hide), gr.update(visible=not hide))
        for tup in respond(message, chat_history, session_state, audience_mode):
            yield tup + chrome

    msg_box.submit(
        respond_with_chrome,
        inputs=[msg_box, chatbot, session_state, audience_mode_box],
        outputs=submit_outputs,
    ).then(lambda: "", outputs=msg_box)

    send_btn.click(
        respond_with_chrome,
        inputs=[msg_box, chatbot, session_state, audience_mode_box],
        outputs=submit_outputs,
    ).then(lambda: "", outputs=msg_box)

    def reset_with_chrome():
        base = reset_session_handler()
        return base + (gr.update(visible=True), gr.update(visible=True))

    reset_btn.click(reset_with_chrome, outputs=submit_outputs)

    def toggle_inspect(open_state):
        new_state = not open_state
        return new_state, gr.update(visible=new_state)

    inspect_btn.click(
        toggle_inspect,
        inputs=[inspect_open],
        outputs=[inspect_open, inspect_drawer],
    )

    chip_counseling.click(
        lambda: set_prompt("I think I need counseling at UMD, but I do not know how to start."),
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
    demo.launch(share=SHARE_DEMO)
