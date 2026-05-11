"""Root-level entry point for Hugging Face Spaces.

HF Spaces conventionally looks for `app.py` at the repository root. The
actual demo lives at `demo/app.py` and assumes the working directory is the
repo root. This shim adjusts sys.path, sets sensible Space defaults, and
runs the demo module so the Space picks up everything without manual
configuration.

For local development run `demo/app.py` directly with the documented
environment variables (see README Quickstart).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT))

# Defaults that make the Space friendlier to anonymous graders. Streaming
# on by default; LLM rephraser on when the secret is present.
os.environ.setdefault("EMPATHRAG_DEMO_BACKEND", "fast")
os.environ.setdefault("EMPATHRAG_RETRIEVAL_CORPUS", "curated_support")
os.environ.setdefault("EMPATHRAG_TOP_K", "5")
if os.environ.get("GROQ_API_KEY"):
    os.environ.setdefault("EMPATHRAG_REPHRASER_ENABLED", "1")

if __name__ == "__main__":
    # Execute demo/app.py as if it were the main script. Its existing
    # if __name__ == "__main__" block calls demo.launch() with the right
    # config (provider availability print, share off, etc.).
    runpy.run_path(str(ROOT / "demo" / "app.py"), run_name="__main__")
