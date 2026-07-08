#!/usr/bin/env python3
"""Build Phase 1 data artifacts from TDX staging files."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.data.build_panel import build


if __name__ == "__main__":
    build()
