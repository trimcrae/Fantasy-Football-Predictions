#!/usr/bin/env python3
"""Launcher that works from any directory: `python eliminator/run.py plan --pool ...`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eliminator.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
