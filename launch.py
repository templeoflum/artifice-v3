#!/usr/bin/env python
"""Launch Artifice v3."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from artifice.ui.window import run

if __name__ == "__main__":
    run()
