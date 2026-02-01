"""Artifice v3 - Real-time GPU glitch art synthesizer.

Built on GLIC principles: prediction, residuals, quantization.
GPU-first, always real-time, no CPU processing paths.
"""

__version__ = "3.0.0"


def main():
    """Launch Artifice."""
    from artifice.ui.window import run
    run()
