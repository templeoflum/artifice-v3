"""Artifice GPU nodes."""

from artifice.nodes.testcard import TestCardNode
from artifice.nodes.image_io import ImageLoaderNode, ImageSaverNode
from artifice.nodes.glic import (
    ColorSpaceNode,
    PredictNode,
    ResidualNode,
    ReconstructNode,
    QuantizeNode,
    GLICPipelineNode,
)

__all__ = [
    # Generators
    "TestCardNode",
    # I/O
    "ImageLoaderNode",
    "ImageSaverNode",
    # GLIC
    "ColorSpaceNode",
    "PredictNode",
    "ResidualNode",
    "ReconstructNode",
    "QuantizeNode",
    "GLICPipelineNode",
]
