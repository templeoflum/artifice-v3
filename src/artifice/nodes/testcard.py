"""Test card generator node."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from artifice.engine.node import GPUNode, Parameter

if TYPE_CHECKING:
    from artifice.engine.context import GPUContext


class TestCardNode(GPUNode):
    """Procedural test card generator.

    Generates calibration patterns for testing effects:
    - Color bars (RGBCMYWK)
    - Gradients
    - Checkerboard patterns
    - Zone plate (frequency test)
    """

    name: ClassVar[str] = "Test Card"
    category: ClassVar[str] = "Generator"
    shader_name: ClassVar[str] = "generators/testcard"

    PATTERNS = [
        "Color Bars",
        "Gradient H",
        "Gradient V",
        "Checkerboard",
        "Zone Plate",
        "RGB Sweep",
    ]

    @classmethod
    def parameters(cls) -> list[Parameter]:
        return [
            Parameter(
                name="pattern",
                type="enum",
                default="Color Bars",
                choices=cls.PATTERNS,
                description="Test pattern type",
            ),
            Parameter(
                name="scale",
                type="float",
                default=1.0,
                min_val=0.1,
                max_val=10.0,
                step=0.1,
                description="Pattern scale",
            ),
        ]

    @classmethod
    def input_slots(cls) -> list[str]:
        return []  # Generator - no inputs

    def execute(self, ctx: GPUContext) -> None:
        """Generate the test pattern."""
        if not self._shader:
            return

        output = self._outputs.get("output")
        if not output:
            return

        # Bind output texture
        output.mgl.bind_to_image(0, read=False, write=True)

        # Set uniforms
        pattern_idx = self.PATTERNS.index(self.get_param("pattern"))
        if "pattern" in self._shader:
            self._shader["pattern"].value = pattern_idx
        if "scale" in self._shader:
            self._shader["scale"].value = self.get_param("scale")
        if "resolution" in self._shader:
            self._shader["resolution"].value = output.size

        # Dispatch compute shader
        gx = (output.width + 15) // 16
        gy = (output.height + 15) // 16
        self._shader.run(gx, gy, 1)
