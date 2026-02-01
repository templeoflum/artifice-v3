"""Base GPU node class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

if TYPE_CHECKING:
    import moderngl
    from artifice.engine.context import GPUContext
    from artifice.engine.texture import Texture


@dataclass
class Parameter:
    """Node parameter definition."""
    name: str
    type: str  # "float", "int", "bool", "enum"
    default: Any
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    description: str = ""


class GPUNode(ABC):
    """Base class for GPU compute nodes.

    Each node has:
    - A compute shader that processes textures
    - Input/output texture slots
    - Parameters that map to shader uniforms
    """

    # Class attributes - override in subclasses
    name: ClassVar[str] = "GPUNode"
    category: ClassVar[str] = "Base"
    shader_name: ClassVar[str] = ""  # Path relative to shaders/

    def __init__(self):
        self.id = str(uuid4())
        self.x = 0  # Position in graph editor
        self.y = 0

        # Runtime state
        self._shader: moderngl.ComputeShader | None = None
        self._params: dict[str, Any] = {}
        self._inputs: dict[str, Texture | None] = {}
        self._outputs: dict[str, Texture | None] = {}

        # Initialize parameters with defaults
        for param in self.parameters():
            self._params[param.name] = param.default

    @classmethod
    @abstractmethod
    def parameters(cls) -> list[Parameter]:
        """Define node parameters.

        Returns:
            List of Parameter definitions
        """
        return []

    @classmethod
    def input_slots(cls) -> list[str]:
        """Define input texture slots.

        Returns:
            List of input names (default: ["input"])
        """
        return ["input"]

    @classmethod
    def output_slots(cls) -> list[str]:
        """Define output texture slots.

        Returns:
            List of output names (default: ["output"])
        """
        return ["output"]

    def get_param(self, name: str) -> Any:
        """Get parameter value."""
        return self._params.get(name)

    def set_param(self, name: str, value: Any) -> None:
        """Set parameter value."""
        self._params[name] = value

    def set_input(self, name: str, texture: Texture | None) -> None:
        """Set input texture."""
        self._inputs[name] = texture

    def get_output(self, name: str) -> Texture | None:
        """Get output texture."""
        return self._outputs.get(name)

    def compile(self, ctx: GPUContext) -> None:
        """Compile the node's shader.

        Args:
            ctx: GPU context
        """
        if self.shader_name:
            self._shader = ctx.load_shader(self.shader_name)

    def allocate(self, ctx: GPUContext, width: int, height: int) -> None:
        """Allocate output textures.

        Args:
            ctx: GPU context
            width: Output width
            height: Output height
        """
        from artifice.engine.texture import Texture

        for slot in self.output_slots():
            existing = self._outputs.get(slot)
            if existing is None or existing.size != (width, height):
                if existing:
                    existing.release()
                tex = ctx.create_texture(width, height)
                self._outputs[slot] = Texture(tex)

    @abstractmethod
    def execute(self, ctx: GPUContext) -> None:
        """Execute the node's computation.

        Args:
            ctx: GPU context
        """
        pass

    def release(self) -> None:
        """Release GPU resources."""
        for tex in self._outputs.values():
            if tex:
                tex.release()
        self._outputs.clear()
        self._shader = None
