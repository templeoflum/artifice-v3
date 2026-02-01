"""GPU context management using ModernGL."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import moderngl

if TYPE_CHECKING:
    pass


class GPUContext:
    """OpenGL 4.3+ compute shader context.

    Manages the ModernGL context and shader compilation.
    Can operate standalone (headless) or share with Qt's OpenGL context.
    """

    SHADER_DIR = Path(__file__).parent.parent / "shaders"

    def __init__(self, standalone: bool = True):
        """Initialize GPU context.

        Args:
            standalone: If True, create standalone context. If False,
                       call set_context() with Qt's context later.
        """
        self._ctx: moderngl.Context | None = None
        self._standalone = standalone
        self._shader_cache: dict[str, moderngl.ComputeShader] = {}

        if standalone:
            self._ctx = moderngl.create_standalone_context(require=430)

    def set_context(self, ctx: moderngl.Context) -> None:
        """Set external ModernGL context (from Qt)."""
        self._ctx = ctx
        self._shader_cache.clear()

    @property
    def ctx(self) -> moderngl.Context:
        """Get the ModernGL context."""
        if self._ctx is None:
            raise RuntimeError("GPU context not initialized")
        return self._ctx

    def load_shader(self, name: str) -> moderngl.ComputeShader:
        """Load and compile a compute shader.

        Args:
            name: Shader name (relative path without .glsl extension)

        Returns:
            Compiled compute shader
        """
        if name in self._shader_cache:
            return self._shader_cache[name]

        shader_path = self.SHADER_DIR / f"{name}.glsl"
        if not shader_path.exists():
            raise FileNotFoundError(f"Shader not found: {shader_path}")

        source = shader_path.read_text()
        shader = self.ctx.compute_shader(source)
        self._shader_cache[name] = shader

        return shader

    def create_texture(self, width: int, height: int) -> moderngl.Texture:
        """Create an RGBA32F texture for compute shader I/O.

        Args:
            width: Texture width
            height: Texture height

        Returns:
            ModernGL texture
        """
        return self.ctx.texture((width, height), 4, dtype='f4')

    def memory_barrier(self) -> None:
        """Insert memory barrier for shader image access."""
        self.ctx.memory_barrier(barriers=moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def release(self) -> None:
        """Release GPU resources."""
        self._shader_cache.clear()
        if self._standalone and self._ctx:
            self._ctx.release()
            self._ctx = None
