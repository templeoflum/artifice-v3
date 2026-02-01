"""GPU texture wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import moderngl


class Texture:
    """Wrapper for ModernGL textures with convenience methods."""

    def __init__(self, tex: moderngl.Texture):
        """Wrap a ModernGL texture.

        Args:
            tex: ModernGL texture object
        """
        self._tex = tex

    @property
    def width(self) -> int:
        """Texture width in pixels."""
        return self._tex.width

    @property
    def height(self) -> int:
        """Texture height in pixels."""
        return self._tex.height

    @property
    def size(self) -> tuple[int, int]:
        """Texture size as (width, height)."""
        return (self._tex.width, self._tex.height)

    @property
    def mgl(self) -> moderngl.Texture:
        """Get underlying ModernGL texture."""
        return self._tex

    def bind_image(self, unit: int, read: bool = True, write: bool = True) -> None:
        """Bind texture as image for compute shader access.

        Args:
            unit: Image unit (0, 1, 2, ...)
            read: Allow reading
            write: Allow writing
        """
        access = 0
        if read and write:
            access = 2  # READ_WRITE
        elif read:
            access = 0  # READ_ONLY
        elif write:
            access = 1  # WRITE_ONLY

        self._tex.bind_to_image(unit, read=read, write=write)

    def upload(self, data: np.ndarray) -> None:
        """Upload numpy array to texture.

        Args:
            data: HxWx4 float32 array (RGBA)
        """
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Ensure RGBA
        if data.ndim == 2:
            # Grayscale -> RGBA
            data = np.stack([data, data, data, np.ones_like(data)], axis=-1)
        elif data.shape[-1] == 3:
            # RGB -> RGBA
            alpha = np.ones((*data.shape[:-1], 1), dtype=np.float32)
            data = np.concatenate([data, alpha], axis=-1)

        # Flip Y for OpenGL
        data = np.ascontiguousarray(np.flipud(data))
        self._tex.write(data.tobytes())

    def download(self) -> np.ndarray:
        """Download texture to numpy array.

        Returns:
            HxWx4 float32 array (RGBA)
        """
        data = np.frombuffer(self._tex.read(), dtype=np.float32)
        data = data.reshape((self._tex.height, self._tex.width, 4))
        # Flip Y back from OpenGL
        return np.flipud(data).copy()

    def clear(self, r: float = 0, g: float = 0, b: float = 0, a: float = 1) -> None:
        """Clear texture to a solid color.

        Args:
            r, g, b, a: Clear color components (0-1)
        """
        data = np.full((self._tex.height, self._tex.width, 4),
                       [r, g, b, a], dtype=np.float32)
        self.upload(data)

    def release(self) -> None:
        """Release GPU resources."""
        self._tex.release()
