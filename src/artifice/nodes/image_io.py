"""Image I/O nodes - load and save images."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from PIL import Image

from artifice.engine.node import GPUNode, Parameter
from artifice.engine.texture import Texture

if TYPE_CHECKING:
    from artifice.engine.context import GPUContext


class ImageLoaderNode(GPUNode):
    """Load an image from disk to GPU texture.

    Supports: PNG, JPG, JPEG, TIFF, BMP, WebP, GIF
    Images are resized to match the pipeline's output size.
    """

    name: ClassVar[str] = "Image Loader"
    category: ClassVar[str] = "I/O"
    shader_name: ClassVar[str] = ""  # No shader - direct upload

    _loaded_path: str = ""
    _image_data: np.ndarray | None = None
    _target_size: tuple[int, int] = (512, 512)

    def __init__(self):
        super().__init__()
        self._loaded_path = ""
        self._image_data = None
        self._target_size = (512, 512)

    @classmethod
    def parameters(cls) -> list[Parameter]:
        return [
            Parameter(
                name="path",
                type="filepath",
                default="",
                description="Path to image file",
            ),
        ]

    @classmethod
    def input_slots(cls) -> list[str]:
        return []  # No inputs - source node

    def compile(self, ctx: GPUContext) -> None:
        """No shader to compile."""
        pass

    def allocate(self, ctx: GPUContext, width: int, height: int) -> None:
        """Allocate output texture at pipeline size."""
        from artifice.engine.texture import Texture

        self._target_size = (width, height)

        # Allocate output at pipeline size
        existing = self._outputs.get("output")
        if existing is None or existing.size != (width, height):
            if existing:
                existing.release()
            tex = ctx.create_texture(width, height)
            self._outputs["output"] = Texture(tex)

        # Force reload to resize to new dimensions
        self._loaded_path = ""

    def _load_image(self, target_size: tuple[int, int]) -> np.ndarray | None:
        """Load image from disk and resize to target size."""
        path = self.get_param("path")
        if not path or not Path(path).exists():
            return None

        try:
            img = Image.open(path)
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # Resize to match pipeline size
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                print(f"[ImageLoader] Resized to {target_size[0]}x{target_size[1]}")

            data = np.array(img, dtype=np.float32) / 255.0
            return data
        except Exception as e:
            print(f"[ImageLoader] Failed to load {path}: {e}")
            return None

    def execute(self, ctx: GPUContext) -> None:
        """Load image and upload to GPU."""
        path = self.get_param("path")

        # Reload if path changed
        if path != self._loaded_path:
            self._image_data = self._load_image(self._target_size)
            self._loaded_path = path

        # Upload image data
        output = self._outputs.get("output")
        if output and self._image_data is not None:
            output.upload(self._image_data)
        elif output is not None and self._image_data is None:
            # No image loaded - fill with magenta placeholder
            w, h = self._target_size
            placeholder = np.full((h, w, 4), [1.0, 0.0, 1.0, 1.0], dtype=np.float32)
            output.upload(placeholder)


class ImageSaverNode(GPUNode):
    """Save GPU texture to disk as image.

    Supports: PNG, JPG, TIFF, BMP, WebP
    """

    name: ClassVar[str] = "Image Saver"
    category: ClassVar[str] = "I/O"
    shader_name: ClassVar[str] = ""  # No shader - direct download

    @classmethod
    def parameters(cls) -> list[Parameter]:
        return [
            Parameter(
                name="path",
                type="filepath_save",
                default="",
                description="Path to save image",
            ),
            Parameter(
                name="auto_save",
                type="bool",
                default=False,
                description="Save every frame (use with caution!)",
            ),
        ]

    @classmethod
    def output_slots(cls) -> list[str]:
        return []  # No outputs - sink node

    def compile(self, ctx: GPUContext) -> None:
        """No shader to compile."""
        pass

    def allocate(self, ctx: GPUContext, width: int, height: int) -> None:
        """No outputs to allocate."""
        pass

    def execute(self, ctx: GPUContext) -> None:
        """Download and save if auto_save is enabled."""
        if not self.get_param("auto_save"):
            return  # Manual save only

        self.save()

    def save(self) -> bool:
        """Save current input to disk.

        Returns:
            True if saved successfully
        """
        path = self.get_param("path")
        if not path:
            return False

        input_tex = self._inputs.get("input")
        if not input_tex:
            return False

        try:
            # Download from GPU
            data = input_tex.download()

            # Convert to uint8
            data = (np.clip(data, 0, 1) * 255).astype(np.uint8)

            # Save with PIL
            img = Image.fromarray(data, mode="RGBA")

            # Convert to RGB for formats that don't support alpha
            path_lower = path.lower()
            if path_lower.endswith(('.jpg', '.jpeg')):
                img = img.convert("RGB")

            img.save(path)
            print(f"[ImageSaver] Saved to {path}")
            return True

        except Exception as e:
            print(f"[ImageSaver] Failed to save {path}: {e}")
            return False
