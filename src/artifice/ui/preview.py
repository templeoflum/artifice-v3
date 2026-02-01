"""OpenGL preview widget - displays GPU textures directly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from OpenGL import GL
from PySide6.QtCore import Signal, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

if TYPE_CHECKING:
    from artifice.engine.texture import Texture


class PreviewWidget(QOpenGLWidget):
    """GPU-accelerated preview widget.

    Displays textures directly from GPU memory without CPU roundtrips.
    """

    context_ready = Signal(object)  # Emits ModernGL context
    frame_rendered = Signal()

    def __init__(self, parent=None):
        # Set up OpenGL format
        fmt = QSurfaceFormat()
        fmt.setVersion(4, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        fmt.setSwapInterval(1)  # VSync
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)

        self._texture: Texture | None = None
        self._display_texture_id: int | None = None
        self._initialized = False

        # Shader for display
        self._program: int | None = None
        self._vao: int | None = None

    def set_texture(self, texture: Texture | None) -> None:
        """Set the texture to display."""
        self._texture = texture
        self.update()

    def initializeGL(self) -> None:
        """Initialize OpenGL resources."""
        # Simple fullscreen quad shader
        vertex_src = """
        #version 330 core
        out vec2 uv;
        void main() {
            vec2 positions[4] = vec2[](
                vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1)
            );
            vec2 uvs[4] = vec2[](
                vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 1)
            );
            gl_Position = vec4(positions[gl_VertexID], 0, 1);
            uv = uvs[gl_VertexID];
        }
        """

        fragment_src = """
        #version 330 core
        in vec2 uv;
        out vec4 fragColor;
        uniform sampler2D tex;
        void main() {
            fragColor = texture(tex, uv);
        }
        """

        # Compile shaders
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, vertex_src)
        GL.glCompileShader(vs)

        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, fragment_src)
        GL.glCompileShader(fs)

        self._program = GL.glCreateProgram()
        GL.glAttachShader(self._program, vs)
        GL.glAttachShader(self._program, fs)
        GL.glLinkProgram(self._program)

        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

        # Create VAO
        self._vao = GL.glGenVertexArrays(1)

        self._initialized = True

        # Emit context ready with ModernGL wrapper
        import moderngl
        ctx = moderngl.create_context(require=430)
        self.context_ready.emit(ctx)

    def paintGL(self) -> None:
        """Render the texture to screen."""
        if not self._initialized:
            return

        # Get framebuffer size (high-DPI aware)
        dpr = self.devicePixelRatio()
        fb_w = int(self.width() * dpr)
        fb_h = int(self.height() * dpr)

        GL.glViewport(0, 0, fb_w, fb_h)
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        if self._texture is None:
            return

        # Bind texture and draw quad
        GL.glUseProgram(self._program)
        GL.glBindVertexArray(self._vao)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture.mgl.glo)

        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        self.frame_rendered.emit()

    def resizeGL(self, w: int, h: int) -> None:
        """Handle resize."""
        dpr = self.devicePixelRatio()
        GL.glViewport(0, 0, int(w * dpr), int(h * dpr))
