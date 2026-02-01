"""Main application window with dark theme."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from artifice.engine.context import GPUContext
from artifice.engine.pipeline import Pipeline
from artifice.nodes import TestCardNode, ImageLoaderNode, ImageSaverNode, GLICPipelineNode
from artifice.ui.preview import PreviewWidget
from artifice.ui.glic_panel import GLICControlPanel


# Project root for debug outputs
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class MainWindow(QMainWindow):
    """Main Artifice window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Artifice v3 - GPU Glitch Synthesizer")

        # Set reasonable size and center on screen
        self.resize(800, 600)
        self._center_on_screen()

        # GPU setup (will connect to preview's context)
        self._ctx: GPUContext | None = None
        self._pipeline: Pipeline | None = None

        # State
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._fps = 0.0
        self._running = False

        # Debug state
        self._debug_frame_count = 0
        self._debug_auto_captured = False

        # GLIC state
        self._glic_enabled = True
        self._glic_node: GLICPipelineNode | None = None
        self._source_node = None  # TestCard or ImageLoader

        # UI setup
        self._setup_ui()
        self._setup_menus()

        # Command file watcher for remote control
        self._command_file = _PROJECT_ROOT / "commands.txt"
        self._command_timer = QTimer(self)
        self._command_timer.timeout.connect(self._check_commands)
        self._command_timer.start(500)  # Check every 500ms

    def _center_on_screen(self) -> None:
        """Center window on primary screen."""
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + (geo.height() - self.height()) // 2
            self.move(x, y)

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        # Status bar (create first so other widgets can reference it)
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Initializing GPU...")

        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet("padding: 0 10px; font-weight: bold;")
        self._statusbar.addPermanentWidget(self._fps_label)

        self._res_label = QLabel("512x512")
        self._res_label.setStyleSheet("padding: 0 10px;")
        self._statusbar.addPermanentWidget(self._res_label)

        # Central preview widget
        self._preview = PreviewWidget()
        self._preview.setMinimumSize(400, 400)
        self._preview.context_ready.connect(self._on_context_ready)
        self.setCentralWidget(self._preview)

        # GLIC Control Panel (dockable)
        self._glic_panel = GLICControlPanel()
        self._glic_dock = QDockWidget("GLIC Controls", self)
        self._glic_dock.setWidget(self._glic_panel)
        self._glic_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._glic_dock.setMinimumWidth(220)
        self.addDockWidget(Qt.RightDockWidgetArea, self._glic_dock)

        # Connect panel signals to status bar feedback
        self._glic_panel.colorspace_changed.connect(
            lambda v: self._statusbar.showMessage(f"Color space: {v}")
        )
        self._glic_panel.prediction_changed.connect(
            lambda v: self._statusbar.showMessage(f"Prediction: {v}")
        )
        self._glic_panel.quantization_changed.connect(
            lambda v: self._statusbar.showMessage(f"Quantization: {v:.0f}")
        )
        self._glic_panel.clamp_changed.connect(
            lambda v: self._statusbar.showMessage(f"Clamp: {v}")
        )

    def _setup_menus(self) -> None:
        """Set up menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_action = QAction("&Load Image...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_image)
        file_menu.addAction(load_action)

        save_action = QAction("&Save Image...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_image)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        for res in [(256, 256), (512, 512), (720, 720), (1024, 1024), (1920, 1080)]:
            action = QAction(f"{res[0]}x{res[1]}", self)
            action.triggered.connect(lambda checked, r=res: self._set_resolution(r))
            view_menu.addAction(action)

        # GLIC menu (simplified - main controls in dock panel)
        glic_menu = menubar.addMenu("&GLIC")

        # Toggle GLIC
        self._toggle_glic_action = QAction("&Enable GLIC", self)
        self._toggle_glic_action.setCheckable(True)
        self._toggle_glic_action.setChecked(True)
        self._toggle_glic_action.setShortcut("G")
        self._toggle_glic_action.triggered.connect(self._toggle_glic)
        glic_menu.addAction(self._toggle_glic_action)

        glic_menu.addSeparator()

        # Randomize
        randomize_action = QAction("&Randomize Settings", self)
        randomize_action.setShortcut("R")
        randomize_action.triggered.connect(self._glic_panel._randomize_settings)
        glic_menu.addAction(randomize_action)

        glic_menu.addSeparator()

        # Show/hide panel
        show_panel_action = self._glic_dock.toggleViewAction()
        show_panel_action.setText("Show &Control Panel")
        glic_menu.addAction(show_panel_action)

        # Generator menu
        gen_menu = menubar.addMenu("&Generator")
        test_action = QAction("&Test Card", self)
        test_action.setShortcut("T")
        test_action.triggered.connect(self._use_testcard)
        gen_menu.addAction(test_action)

        # Debug menu
        debug_menu = menubar.addMenu("&Debug")

        screenshot_action = QAction("Save &Screenshot", self)
        screenshot_action.setShortcut("F12")
        screenshot_action.triggered.connect(self._save_debug_screenshot)
        debug_menu.addAction(screenshot_action)

        dump_info_action = QAction("Dump &Info", self)
        dump_info_action.triggered.connect(self._dump_debug_info)
        debug_menu.addAction(dump_info_action)

    def _on_context_ready(self, mgl_ctx) -> None:
        """Initialize GPU pipeline when OpenGL context is ready."""
        self._ctx = GPUContext(standalone=False)
        self._ctx.set_context(mgl_ctx)

        self._pipeline = Pipeline(self._ctx)
        self._pipeline.set_output_size(512, 512)

        # Add default test card + GLIC
        self._source_node = TestCardNode()
        self._pipeline.add_node(self._source_node)

        self._glic_node = GLICPipelineNode()
        self._pipeline.add_node(self._glic_node)

        # Connect control panel to GLIC node
        self._glic_panel.set_glic_node(self._glic_node)

        self._statusbar.showMessage("GPU Ready - GLIC active")

        # Start render loop
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._render_frame)
        self._timer.start(16)  # ~60 FPS
        self._running = True

    def _render_frame(self) -> None:
        """Execute pipeline and update preview."""
        if not self._pipeline:
            return

        output = self._pipeline.execute()
        self._preview.set_texture(output)

        # Auto-capture debug screenshot after a few frames
        self._debug_frame_count += 1
        if self._debug_frame_count == 5 and not self._debug_auto_captured:
            self._debug_auto_captured = True
            self._save_debug_screenshot()

        # Update FPS counter
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time

        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._fps_label.setText(f"FPS: {self._fps:.1f}")
            self._frame_count = 0
            self._last_fps_time = now

    def _set_resolution(self, res: tuple[int, int]) -> None:
        """Set output resolution."""
        if self._pipeline:
            self._pipeline.set_output_size(res[0], res[1])
            self._res_label.setText(f"{res[0]}x{res[1]}")

    def _toggle_glic(self) -> None:
        """Toggle GLIC effect on/off."""
        self._glic_enabled = self._toggle_glic_action.isChecked()
        self._rebuild_pipeline()
        status = "GLIC enabled" if self._glic_enabled else "GLIC disabled"
        self._statusbar.showMessage(status)

    def _set_colorspace(self, cs: str) -> None:
        """Set GLIC color space (updates panel which updates node)."""
        self._glic_panel._colorspace_combo.setCurrentText(cs)

    def _set_prediction(self, pred: str) -> None:
        """Set GLIC prediction method (updates panel which updates node)."""
        self._glic_panel._prediction_combo.setCurrentText(pred)

    def _set_quantization(self, value: float) -> None:
        """Set GLIC quantization amount (updates panel which updates node)."""
        self._glic_panel._quant_slider.setValue(int(value))

    def _set_clamp(self, method: str) -> None:
        """Set GLIC clamp method (updates panel which updates node)."""
        self._glic_panel._clamp_combo.setCurrentText(method)

    def _use_testcard(self) -> None:
        """Switch to test card generator."""
        self._source_node = TestCardNode()
        self._rebuild_pipeline()
        self._statusbar.showMessage("Source: Test Card")

    def _rebuild_pipeline(self) -> None:
        """Rebuild pipeline with current settings."""
        if not self._pipeline or not self._source_node:
            return

        # Save current settings before clearing
        w, h = self._pipeline.size
        glic_settings = None
        if self._glic_node:
            glic_settings = {
                "colorspace": self._glic_panel.colorspace,
                "prediction": self._glic_panel.prediction,
                "quantization": self._glic_panel.quantization,
                "clamp": self._glic_panel.clamp,
            }

        # Clear pipeline (releases old nodes)
        self._pipeline.clear()
        self._pipeline.set_output_size(w, h)

        # Add source node (already a fresh instance)
        self._pipeline.add_node(self._source_node)

        # Add GLIC if enabled (create fresh instance)
        if self._glic_enabled:
            self._glic_node = GLICPipelineNode()
            # Restore settings
            if glic_settings:
                self._glic_node.set_param("colorspace", glic_settings["colorspace"])
                self._glic_node.set_param("prediction", glic_settings["prediction"])
                self._glic_node.set_param("quantization", glic_settings["quantization"])
                self._glic_node.set_param("clamp", glic_settings["clamp"])
            self._pipeline.add_node(self._glic_node)
            # Reconnect panel to new node
            self._glic_panel.set_glic_node(self._glic_node)

    def _load_image(self) -> None:
        """Load an image file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tiff *.bmp *.webp);;All Files (*)"
        )
        if path and self._pipeline:
            # Create image loader as new source
            self._source_node = ImageLoaderNode()
            self._source_node.set_param("path", path)
            self._rebuild_pipeline()
            self._statusbar.showMessage(f"Loaded: {path}")

    def _save_image(self) -> None:
        """Save current output to file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        if path and self._pipeline:
            # Get current output texture
            output = self._pipeline.execute()
            if output:
                # Create temporary saver node
                saver = ImageSaverNode()
                saver.set_param("path", path)
                saver.set_input("input", output)
                if saver.save():
                    self._statusbar.showMessage(f"Saved: {path}")
                else:
                    QMessageBox.warning(self, "Save Failed", "Could not save image.")

    # ─────────────────────────────────────────────────────────────────────────
    # Debug Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _save_debug_screenshot(self) -> None:
        """Save debug screenshot for verification."""
        if not self._pipeline:
            print("[DEBUG] No pipeline available")
            return

        output = self._pipeline.execute()
        if not output:
            print("[DEBUG] No output from pipeline")
            return

        try:
            # Download from GPU
            data = output.download()
            data = (np.clip(data, 0, 1) * 255).astype(np.uint8)

            # Save to project root
            debug_path = _PROJECT_ROOT / "debug_output.png"

            # Handle RGBA vs RGB
            if data.shape[2] == 4:
                img = Image.fromarray(data, mode="RGBA")
            else:
                img = Image.fromarray(data, mode="RGB")

            img.save(debug_path)

            print(f"[DEBUG] Screenshot saved: {debug_path}")
            self._statusbar.showMessage(f"Debug screenshot: {debug_path}")
        except Exception as e:
            print(f"[DEBUG] Screenshot failed: {e}")
            import traceback
            traceback.print_exc()

    def _dump_debug_info(self) -> None:
        """Dump current state info to console."""
        print("\n" + "=" * 60)
        print("[DEBUG INFO]")
        print(f"  Window size: {self.width()}x{self.height()}")
        print(f"  Window position: ({self.x()}, {self.y()})")
        print(f"  Pipeline: {self._pipeline is not None}")
        print(f"  GLIC enabled: {self._glic_enabled}")
        print(f"  FPS: {self._fps:.1f}")
        print(f"  Frame count: {self._debug_frame_count}")
        if self._pipeline:
            print(f"  Output size: {self._pipeline.size}")
        if self._glic_node:
            print(f"  Colorspace: {self._glic_node.get_param('colorspace')}")
            print(f"  Prediction: {self._glic_node.get_param('prediction')}")
            print(f"  Quantization: {self._glic_node.get_param('quantization')}")
            print(f"  Clamp: {self._glic_node.get_param('clamp')}")
        print("=" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Remote Command Interface
    # ─────────────────────────────────────────────────────────────────────────

    def _check_commands(self) -> None:
        """Check for and execute commands from file."""
        if not self._command_file.exists():
            return

        try:
            commands = self._command_file.read_text().strip()
            if not commands:
                return

            # Clear the file immediately
            self._command_file.write_text("")

            # Execute each command
            for line in commands.split("\n"):
                self._execute_command(line.strip())
        except Exception as e:
            print(f"[CMD] Error reading commands: {e}")

    def _execute_command(self, cmd: str) -> None:
        """Execute a single command."""
        if not cmd or cmd.startswith("#"):
            return

        print(f"[CMD] Executing: {cmd}")
        parts = cmd.split()
        action = parts[0].lower()

        try:
            if action == "screenshot":
                self._save_debug_screenshot()

            elif action == "colorspace" and len(parts) > 1:
                self._set_colorspace(parts[1])

            elif action == "prediction" and len(parts) > 1:
                self._set_prediction(" ".join(parts[1:]))

            elif action == "quantization" and len(parts) > 1:
                self._set_quantization(float(parts[1]))

            elif action == "clamp" and len(parts) > 1:
                self._set_clamp(" ".join(parts[1:]))

            elif action == "glic":
                if len(parts) > 1 and parts[1].lower() == "off":
                    self._toggle_glic_action.setChecked(False)
                    self._toggle_glic()
                else:
                    self._toggle_glic_action.setChecked(True)
                    self._toggle_glic()

            elif action == "testcard":
                self._use_testcard()

            elif action == "load" and len(parts) > 1:
                path = " ".join(parts[1:])
                self._source_node = ImageLoaderNode()
                self._source_node.set_param("path", path)
                self._rebuild_pipeline()
                self._statusbar.showMessage(f"Loaded: {path}")

            elif action == "resolution" and len(parts) > 1:
                size = int(parts[1])
                self._set_resolution((size, size))

            elif action == "info":
                self._dump_debug_info()

            elif action == "randomize":
                self._glic_panel._randomize_settings()

            elif action == "quit":
                self.close()

            else:
                print(f"[CMD] Unknown command: {cmd}")

        except Exception as e:
            print(f"[CMD] Failed '{cmd}': {e}")
            import traceback
            traceback.print_exc()

    def closeEvent(self, event) -> None:
        """Clean up on close."""
        self._running = False
        if hasattr(self, '_timer'):
            self._timer.stop()
        if hasattr(self, '_command_timer'):
            self._command_timer.stop()
        if self._pipeline:
            self._pipeline.release()
        if self._ctx:
            self._ctx.release()
        event.accept()


def apply_dark_theme(app: QApplication) -> None:
    """Apply dark color palette and stylesheet."""
    app.setStyle("Fusion")

    palette = QPalette()

    # Window colors
    palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 48))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(212, 212, 212))

    # Base colors
    palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 32))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(40, 40, 42))

    # Text
    palette.setColor(QPalette.ColorRole.Text, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(128, 128, 128))

    # Buttons
    palette.setColor(QPalette.ColorRole.Button, QColor(55, 55, 58))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))

    # Highlight
    palette.setColor(QPalette.ColorRole.Highlight, QColor(80, 140, 200))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    # Tooltips
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(30, 30, 32))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(212, 212, 212))

    # Disabled
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(128, 128, 128))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(128, 128, 128))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(128, 128, 128))

    app.setPalette(palette)

    # Additional styling
    app.setStyleSheet("""
        QMenuBar {
            background: #3a3a3c;
            border-bottom: 1px solid #505052;
        }
        QMenuBar::item:selected {
            background: #505052;
        }
        QMenu {
            background: #3a3a3c;
            border: 1px solid #505052;
        }
        QMenu::item:selected {
            background: #5080c8;
        }
        QStatusBar {
            background: #3a3a3c;
            border-top: 1px solid #505052;
        }
    """)


def run():
    """Launch the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Artifice")

    apply_dark_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
