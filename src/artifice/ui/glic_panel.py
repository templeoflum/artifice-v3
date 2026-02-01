"""GLIC Control Panel - dockable panel for GLIC effect settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from artifice.nodes import GLICPipelineNode


class GLICControlPanel(QWidget):
    """Control panel for GLIC effect parameters."""

    # Signals emitted when settings change
    colorspace_changed = Signal(str)
    prediction_changed = Signal(str)
    quantization_changed = Signal(float)
    clamp_changed = Signal(str)
    enabled_changed = Signal(bool)

    # Available options
    COLORSPACES = [
        "RGB", "OHTA", "CMY", "HSB", "XYZ", "YXY", "HCL", "LUV",
        "LAB", "HWB", "RGGBG", "YPbPr", "YCbCr", "YDbDr", "Grayscale", "YUV"
    ]
    PREDICTIONS = [
        "None", "Corner", "Horizontal", "Vertical", "DC", "DC Median",
        "Median", "Average", "TrueMotion", "Paeth", "L-Diagonal",
        "H/V Select", "JPEG-LS", "Difference", "Gradient", "Auto Select"
    ]
    CLAMP_METHODS = ["None", "Mod 256", "Absolute", "Clip"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._glic_node: GLICPipelineNode | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Title
        title = QLabel("GLIC Effect Controls")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #80c0ff;")
        layout.addWidget(title)

        # Color Space
        cs_group = QGroupBox("Color Space")
        cs_layout = QVBoxLayout(cs_group)
        self._colorspace_combo = QComboBox()
        self._colorspace_combo.addItems(self.COLORSPACES)
        self._colorspace_combo.setCurrentText("YCbCr")
        self._colorspace_combo.currentTextChanged.connect(self._on_colorspace_changed)
        cs_layout.addWidget(self._colorspace_combo)
        layout.addWidget(cs_group)

        # Prediction Method
        pred_group = QGroupBox("Prediction")
        pred_layout = QVBoxLayout(pred_group)
        self._prediction_combo = QComboBox()
        self._prediction_combo.addItems(self.PREDICTIONS)
        self._prediction_combo.setCurrentText("Paeth")
        self._prediction_combo.currentTextChanged.connect(self._on_prediction_changed)
        pred_layout.addWidget(self._prediction_combo)
        layout.addWidget(pred_group)

        # Quantization
        quant_group = QGroupBox("Quantization")
        quant_layout = QVBoxLayout(quant_group)

        # Slider + value display
        slider_layout = QHBoxLayout()
        self._quant_slider = QSlider(Qt.Horizontal)
        self._quant_slider.setRange(1, 128)
        self._quant_slider.setValue(16)
        self._quant_slider.valueChanged.connect(self._on_quant_slider_changed)

        self._quant_label = QLabel("16")
        self._quant_label.setMinimumWidth(30)
        self._quant_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        slider_layout.addWidget(self._quant_slider)
        slider_layout.addWidget(self._quant_label)
        quant_layout.addLayout(slider_layout)

        # Quick presets
        preset_layout = QHBoxLayout()
        for val in [4, 8, 16, 32, 64, 128]:
            btn = QPushButton(str(val))
            btn.setMaximumWidth(40)
            btn.clicked.connect(lambda checked, v=val: self._set_quantization(v))
            preset_layout.addWidget(btn)
        quant_layout.addLayout(preset_layout)

        layout.addWidget(quant_group)

        # Clamp Method
        clamp_group = QGroupBox("Clamp Method")
        clamp_layout = QVBoxLayout(clamp_group)
        self._clamp_combo = QComboBox()
        self._clamp_combo.addItems(self.CLAMP_METHODS)
        self._clamp_combo.setCurrentText("Mod 256")
        self._clamp_combo.currentTextChanged.connect(self._on_clamp_changed)
        clamp_layout.addWidget(self._clamp_combo)
        layout.addWidget(clamp_group)

        # Randomize button
        randomize_btn = QPushButton("Randomize")
        randomize_btn.clicked.connect(self._randomize_settings)
        layout.addWidget(randomize_btn)

        # Spacer
        layout.addStretch()

        # Apply styling
        self._apply_styles()

    def _apply_styles(self) -> None:
        """Apply dark theme styles."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #505052;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #a0a0a0;
            }
            QComboBox {
                padding: 4px 8px;
                background: #3a3a3c;
                border: 1px solid #505052;
                border-radius: 3px;
            }
            QComboBox:hover {
                border-color: #80c0ff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3a3a3c;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                margin: -4px 0;
                background: #80c0ff;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #a0d0ff;
            }
            QPushButton {
                padding: 4px 8px;
                background: #4a4a4c;
                border: 1px solid #606062;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #5a5a5c;
                border-color: #80c0ff;
            }
            QPushButton:pressed {
                background: #3a3a3c;
            }
        """)

    def set_glic_node(self, node: GLICPipelineNode | None) -> None:
        """Connect to a GLIC node for direct parameter updates."""
        self._glic_node = node
        if node:
            # Sync UI to node's current values
            self._colorspace_combo.setCurrentText(node.get_param("colorspace"))
            self._prediction_combo.setCurrentText(node.get_param("prediction"))
            self._quant_slider.setValue(int(node.get_param("quantization")))
            self._clamp_combo.setCurrentText(node.get_param("clamp"))

    def _on_colorspace_changed(self, value: str) -> None:
        """Handle color space change."""
        if self._glic_node:
            self._glic_node.set_param("colorspace", value)
        self.colorspace_changed.emit(value)

    def _on_prediction_changed(self, value: str) -> None:
        """Handle prediction method change."""
        if self._glic_node:
            self._glic_node.set_param("prediction", value)
        self.prediction_changed.emit(value)

    def _on_quant_slider_changed(self, value: int) -> None:
        """Handle quantization slider change."""
        self._quant_label.setText(str(value))
        if self._glic_node:
            self._glic_node.set_param("quantization", float(value))
        self.quantization_changed.emit(float(value))

    def _set_quantization(self, value: int) -> None:
        """Set quantization from preset button."""
        self._quant_slider.setValue(value)

    def _on_clamp_changed(self, value: str) -> None:
        """Handle clamp method change."""
        if self._glic_node:
            self._glic_node.set_param("clamp", value)
        self.clamp_changed.emit(value)

    def _randomize_settings(self) -> None:
        """Randomize all GLIC settings."""
        import random

        self._colorspace_combo.setCurrentIndex(random.randint(0, len(self.COLORSPACES) - 1))
        self._prediction_combo.setCurrentIndex(random.randint(0, len(self.PREDICTIONS) - 1))
        self._quant_slider.setValue(random.choice([4, 8, 16, 32, 64, 128]))
        self._clamp_combo.setCurrentIndex(random.randint(0, len(self.CLAMP_METHODS) - 1))

    # Public accessors for current values
    @property
    def colorspace(self) -> str:
        return self._colorspace_combo.currentText()

    @property
    def prediction(self) -> str:
        return self._prediction_combo.currentText()

    @property
    def quantization(self) -> float:
        return float(self._quant_slider.value())

    @property
    def clamp(self) -> str:
        return self._clamp_combo.currentText()
