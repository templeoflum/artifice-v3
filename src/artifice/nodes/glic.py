"""GLIC nodes - prediction, residual, quantization, reconstruction."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from artifice.engine.node import GPUNode, Parameter

if TYPE_CHECKING:
    from artifice.engine.context import GPUContext


class ColorSpaceNode(GPUNode):
    """Convert between color spaces.

    All 16 GLIC color spaces available.
    """

    name: ClassVar[str] = "Color Space"
    category: ClassVar[str] = "Color"
    shader_name: ClassVar[str] = "color/colorspace"

    COLORSPACES = [
        "RGB", "OHTA", "CMY", "HSB", "XYZ", "YXY", "HCL", "LUV",
        "LAB", "HWB", "RGGBG", "YPbPr", "YCbCr", "YDbDr", "Grayscale", "YUV"
    ]

    @classmethod
    def parameters(cls) -> list[Parameter]:
        return [
            Parameter(
                name="colorspace",
                type="enum",
                default="YCbCr",
                choices=cls.COLORSPACES,
                description="Target color space",
            ),
            Parameter(
                name="forward",
                type="bool",
                default=True,
                description="RGB to CS (on) or CS to RGB (off)",
            ),
        ]

    def execute(self, ctx: GPUContext) -> None:
        if not self._shader:
            return

        input_tex = self._inputs.get("input")
        output_tex = self._outputs.get("output")
        if not input_tex or not output_tex:
            return

        input_tex.mgl.bind_to_image(0, read=True, write=False)
        output_tex.mgl.bind_to_image(1, read=False, write=True)

        cs_idx = self.COLORSPACES.index(self.get_param("colorspace"))
        if "colorspace" in self._shader:
            self._shader["colorspace"].value = cs_idx
        if "forward" in self._shader:
            self._shader["forward"].value = self.get_param("forward")

        gx = (output_tex.width + 15) // 16
        gy = (output_tex.height + 15) // 16
        self._shader.run(gx, gy, 1)


class PredictNode(GPUNode):
    """GLIC-style prediction.

    Predicts pixel values from neighbors using various algorithms.
    """

    name: ClassVar[str] = "Predict"
    category: ClassVar[str] = "GLIC"
    shader_name: ClassVar[str] = "glic/predict"

    METHODS = [
        "None", "Corner", "Horizontal", "Vertical", "DC", "DC Median",
        "Median", "Average", "TrueMotion", "Paeth", "L-Diagonal",
        "H/V Select", "JPEG-LS", "Difference", "Gradient", "Auto Select"
    ]

    @classmethod
    def parameters(cls) -> list[Parameter]:
        return [
            Parameter(
                name="method",
                type="enum",
                default="Paeth",
                choices=cls.METHODS,
                description="Prediction method",
            ),
            Parameter(
                name="block_size",
                type="int",
                default=8,
                min_val=2,
                max_val=256,
                description="Block size for prediction",
            ),
        ]

    def execute(self, ctx: GPUContext) -> None:
        if not self._shader:
            return

        input_tex = self._inputs.get("input")
        output_tex = self._outputs.get("output")
        if not input_tex or not output_tex:
            return

        input_tex.mgl.bind_to_image(0, read=True, write=False)
        output_tex.mgl.bind_to_image(1, read=False, write=True)

        method_idx = self.METHODS.index(self.get_param("method"))
        if "prediction_method" in self._shader:
            self._shader["prediction_method"].value = method_idx
        if "block_size" in self._shader:
            self._shader["block_size"].value = self.get_param("block_size")

        gx = (output_tex.width + 15) // 16
        gy = (output_tex.height + 15) // 16
        self._shader.run(gx, gy, 1)


class ResidualNode(GPUNode):
    """Calculate residuals (original - prediction).

    The residual is where GLIC glitch effects emerge.
    """

    name: ClassVar[str] = "Residual"
    category: ClassVar[str] = "GLIC"
    shader_name: ClassVar[str] = "glic/residual"

    CLAMP_METHODS = ["None", "Mod 256", "Absolute", "Clip"]

    @classmethod
    def parameters(cls) -> list[Parameter]:
        return [
            Parameter(
                name="clamp_method",
                type="enum",
                default="Mod 256",
                choices=cls.CLAMP_METHODS,
                description="How to handle residual values",
            ),
            Parameter(
                name="scale",
                type="float",
                default=1.0,
                min_val=0.1,
                max_val=10.0,
                step=0.1,
                description="Residual amplification",
            ),
        ]

    @classmethod
    def input_slots(cls) -> list[str]:
        return ["original", "prediction"]

    def execute(self, ctx: GPUContext) -> None:
        if not self._shader:
            return

        original = self._inputs.get("original")
        prediction = self._inputs.get("prediction")
        output_tex = self._outputs.get("output")

        if not original or not prediction or not output_tex:
            return

        original.mgl.bind_to_image(0, read=True, write=False)
        prediction.mgl.bind_to_image(1, read=True, write=False)
        output_tex.mgl.bind_to_image(2, read=False, write=True)

        clamp_idx = self.CLAMP_METHODS.index(self.get_param("clamp_method"))
        if "clamp_method" in self._shader:
            self._shader["clamp_method"].value = clamp_idx
        if "scale" in self._shader:
            self._shader["scale"].value = self.get_param("scale")

        gx = (output_tex.width + 15) // 16
        gy = (output_tex.height + 15) // 16
        self._shader.run(gx, gy, 1)


class ReconstructNode(GPUNode):
    """Reconstruct image from prediction + residual.

    With quantized residuals, produces characteristic GLIC glitch look.
    """

    name: ClassVar[str] = "Reconstruct"
    category: ClassVar[str] = "GLIC"
    shader_name: ClassVar[str] = "glic/reconstruct"

    CLAMP_METHODS = ["None", "Mod 256", "Absolute", "Clip"]

    @classmethod
    def parameters(cls) -> list[Parameter]:
        return [
            Parameter(
                name="clamp_method",
                type="enum",
                default="Mod 256",
                choices=cls.CLAMP_METHODS,
                description="Must match residual's clamp method",
            ),
            Parameter(
                name="scale",
                type="float",
                default=1.0,
                min_val=0.1,
                max_val=10.0,
                step=0.1,
                description="Inverse of residual scale",
            ),
        ]

    @classmethod
    def input_slots(cls) -> list[str]:
        return ["prediction", "residual"]

    def execute(self, ctx: GPUContext) -> None:
        if not self._shader:
            return

        prediction = self._inputs.get("prediction")
        residual = self._inputs.get("residual")
        output_tex = self._outputs.get("output")

        if not prediction or not residual or not output_tex:
            return

        prediction.mgl.bind_to_image(0, read=True, write=False)
        residual.mgl.bind_to_image(1, read=True, write=False)
        output_tex.mgl.bind_to_image(2, read=False, write=True)

        clamp_idx = self.CLAMP_METHODS.index(self.get_param("clamp_method"))
        if "clamp_method" in self._shader:
            self._shader["clamp_method"].value = clamp_idx
        if "scale" in self._shader:
            self._shader["scale"].value = self.get_param("scale")

        gx = (output_tex.width + 15) // 16
        gy = (output_tex.height + 15) // 16
        self._shader.run(gx, gy, 1)


class QuantizeNode(GPUNode):
    """Quantize values - reduces precision for glitch effects."""

    name: ClassVar[str] = "Quantize"
    category: ClassVar[str] = "GLIC"
    shader_name: ClassVar[str] = "glic/quantize"

    MODES = ["Uniform", "Levels", "Posterize", "Bitcrush"]

    @classmethod
    def parameters(cls) -> list[Parameter]:
        return [
            Parameter(
                name="mode",
                type="enum",
                default="Uniform",
                choices=cls.MODES,
                description="Quantization mode",
            ),
            Parameter(
                name="value",
                type="float",
                default=8.0,
                min_val=1.0,
                max_val=128.0,
                step=1.0,
                description="Quantization amount",
            ),
        ]

    def execute(self, ctx: GPUContext) -> None:
        if not self._shader:
            return

        input_tex = self._inputs.get("input")
        output_tex = self._outputs.get("output")
        if not input_tex or not output_tex:
            return

        input_tex.mgl.bind_to_image(0, read=True, write=False)
        output_tex.mgl.bind_to_image(1, read=False, write=True)

        mode_idx = self.MODES.index(self.get_param("mode"))
        if "quant_mode" in self._shader:
            self._shader["quant_mode"].value = mode_idx
        if "quant_value" in self._shader:
            self._shader["quant_value"].value = self.get_param("value")
        if "forward" in self._shader:
            self._shader["forward"].value = True

        gx = (output_tex.width + 15) // 16
        gy = (output_tex.height + 15) // 16
        self._shader.run(gx, gy, 1)


class GLICPipelineNode(GPUNode):
    """Complete GLIC encode/decode in one node.

    Combines: ColorSpace -> Predict -> Residual -> Quantize -> Reconstruct -> ColorSpace
    This is the full GLIC glitch effect in a single node.
    """

    name: ClassVar[str] = "GLIC Pipeline"
    category: ClassVar[str] = "GLIC"
    shader_name: ClassVar[str] = ""  # Uses multiple shaders

    COLORSPACES = ColorSpaceNode.COLORSPACES
    PREDICTIONS = PredictNode.METHODS
    CLAMP_METHODS = ResidualNode.CLAMP_METHODS

    @classmethod
    def parameters(cls) -> list[Parameter]:
        return [
            Parameter(
                name="colorspace",
                type="enum",
                default="YCbCr",
                choices=cls.COLORSPACES,
                description="Working color space",
            ),
            Parameter(
                name="prediction",
                type="enum",
                default="Paeth",
                choices=cls.PREDICTIONS,
                description="Prediction method",
            ),
            Parameter(
                name="quantization",
                type="float",
                default=16.0,
                min_val=1.0,
                max_val=128.0,
                step=1.0,
                description="Quantization amount (higher = more glitch)",
            ),
            Parameter(
                name="clamp",
                type="enum",
                default="Mod 256",
                choices=cls.CLAMP_METHODS,
                description="Residual clamping method",
            ),
        ]

    def __init__(self):
        super().__init__()
        # Internal shaders
        self._cs_shader = None
        self._pred_shader = None
        self._res_shader = None
        self._quant_shader = None
        self._recon_shader = None
        # Intermediate textures
        self._tex_cs = None
        self._tex_pred = None
        self._tex_res = None
        self._tex_quant = None
        self._tex_recon = None

    def compile(self, ctx: GPUContext) -> None:
        self._cs_shader = ctx.load_shader("color/colorspace")
        self._pred_shader = ctx.load_shader("glic/predict")
        self._res_shader = ctx.load_shader("glic/residual")
        self._quant_shader = ctx.load_shader("glic/quantize")
        self._recon_shader = ctx.load_shader("glic/reconstruct")

    def allocate(self, ctx: GPUContext, width: int, height: int) -> None:
        from artifice.engine.texture import Texture

        # Allocate output
        super().allocate(ctx, width, height)

        # Allocate intermediate textures
        for name in ['_tex_cs', '_tex_pred', '_tex_res', '_tex_quant', '_tex_recon']:
            existing = getattr(self, name)
            if existing is None or existing.size != (width, height):
                if existing:
                    existing.release()
                tex = ctx.create_texture(width, height)
                setattr(self, name, Texture(tex))

    def execute(self, ctx: GPUContext) -> None:
        input_tex = self._inputs.get("input")
        output_tex = self._outputs.get("output")
        if not input_tex or not output_tex:
            return

        w, h = output_tex.size
        gx = (w + 15) // 16
        gy = (h + 15) // 16

        cs_idx = self.COLORSPACES.index(self.get_param("colorspace"))
        pred_idx = self.PREDICTIONS.index(self.get_param("prediction"))
        clamp_idx = self.CLAMP_METHODS.index(self.get_param("clamp"))
        quant_val = self.get_param("quantization")

        # Step 1: RGB -> ColorSpace
        input_tex.mgl.bind_to_image(0, read=True, write=False)
        self._tex_cs.mgl.bind_to_image(1, read=False, write=True)
        self._cs_shader["colorspace"].value = cs_idx
        self._cs_shader["forward"].value = True
        self._cs_shader.run(gx, gy, 1)
        ctx.memory_barrier()

        # Step 2: Predict
        self._tex_cs.mgl.bind_to_image(0, read=True, write=False)
        self._tex_pred.mgl.bind_to_image(1, read=False, write=True)
        self._pred_shader["prediction_method"].value = pred_idx
        self._pred_shader["block_size"].value = 8
        self._pred_shader.run(gx, gy, 1)
        ctx.memory_barrier()

        # Step 3: Calculate Residual
        self._tex_cs.mgl.bind_to_image(0, read=True, write=False)
        self._tex_pred.mgl.bind_to_image(1, read=True, write=False)
        self._tex_res.mgl.bind_to_image(2, read=False, write=True)
        self._res_shader["clamp_method"].value = clamp_idx
        self._res_shader["scale"].value = 1.0
        self._res_shader.run(gx, gy, 1)
        ctx.memory_barrier()

        # Step 4: Quantize Residual
        self._tex_res.mgl.bind_to_image(0, read=True, write=False)
        self._tex_quant.mgl.bind_to_image(1, read=False, write=True)
        self._quant_shader["quant_mode"].value = 0  # Uniform
        self._quant_shader["quant_value"].value = quant_val
        self._quant_shader["forward"].value = True
        self._quant_shader.run(gx, gy, 1)
        ctx.memory_barrier()

        # Step 5: Reconstruct
        self._tex_pred.mgl.bind_to_image(0, read=True, write=False)
        self._tex_quant.mgl.bind_to_image(1, read=True, write=False)
        self._tex_recon.mgl.bind_to_image(2, read=False, write=True)
        self._recon_shader["clamp_method"].value = clamp_idx
        self._recon_shader["scale"].value = 1.0
        self._recon_shader.run(gx, gy, 1)
        ctx.memory_barrier()

        # Step 6: ColorSpace -> RGB
        self._tex_recon.mgl.bind_to_image(0, read=True, write=False)
        output_tex.mgl.bind_to_image(1, read=False, write=True)
        self._cs_shader["colorspace"].value = cs_idx
        self._cs_shader["forward"].value = False
        self._cs_shader.run(gx, gy, 1)

    def release(self) -> None:
        super().release()
        for name in ['_tex_cs', '_tex_pred', '_tex_res', '_tex_quant', '_tex_recon']:
            tex = getattr(self, name)
            if tex:
                tex.release()
                setattr(self, name, None)
