"""Backend adapters. A backend accepts and returns float32 NCHW arrays."""
from pathlib import Path
import numpy as np

from .errors import ModelError


def require_file(path):
    if not Path(path).is_file():
        raise ModelError(f"Missing model: {path}. Run ptoart doctor and consult docs/MODELS.md.")


class TorchBackend:
    def __init__(self, config):
        require_file(config.weights)
        try:
            import torch
            from models.networks import ResnetGenerator
        except ImportError as exc:
            raise ModelError("Torch backend requires: pip install 'ptoart[torch]'") from exc
        device = config.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelError("CUDA was requested but is not available; use --device cpu")
        self.torch = torch
        self.device = torch.device(device)
        try:
            self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
            # Never fall back to unrestricted pickle deserialization.
            params = torch.load(config.weights, map_location=self.device, weights_only=True)
            state = params["genA2B"]
            self.net.load_state_dict(state, strict=True)
            self.net.eval()
        except Exception as exc:
            raise ModelError(f"Unable to load trusted genA2B checkpoint: {exc}") from exc

    def run(self, face):
        with self.torch.inference_mode():
            output = self.net(self.torch.from_numpy(face).to(self.device))[0]
        return output.cpu().numpy()

    def close(self):
        self.net = None


class OnnxBackend:
    def __init__(self, config):
        require_file(config.weights)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelError("ONNX backend requires: pip install 'ptoart[onnx]'") from exc
        available = ort.get_available_providers()
        if config.device.startswith("cuda") and "CUDAExecutionProvider" not in available:
            raise ModelError("CUDAExecutionProvider is unavailable; install onnxruntime-gpu or use cpu")
        use_cuda = config.device != "cpu" and "CUDAExecutionProvider" in available
        providers = (["CUDAExecutionProvider"] if use_cuda else []) + ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(config.weights), providers=providers)
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            if len(inputs) != 1 or inputs[0].type != "tensor(float)":
                raise ValueError("Expected one float32 image input")
            shape = inputs[0].shape
            if len(shape) != 4 or any(isinstance(v, int) and v != expected
                                      for v, expected in zip(shape, (1, 3, 256, 256))):
                raise ValueError(f"Expected NCHW [1,3,256,256], got {shape}")
            self.input_name = inputs[0].name
            self.output_name = next((o.name for o in outputs if o.name == "output"), outputs[0].name)
        except Exception as exc:
            raise ModelError(f"Unable to load ONNX model: {exc}") from exc

    def run(self, face):
        return self.session.run([self.output_name], {self.input_name: face})[0]

    def close(self):
        self.session = None


class StyleBackend:
    """Whole-image style ONNX generator: NCHW [-1,1] in and out, any size.

    Optional ``metadata_props`` (written by tools/export_dctnet.py): ``ptoart_multiple``,
    the side multiple the graph needs (default 8), and ``ptoart_face_side``, the square
    size the model expects when it paints a face crop (default 512).
    """

    def __init__(self, model_path):
        require_file(model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelError("Style output requires: pip install 'ptoart[onnx]'") from exc
        try:
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            inputs = self.session.get_inputs()
            if len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].shape[1] not in (3, "3"):
                raise ValueError(f"Expected one NCHW RGB input, got {inputs[0].shape}")
            self.input_name = inputs[0].name
            self.output_name = self.session.get_outputs()[0].name
            meta = self.session.get_modelmeta().custom_metadata_map
            self.multiple = max(8, int(meta.get("ptoart_multiple", 8)))
            self.face_side = int(meta.get("ptoart_face_side", 512))
        except Exception as exc:
            raise ModelError(f"Unable to load style model {model_path}: {exc}") from exc

    def run(self, image):
        return self.session.run([self.output_name], {self.input_name: image})[0]

    def close(self):
        self.session = None


class UpscaleBackend:
    """Real-ESRGAN-style ONNX upscaler: NCHW RGB in [0,1], returns ``scale`` times larger."""

    def __init__(self, model_path, tile=256, pad=10):
        require_file(model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelError("Upscaling requires: pip install 'ptoart[onnx]'") from exc
        try:
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            meta = self.session.get_modelmeta().custom_metadata_map
            self.scale = int(meta.get("ptoart_scale", 4))
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as exc:
            raise ModelError(f"Unable to load upscaler {model_path}: {exc}") from exc
        self.tile, self.pad = tile, pad

    def run(self, image):
        """Upscale float32 HWC RGB in [0,255]; tiles keep memory bounded on large images."""
        h, w = image.shape[:2]
        s, tile, pad = self.scale, self.tile, self.pad
        out = np.empty((h * s, w * s, 3), dtype=np.float32)
        src = image.astype(np.float32) / 255.0
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y0, x0 = max(0, y - pad), max(0, x - pad)
                y1, x1 = min(h, y + tile + pad), min(w, x + tile + pad)
                patch = np.ascontiguousarray(src[y0:y1, x0:x1].transpose(2, 0, 1)[None])
                result = self.session.run([self.output_name], {self.input_name: patch})[0][0]
                if result.shape != (3, (y1 - y0) * s, (x1 - x0) * s):
                    raise ModelError(f"Invalid upscaler output shape {result.shape}")
                th, tw = min(tile, h - y), min(tile, w - x)
                oy, ox = (y - y0) * s, (x - x0) * s
                out[y * s:(y + th) * s, x * s:(x + tw) * s] = \
                    result[:, oy:oy + th * s, ox:ox + tw * s].transpose(1, 2, 0)
        return np.clip(out, 0, 1) * 255.0

    def close(self):
        self.session = None


def create_backend(config):
    return {"torch": TorchBackend, "onnx": OnnxBackend}[config.backend](config)


def validate_output(output):
    output = np.asarray(output)
    if output.shape != (1, 3, 256, 256) or not np.isfinite(output).all():
        raise ModelError(f"Invalid model output; expected finite [1,3,256,256], got {output.shape}")
    return output
