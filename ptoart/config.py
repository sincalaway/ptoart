"""Validated public inference settings; imports no ML frameworks."""
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ArtConfig:
    backend: str = "torch"
    device: str = "auto"
    model_dir: Path | str | None = None
    weights: Path | str | None = None
    segmentation: Path | str | None = None
    aligned: bool = False
    detector: str = "dlib"
    background: tuple[int, int, int] | None = (255, 255, 255)
    strength: float = 1.0
    max_pixels: int = 25_000_000

    def __post_init__(self):
        if self.backend not in {"torch", "onnx"}:
            raise ValueError("backend must be torch or onnx")
        if self.device not in {"auto", "cpu", "cuda", "cuda:0"}:
            raise ValueError("device must be auto, cpu, cuda or cuda:0")
        if self.detector not in {"dlib", "sfd"}:
            raise ValueError("detector must be dlib or sfd")
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if self.max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        if self.background is not None and (
            len(self.background) != 3
            or any(not isinstance(x, int) or not 0 <= x <= 255 for x in self.background)
        ):
            raise ValueError("background must contain three integer RGB values from 0 to 255")
        root = Path(self.model_dir or os.environ.get("PTOART_MODEL_DIR", "~/.cache/ptoart/models"))
        object.__setattr__(self, "model_dir", root.expanduser().resolve())
        filename = "photo2cartoon_weights.pt" if self.backend == "torch" else "photo2cartoon_weights.onnx"
        object.__setattr__(self, "weights", Path(self.weights or root / filename).expanduser().resolve())
        object.__setattr__(self, "segmentation", Path(self.segmentation or root / "seg_model_384.pb").expanduser().resolve())
