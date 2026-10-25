"""Command line interface; --help and doctor do not import ML frameworks."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from . import __version__
from .config import ArtConfig
from .errors import PtoartError
from .io import SUPPORTED, read_image, write_image


def parse_background(value):
    if value.lower() == "transparent":
        return None
    value = value.lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError("Use RRGGBB or transparent")
    try:
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use RRGGBB or transparent") from exc


def add_options(parser):
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0"], default="auto")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--segmentation", type=Path)
    parser.add_argument("--aligned", action="store_true", help="Input is an already aligned square portrait")
    parser.add_argument("--detector", choices=["dlib", "sfd"], default="dlib")
    parser.add_argument("--background", type=parse_background, default=(255, 255, 255), metavar="RRGGBB|transparent")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=25_000_000)
    parser.add_argument("--output", choices=["face", "photo", "style"], default=None,
                        help="face: square cartoon avatar; photo: whole photo with cartoon faces "
                             "pasted back; style: whole-image AnimeGANv2 style (default: face, "
                             "or style when --style is given)")
    parser.add_argument("--style", help="Style name in <model-dir>/styles (hayao, shinkai, paprika, "
                                        "celeba_distill, face_paint_512_v1, face_paint_512_v2) or a .onnx path. "
                                        "With --output face/photo it paints the faces instead of the "
                                        "default generator (face_paint_512_v2 recommended)")
    parser.add_argument("--upscale", help="Super-resolution model in <model-dir>/upscale (anime, general) "
                                          "or a .onnx path; used when output is larger than the model's")
    parser.add_argument("--style-max-side", type=int, default=1536,
                        help="Longest side used for style inference; larger is sharper but slower")
    parser.add_argument("--size", type=int, default=256, help="Avatar size in pixels for --output face (64-2048)")
    parser.add_argument("--max-faces", type=int, default=10, help="Faces to convert for --output photo")


def config_from(args):
    keys = ("backend", "device", "model_dir", "weights", "segmentation", "aligned",
            "detector", "background", "strength", "max_pixels", "output", "size", "max_faces",
            "style", "style_max_side", "upscale")
    values = {key: getattr(args, key) for key in keys}
    if values["output"] is None:
        values["output"] = "style" if values["style"] else "face"
    config = ArtConfig(**values)
    from ptoart.gui import use_bundled_torch_hub

    use_bundled_torch_hub(config.model_dir)
    return config


def doctor(config):
    if config.output == "style":
        dependencies = {p: importlib.util.find_spec(p) is not None for p in ("numpy", "cv2", "PIL", "onnxruntime")}
        models = {str(config.style): config.style.is_file()}
        if config.upscale:
            models[str(config.upscale)] = config.upscale.is_file()
        return {"version": __version__, "output": "style", "dependencies": dependencies, "models": models,
                "ready": all(dependencies.values()) and all(models.values()),
                "note": "Presence checks only; not a model-load or inference validation."}
    packages = ["numpy", "cv2", "PIL", "torch" if config.backend == "torch" else "onnxruntime"]
    if not config.aligned:
        packages += ["face_alignment", "tensorflow"]
        if config.detector == "dlib":
            packages.append("dlib")
    dependencies = {p: importlib.util.find_spec(p) is not None for p in packages}
    weights = {str(config.weights): config.weights.is_file()}
    if not config.aligned:
        weights[str(config.segmentation)] = config.segmentation.is_file()
    if config.upscale:
        weights[str(config.upscale)] = config.upscale.is_file()
        dependencies.setdefault("onnxruntime", importlib.util.find_spec("onnxruntime") is not None)
    return {"version": __version__, "backend": config.backend, "device": config.device,
            "dependencies": dependencies, "models": weights,
            "ready": all(dependencies.values()) and all(weights.values()),
            "note": "Presence checks only; not a model-load, CUDA or inference validation."}


def collect_batch_inputs(source, destination, *, recursive=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise ValueError(f"Not an input directory: {source}")
    if source == destination or source.is_relative_to(destination):
        raise ValueError("Output directory must not equal or contain the input directory")
    iterator = source.rglob("*") if recursive else source.iterdir()
    files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED
                   and not p.resolve().is_relative_to(destination))
    if not files:
        raise ValueError("No supported images found")
    return source, destination, files


def batch(pipeline, source, destination, *, recursive=False, overwrite=False):
    source, destination, files = collect_batch_inputs(source, destination, recursive=recursive)
    results = []
    for path in files:
        # Keep the original extension in the stem to avoid a.jpg / a.png collisions.
        target = destination / path.relative_to(source).with_name(path.name + ".png")
        try:
            pipeline.convert_file(path, target, overwrite=overwrite)
            results.append({"input": str(path), "output": str(target), "status": "ok"})
        except Exception as exc:
            results.append({"input": str(path), "output": str(target),
                            "status": "error", "error": str(exc), "error_type": type(exc).__name__})
    return {"total": len(results), "succeeded": sum(x["status"] == "ok" for x in results),
            "failed": sum(x["status"] == "error" for x in results), "results": results}


def build_parser():
    parser = argparse.ArgumentParser(prog="ptoart", description="ptoart: portrait to cartoon")
    parser.add_argument("--version", action="version", version=f"ptoart {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("convert", help="Convert one photo")
    single.add_argument("source", type=Path)
    single.add_argument("destination", type=Path)
    single.add_argument("--overwrite", action="store_true")
    directory = commands.add_parser("batch", help="Convert a directory; continue after per-image errors")
    directory.add_argument("source", type=Path)
    directory.add_argument("destination", type=Path)
    directory.add_argument("--recursive", action="store_true")
    directory.add_argument("--overwrite", action="store_true")
    directory.add_argument("--report", type=Path, help="Write JSON report, refusing to overwrite existing files")
    check = commands.add_parser("doctor", help="Check dependency and model presence, without loading models")
    for sub in (single, directory, check):
        add_options(sub)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        config = config_from(args)
        if args.command == "doctor":
            result = doctor(config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] else 2
        image = None
        if args.command == "convert":
            if args.source.resolve() == args.destination.resolve():
                raise ValueError("Input and output must differ; original photos are never overwritten")
            if args.destination.exists() and not args.overwrite:
                raise ValueError(f"Output already exists: {args.destination}; use --overwrite")
            if not args.source.is_file():
                raise ValueError(f"Not an input image file: {args.source}")
            if args.source.suffix.lower() not in SUPPORTED:
                raise ValueError(f"Unsupported input extension: {args.source.suffix}")
            suffix = args.destination.suffix.lower()
            if suffix not in SUPPORTED:
                raise ValueError(f"Unsupported output extension: {suffix}")
            if config.background is None and config.output == "face" and suffix not in {".png", ".webp", ".tif", ".tiff"}:
                raise ValueError("Transparent output requires PNG, WebP or TIFF")
            image = read_image(args.source, config.max_pixels)
            if min(image.shape[:2]) < 2:
                raise ValueError("Input image must be at least 2 by 2 pixels")
            if config.output == "face" and config.aligned and image.shape[0] != image.shape[1]:
                raise ValueError("--aligned requires a square, pre-aligned portrait")
        else:
            collect_batch_inputs(args.source, args.destination, recursive=args.recursive)
            if args.report and args.report.exists():
                raise ValueError(f"Report already exists: {args.report}")
        from .pipeline import ArtPipeline
        with ArtPipeline(config) as pipeline:
            if args.command == "convert":
                output = write_image(args.destination, pipeline.convert(image), overwrite=args.overwrite)
                print(output)
                return 0
            report = batch(pipeline, args.source, args.destination,
                           recursive=args.recursive, overwrite=args.overwrite)
        content = json.dumps(report, ensure_ascii=False, indent=2)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as handle:
                handle.write(content + "\n")
        print(content)
        return 1 if report["failed"] else 0
    except (PtoartError, ValueError, OSError, ImportError) as exc:
        print(f"ptoart: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
