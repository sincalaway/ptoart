"""Append every release model to the Windows portable ZIP (used by .github/workflows/release.yml).

Layout added inside the ZIP, next to ptoart.exe (``<root>/weights/...``):

- ``weights/ptoart_weights.pt``, ``weights/seg_model_384.pb`` (and ``ptoart_weights.onnx`` for the
  CLI ``--backend onnx``) from the ``ptoart-models`` artifact
- ``weights/styles/``: everything in ``ptoart-styles.zip`` and ``ptoart-dctnet.zip`` (models and
  their licence/readme .txt files)
- ``weights/upscale/``: everything in ``ptoart-upscalers-realesrgan.zip``

The ZIP is appended to in place, keeping the path separator the EXE build used (PowerShell's
Compress-Archive may write backslashes). Entries that already exist are left alone. Usage::

    python tools/bundle_windows_models.py ptoart-windows-x64.zip --models models/ --dctnet dct/
"""
import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

GITHUB_ASSET_LIMIT = 2 * 1024 ** 3  # GitHub release assets must be under 2 GiB
LOOSE_MODELS = ("ptoart_weights.pt", "seg_model_384.pb", "ptoart_weights.onnx")
REQUIRED = (
    "weights/ptoart_weights.pt",
    "weights/seg_model_384.pb",
    "weights/styles/face_paint_512_v2.onnx",
    "weights/styles/hayao.onnx",
    "weights/styles/jp_face.onnx",
    "weights/styles/dct_3d.onnx",
    "weights/styles/dct_anime_bg.onnx",
    "weights/styles/LICENSE-AnimeGANv2.txt",
    "weights/styles/LICENSE-AnimeGANv3.txt",
    "weights/styles/LICENSE-DCT-Net-Apache-2.0.txt",
    "weights/upscale/anime.onnx",
    "weights/upscale/general.onnx",
    "weights/upscale/LICENSE-Real-ESRGAN.txt",
)


def zip_layout(names):
    """Return (separator, root folder) of an existing ZIP's entries."""
    sep = "\\" if any("\\" in n for n in names) else "/"
    root = names[0].replace("\\", "/").split("/")[0]
    return sep, root


def plan(models_dir, dctnet_dir):
    """List (folder under weights, file name, opener) for every file to add."""
    models_dir, dctnet_dir = Path(models_dir), Path(dctnet_dir)
    items = []
    for name in LOOSE_MODELS:
        path = models_dir / name
        if path.is_file():
            items.append(("", name, lambda p=path: p.open("rb")))
        elif name != "ptoart_weights.onnx":
            raise SystemExit(f"missing {path}")
    archives = (
        (models_dir / "ptoart-styles.zip", "styles"),
        (dctnet_dir / "ptoart-dctnet.zip", "styles"),
        (models_dir / "ptoart-upscalers-realesrgan.zip", "upscale"),
    )
    for archive, folder in archives:
        if not archive.is_file():
            raise SystemExit(f"missing {archive}")
        src = zipfile.ZipFile(archive)
        for member in src.namelist():
            if member.endswith("/"):
                continue
            items.append((folder, member.replace("\\", "/").split("/")[-1],
                          lambda s=src, m=member: s.open(m)))
    return items


def bundle(zip_path, models_dir, dctnet_dir):
    items = plan(models_dir, dctnet_dir)
    added = []
    with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        names = z.namelist()
        sep, root = zip_layout(names)
        existing = {n.replace("\\", "/") for n in names}
        for folder, name, opener in items:
            parts = [root, "weights"] + ([folder] if folder else []) + [name]
            if "/".join(parts) in existing:
                continue
            info = zipfile.ZipInfo(sep.join(parts), date_time=time.localtime()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with opener() as fin, z.open(info, "w") as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)
            existing.add("/".join(parts))
            added.append("/".join(parts))
    missing = [r for r in REQUIRED if f"{root}/{r}" not in existing]
    if missing:
        raise SystemExit(f"models missing from the ZIP: {missing}")
    size = Path(zip_path).stat().st_size
    print(f"added {len(added)} files under {root}/weights (separator {sep!r}); ZIP is {size / 1e9:.2f} GB")
    for name in added:
        print("  +", name)
    if size >= GITHUB_ASSET_LIMIT:
        raise SystemExit(f"ZIP is {size} bytes, over GitHub's 2 GiB release asset limit")
    return added


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip", type=Path)
    parser.add_argument("--models", type=Path, required=True, help="ptoart-models artifact folder")
    parser.add_argument("--dctnet", type=Path, required=True, help="ptoart-dctnet artifact folder")
    args = parser.parse_args(argv)
    bundle(args.zip, args.models, args.dctnet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
