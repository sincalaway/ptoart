# ptoart

A maintainable refactor of [photo2cartoon](https://github.com/minivision-ai/photo2cartoon), preserving its generator and training algorithms while adding a shared Torch/ONNX pipeline, batch processing, structured errors, background/alpha output, intensity blending and diagnostics.

[Complete Chinese tutorial](README.md) · [Windows package guide (Chinese)](packaging/WINDOWS-README.md) · [Model setup](docs/MODELS.md) · [Model download (Guangya Pan)](https://www.guangyapan.com/s/1961237335383875651_alCCcsK8loJ3qRN9)

> **Version 0.2.0 is an engineering preview. Official weights are not bundled.** Windows packaging and frozen-runtime smoke tests passed, but full real-photo inference with official weights has not been validated. A successful build, `doctor` result or random-weight test is not evidence of visual quality.

## Windows portable quick start

For Windows 10/11 x64, CPU only; no Python installation required. This is a **console application, not a GUI**. CUDA and dlib are not included.

1. Open [Windows EXE workflow runs](https://github.com/sincalaway/ptoart/actions/workflows/windows-exe.yml) and select a successful run on `refactor/ptoart-20261023`.
2. Download **ptoart-windows-x64-cpu** under **Artifacts** (web downloads typically require GitHub sign-in). Extract the downloaded ZIP, then extract the inner `ptoart-windows-x64-cpu.zip`.
3. Keep the entire `ptoart` folder, including `ptoart.exe` and `_internal`. Do not run from inside the ZIP or copy only the executable.
4. Download the weights from the [Guangya Pan share](https://www.guangyapan.com/s/1961237335383875651_alCCcsK8loJ3qRN9) (other sources: [model setup](docs/MODELS.md)). The share has no ONNX file; create one with `python tools/export_onnx.py photo2cartoon_weights.pt`. Put `photo2cartoon_weights.pt` and `seg_model_384.pb` in a `weights` folder beside the EXE; the bundled `weights/README.txt` is not a model.
5. Put your photo at `portrait.jpg`, open PowerShell and change to the EXE folder (replace the example location):

```powershell
cd "C:\ptoart"
.\ptoart.exe --version
.\ptoart.exe --help
.\ptoart.exe doctor --model-dir .\weights --device cpu
.\ptoart.exe convert .\portrait.jpg .\output\avatar.png --model-dir .\weights --device cpu
.\ptoart.exe batch .\photos .\output-batch --recursive --model-dir .\weights --device cpu --report .\batch-report.json
```

Always specify `--model-dir .\weights` for this layout; the folder is not automatically selected just because it is beside the EXE. Relative paths use the current working directory. Quote paths containing spaces.

The Windows package defaults to **SFD**, unlike the source CLI's dlib default. Face-alignment may still download detector/landmark assets on first use: keep the first real-photo run online. Offline first launch and these downloads have not been validated.

Artifacts are retained for 30 days. A known successful historical build is [run 39937315949](https://github.com/sincalaway/ptoart/actions/runs/39937315949), source `97a3d9f`; use newer successful runs when available. The workflow runs on pushes to `refactor/ptoart-20261023`, `v*` tags and manual dispatch. Update the branch filter if the default branch changes. `BUILD.txt` identifies the source commit; `SHA256SUMS.txt` checks the inner ZIP. The executable is unsigned: use trusted builds. Extract updates into a new folder rather than mixing `_internal` files.

## Options and troubleshooting

- `doctor` checks module/file presence only, not model loading, validity, CUDA or inference quality. `ready: true` returns 0; missing items return 2.
- Default output is a **256×256 cropped portrait**, not a full-image cartoon or an HD result. Multiple faces use the upstream largest-face policy.
- `--background transparent` requires PNG, WebP or TIFF output. Default background is white. Segmentation quality determines alpha edges.
- `--strength 0.7` blends 70% model output with 30% original cropped portrait; it is not learned style control. Range: 0–1, default 1.
- `--backend onnx` requires `photo2cartoon_weights.onnx`. Normal-photo preprocessing still needs segmentation, face-alignment/Torch and TensorFlow; this is not an ONNX-only pipeline.
- `--aligned` is only for already aligned, tightly cropped square portraits. It still needs generator weights, skips detection/segmentation and treats the whole image as opaque. Do not use it merely to bypass errors on normal photos.
- Existing images are not overwritten unless `--overwrite` is given. Source and destination must always differ. Existing JSON reports are **never** overwritten, even with `--overwrite`; choose a new report name.
- Batch outputs retain relative directories and append `.png` to the original filename (`person.jpg.png`). Per-image failures are recorded and processing continues after successful model initialization.
- Exit codes: **0** success; **1** batch has failed images; **2** command-level error or doctor not ready.

If double-clicking makes a window disappear, run from PowerShell to see the error. For missing models, check real files and `--model-dir`; presence alone does not prove valid weights. For no-face errors, try a clear front-facing single-person photo. Do not request CUDA or dlib with this CPU package.

## Install from source

Python 3.11 in a virtual environment is recommended; the base package requires Python 3.10+. Machine-learning dependency combinations still require platform-specific validation.

```bash
git clone --branch refactor/ptoart-20261023 https://github.com/sincalaway/ptoart.git
cd ptoart
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[torch,face]"
```

The source CLI defaults to **dlib**, which may need C++ build tools and CMake. For pre-aligned portraits only, install `.[torch]` and use `--aligned`; ONNX users can install `.[onnx,face]`. Avoid mixing conflicting CPU/GPU framework packages.

Prepare official models in `weights`, then run:

```bash
python -m ptoart doctor --model-dir ./weights --device cpu
python -m ptoart convert portrait.jpg output/avatar.png --model-dir ./weights --device cpu
python -m ptoart batch ./photos ./output-batch --recursive --model-dir ./weights --device cpu --report results.json
```

The installed `ptoart` command is equivalent to `python -m ptoart`. Model path priority is explicit `--weights` / `--segmentation`, then `--model-dir`, then `PTOART_MODEL_DIR`, then `~/.cache/ptoart/models/`.

## API, validation and licensing

See the [Chinese README](README.md) for Python API examples, legacy migration and development commands. New APIs use RGB/RGBA. Legacy `test.py` / `test_onnx.py` classes retain BGR outputs and legacy arguments, but no longer overwrite existing files by default.

Frozen smoke tests use random Torch weights and a synthetic TensorFlow segmentation graph, not official pretrained inference. Real-photo preprocessing and quality, first-use downloads, GPU, Torch/ONNX agreement, Cog deployment and training remain unvalidated end to end. [Earlier source validation notes](docs/VALIDATION.md) describe the original local refactor checks; remote Windows evidence is in the corresponding Actions logs.

The upstream MIT [license](LICENSE) and attribution are retained. The original English README is archived at [docs/UPSTREAM_README_EN.md](docs/UPSTREAM_README_EN.md). Model/data distribution and commercial rights require separate verification. Upstream training data coverage is narrow; refactoring does not remove that bias. Do not use this as an identity verification system.
