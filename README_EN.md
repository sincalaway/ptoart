# ptoart

A maintainable refactor of [photo2cartoon](https://github.com/minivision-ai/photo2cartoon), preserving its generator and training algorithms while adding a shared Torch/ONNX pipeline, batch processing, structured errors, background/alpha output, intensity blending and diagnostics.

[Complete Chinese tutorial](README.md) · [Windows package guide (Chinese)](packaging/WINDOWS-README.md) · [Models](docs/MODELS.md) · [Downloads (Release)](https://github.com/sincalaway/ptoart/releases/latest)

> Version 0.3.11: **every model is built in.** The Windows ZIP and the Android APK work right after download, with no extra model downloads, and the Android app needs no import. The Windows EXE opens a GUI. Looks include Alibaba DAMO Academy's DCT-Net (DCT Anime / DCT 3D / DCT Hand-drawn), which keep baby and child faces undistorted.

## Windows portable quick start

For Windows 10/11 x64, CPU only; no Python and no command line needed. **Double-click `ptoart.exe` to open the GUI.**

1. From the [latest Release](https://github.com/sincalaway/ptoart/releases/latest) download `ptoart-windows-x64.zip` and extract it (keep `ptoart.exe` together with `_internal`).
2. Nothing else to download: every model is already in the `weights` folder beside the EXE (generator and segmentation in `weights\`, all styles and the DCT-Net looks in `weights\styles\`, Real-ESRGAN in `weights\upscale\`, the face detector in `weights\torch\`). Do not move or rename them. Licences: cartoon generator MIT (from minivision-ai/photo2cartoon), DCT-Net looks Apache-2.0 ([menyifang/DCT-Net](https://github.com/menyifang/DCT-Net), [ModelScope](https://modelscope.cn/models/damo/cv_unet_person-image-cartoon_compound-models)), Real-ESRGAN BSD-3, AnimeGANv2/v3 non-commercial only. Licence files sit next to the models. Nothing is downloaded at first run.
3. Double-click `ptoart.exe`, pick photos (multi-select works), choose Avatar / Whole photo / Whole-image style, pick a face look from the list (Anime face (recommended), DCT Anime, DCT 3D, DCT Hand-drawn, JP anime, Ghibli portrait, Pencil sketch, Classic; the selected one turns blue with a ✓ and the top bar reads "✓ 头像画风：…"); for baby and child photos DCT 3D or DCT Anime are recommended. Adjust size, strength, transparency and super-resolution, then Convert. Every result is auto-saved to a `ptoart输出` folder next to the photo. The toolbar at the top has Undo (Ctrl+Z), Redo (Ctrl+Y) — greyed out when there is nothing to undo/redo — Save (Ctrl+S, confirms the current result is on disk and shows where), Save as… and Open output folder. The result title states the mode, look and strength used; changing a setting after a result shows a yellow "settings changed" banner with a "re-run with new settings" button. Models load once in the background at startup.

Command-line use is still available for advanced users (`ptoart.exe --help`); see the Chinese README.

The Windows package defaults to **SFD**, unlike the source CLI's dlib default; its detector and landmark weights are bundled in `weights\torch`.

Development builds from Actions carry only the face detector (copy the other models from the Release into `weights`); normal users should take the Release ZIP. Artifacts are retained for 30 days. A known successful historical build is [run 39937315949](https://github.com/sincalaway/ptoart/actions/runs/39937315949), source `97a3d9f`; use newer successful runs when available. The workflow runs on pushes to `refactor/ptoart-20261023`, `v*` tags and manual dispatch. Update the branch filter if the default branch changes. `BUILD.txt` identifies the source commit; `SHA256SUMS.txt` checks the inner ZIP. The executable is unsigned: use trusted builds. Extract updates into a new folder rather than mixing `_internal` files.

## Android

Install `ptoart-android.apk` (about 270 MB, Android 7.0+) from the [latest Release](https://github.com/sincalaway/ptoart/releases/latest). Every model is bundled in the APK: open the app, pick a photo, choose a mode and look, and convert; no download and no import. The Import button is optional and only replaces a bundled model with your own `.onnx` file of the same name.

## Separate model assets (CLI and source users)

The Release still carries every model as a separate asset (`ptoart_weights.pt`, `ptoart_weights.onnx`, `seg_model_384.pb`, `ptoart-styles.zip`, `ptoart-dctnet.zip`, `ptoart-upscalers-realesrgan.zip`) for command-line and source installs. Windows and Android users do not need them.

## Options and troubleshooting

- `doctor` checks module/file presence only, not model loading, validity, CUDA or inference quality. `ready: true` returns 0; missing items return 2.
- `--output face` (default) returns the largest face as a square avatar; `--size` (64–2048, default 256) upscales the 256px generator output with Lanczos + light sharpening while alpha and strength blending use the full-resolution crop.
- `--output photo` keeps the original photo size and blends every detected face (`--max-faces`, default 10) back in place with feathered edges; bodies and backgrounds stay photographic.
- `--style NAME` (implies `--output style`) applies a whole-image style from `<model-dir>/styles/NAME.onnx`: `hayao`, `shinkai`, `paprika`, `celeba_distill`, `face_paint_512_v1`, `face_paint_512_v2` (AnimeGANv2) and `hayao_v3`, `shinkai_v3`, `jp_face`, `ghibli_face`, `portrait_sketch` (AnimeGANv3), all in `ptoart-styles.zip`. With `--output face/photo`, `face_paint_512_v2`, `jp_face`, `ghibli_face` or `portrait_sketch` paint the faces instead of the classic generator.
- `--look paint|dct_anime|dct_3d|dct_handdrawn|jp|ghibli|sketch|classic` picks a face look for `--output face/photo` by name. The `dct_*` looks use DCT-Net (`ptoart-dctnet.zip`, Apache-2.0): avatar mode runs the DCT-Net head model on the aligned 288×288 face crop; `--output photo` first stylizes the whole picture with the DCT-Net background model (`--scene-style`, set automatically) and then pastes the DCT-Net faces with ptoart's segmentation mask, mirroring upstream DCT-Net. Convert them yourself with `tools/export_dctnet.py` (TensorFlow 2.15 + tf2onnx). Create them with `tools/export_styles.py` and `tools/export_styles_v3.py`. AnimeGAN weights are for **non-commercial use only**. `--style-max-side` (default 1536) sets the inference resolution.
- `--upscale anime|general` uses a Real-ESRGAN compact x4 model from `<model-dir>/upscale/` instead of plain interpolation whenever output is enlarged (avatar `--size` > 256, large pasted faces, downscaled style inference).
- `--background transparent` requires PNG, WebP or TIFF output. Default background is white. Segmentation quality determines alpha edges.
- `--strength 0.7` blends 70% model output with 30% original cropped portrait; it is not learned style control. Range: 0–1, default 1.
- `--backend onnx` requires `ptoart_weights.onnx`. Normal-photo preprocessing still needs segmentation, face-alignment/Torch and TensorFlow; this is not an ONNX-only pipeline.
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

Download the separate model assets from the Release into `weights` (unzip the styles, DCT-Net and upscaler zips into `weights/styles` and `weights/upscale`), then run:

```bash
python -m ptoart doctor --model-dir ./weights --device cpu
python -m ptoart convert portrait.jpg output/avatar.png --model-dir ./weights --device cpu
python -m ptoart batch ./photos ./output-batch --recursive --model-dir ./weights --device cpu --report results.json
```

The installed `ptoart` command is equivalent to `python -m ptoart`. Model path priority is explicit `--weights` / `--segmentation`, then `--model-dir`, then `PTOART_MODEL_DIR`, then `~/.cache/ptoart/models/`.

## API, validation and licensing

See the [Chinese README](README.md) for Python API examples, legacy migration and development commands. New APIs use RGB/RGBA. Legacy `test.py` / `test_onnx.py` classes retain BGR outputs and legacy arguments, but no longer overwrite existing files by default.

Frozen smoke tests use random Torch weights and a synthetic TensorFlow segmentation graph, not official pretrained inference. Real-photo preprocessing and quality, first-use downloads, GPU, Torch/ONNX agreement, Cog deployment and training remain unvalidated end to end. [Earlier source validation notes](docs/VALIDATION.md) describe the original local refactor checks; remote Windows evidence is in the corresponding Actions logs.

DCT looks use [DCT-Net](https://github.com/menyifang/DCT-Net) by Alibaba DAMO Academy (Men Yifang et al., SIGGRAPH 2022), anime / 3d / handdrawn weights from [ModelScope](https://modelscope.cn/models/damo/cv_unet_person-image-cartoon_compound-models), Apache License 2.0; ptoart only converts the TensorFlow frozen graphs to ONNX (weights unchanged) and ships the licence text inside `ptoart-dctnet.zip`.

The upstream MIT [license](LICENSE) and attribution are retained. The original English README is archived at [docs/UPSTREAM_README_EN.md](docs/UPSTREAM_README_EN.md). Model/data distribution and commercial rights require separate verification. Upstream training data coverage is narrow; refactoring does not remove that bias. Do not use this as an identity verification system.
