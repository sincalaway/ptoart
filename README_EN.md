# ptoart

A maintainable refactor of [photo2cartoon](https://github.com/minivision-ai/photo2cartoon), preserving its generator and training algorithms while adding a shared inference pipeline, Torch/ONNX adapters, batch processing, structured failures, background/alpha output, intensity blending, model diagnostics and tests.

See the complete [Chinese guide](README.md), [model setup](docs/MODELS.md), [refactor analysis](docs/REFACTOR.md) and [validation status](docs/VALIDATION.md).

```bash
pip install -e '.[torch,face]'
ptoart doctor --model-dir ./weights
ptoart convert portrait.jpg avatar.png --model-dir ./weights
ptoart batch ./photos ./output --recursive --model-dir ./weights --report results.json
```

This is an engineering preview, not a newly trained model or a production-readiness claim. Official pretrained weights are not bundled. Full face preprocessing, pretrained inference, GPU execution, Cog deployment and training have not been validated end to end. Output remains a 256×256 cropped portrait. `--strength` blends pixels rather than changing learned styles. `--aligned` is only for pre-aligned square portraits and does not perform segmentation.

New APIs use RGB/RGBA. Legacy `test.py` / `test_onnx.py` classes retain BGR outputs and legacy command parameters, but no longer overwrite existing files by default. Preserve the upstream model filenames for checkpoint compatibility.

The upstream MIT license and attribution are retained. The original English README is archived at [docs/UPSTREAM_README_EN.md](docs/UPSTREAM_README_EN.md). Model/data distribution and commercial rights require separate verification. Upstream training data coverage is narrow and refactoring does not remove that bias.
