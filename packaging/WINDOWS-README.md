# ptoart Windows 便携版

适用 Windows 10/11 x64，仅 CPU，无需自行安装 Python。它是**命令行工具，不是图形界面软件**。依赖较大，采用文件夹便携包而不是单文件：请保留 `ptoart.exe` 和 `_internal` 在同一目录，不要只复制 exe。

## 从 GitHub 获取

进入仓库的 **Actions → Windows EXE → 成功的运行 → Artifacts → ptoart-windows-x64-cpu**。先解压下载的 artifact，再解压其中的 `ptoart-windows-x64-cpu.zip`。内附 SHA256SUMS.txt 可检查内层 ZIP。构建产物保留 30 天；可在 Actions 中通过 Run workflow 再次生成。

默认分支每次推送、`v*` 标签推送，以及手动运行均会构建。`BUILD.txt` 记录源代码提交，`build-dependencies.txt` 记录构建时依赖。exe 未做代码签名，Windows 可能显示未知发布者提示；请仅使用自己仓库中可信运行的产物。

## 准备权重

权重不包含在包内，也不会在构建过程中下载。请参照仓库的 [docs/MODELS.md](https://github.com/sincalaway/ptoart/blob/refactor/ptoart-20261023/docs/MODELS.md)，从上游入口取得可信模型，放入 exe 同目录的 `weights`：

```text
ptoart/
├── ptoart.exe
├── _internal/
└── weights/
    ├── photo2cartoon_weights.pt
    └── seg_model_384.pb
```

本包使用 **SFD** 作为默认人脸检测器，不打包需要 C++ 编译的 dlib。首次处理普通照片时，face-alignment 仍可能联网下载 SFD/关键点检测权重到用户缓存目录；首次运行应保持联网，不保证离线首启。SFD 行为与原默认 dlib 不完全相同。CPU 包不支持 CUDA。

## PowerShell 使用示例

在解压后的 ptoart 目录打开 PowerShell：

```powershell
.\ptoart.exe --version
.\ptoart.exe --help
.\ptoart.exe doctor --model-dir .\weights --device cpu
.\ptoart.exe convert .\portrait.jpg .\avatar.png --model-dir .\weights --device cpu
.\ptoart.exe batch .\photos .\output --model-dir .\weights --device cpu --report .\report.json
```

ONNX 后端同样包含在包内；使用对应的 `photo2cartoon_weights.onnx`，并加 `--backend onnx`。已经裁剪、对齐的正方形头像可加 `--aligned`，这会跳过人脸检测和分割，不需要分割权重，也不会自动抠图。

双击 exe 后窗口立即关闭不代表有图形界面：请从 PowerShell 中按上述命令使用。模型检查缺项返回退出码 2，是预期行为。

## 已验证与未验证

GitHub 工作流会在 Windows 上运行回归测试，再从源码目录以外执行冻结后的 exe，检查帮助、版本、缺少模型的退出码，以及 `--self-test`。该自检实际加载打包后的 Torch / TensorFlow / ONNX Runtime / 人脸处理模块，用随机权重运行一次对齐头像推理，并用合成 TensorFlow 图测试分割接口。

**这些检查不代表官方模型真实出图已通过验收。** 官方权重、真实照片完整检测/分割链路、首次自动下载以及图像质量仍需要实际验证。请勿将随机权重自检生成的结果用于判断卡通效果。
