## 下载

| 平台 | 文件 | 说明 |
| --- | --- | --- |
| Windows 10/11 x64 | `ptoart-windows-x64-cpu.zip` | 便携命令行版，仅 CPU，解压后运行 `ptoart.exe`；把模型放进 EXE 旁的 `weights` 文件夹即可自动识别 |
| Android 7.0+ | `ptoart-android.apk` | 图形界面 App：导入 `photo2cartoon_weights.onnx` → 选照片 → 保存卡通头像，离线运行 |

`SHA256SUMS-all.txt` 可用于校验下载文件。

## 重要说明

- **模型权重不随包分发**，请按 [docs/MODELS.md](https://github.com/sincalaway/ptoart/blob/refactor/ptoart-20261023/docs/MODELS.md) 从上游取得。
- Windows 版需要 `photo2cartoon_weights.pt` + `seg_model_384.pb`；安卓版只需 `photo2cartoon_weights.onnx`（人脸检测与人像分割使用内置 ML Kit 模型）。
- 安卓 APK 使用调试签名，安装时需允许“未知来源”；EXE 未签名。
- 这是预览版：自动构建与自检已通过，真实照片 + 官方权重的出图效果未做完整验收。
