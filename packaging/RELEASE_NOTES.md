## 下载

| 平台 | 文件 | 说明 |
| --- | --- | --- |
| Windows 10/11 x64 | `ptoart-windows-x64.zip` | 便携版，仅 CPU。**双击 `ptoart.exe` 打开图形界面**；也可命令行或拖拽 `.bat` 使用。把模型放进 EXE 旁的 `weights` 文件夹即可自动识别 |
| Android 7.0+ | `ptoart-android.apk` | 图形界面 App：卡通头像、整张照片人脸卡通化、整图风格三种模式，离线运行 |
| 头像生成器 | `ptoart_weights.pt` | Windows 默认后端（MIT） |
| 人像分割 | `seg_model_384.pb` | Windows 头像/整张照片模式必需 |
| 头像生成器 ONNX | `ptoart_weights.onnx` | 安卓必需；Windows `--backend onnx` 可选 |
| 风格模型 | `ptoart-styles.zip` | 宫崎骏、新海诚、Paprika、CelebA 蒸馏、Face Paint v1/v2 六种 AnimeGANv2 风格 ONNX（仅限非商业用途） |
| 超分模型 | `ptoart-upscalers-realesrgan.zip` | Real-ESRGAN `anime`、`general` x4（BSD-3） |

## 新功能（v0.3.0）

- `--output photo`：所有人脸卡通化后羽化贴回原图，原尺寸输出（安卓同步支持）。
- `--size`：高清头像，最高 2048（安卓头像默认 512）。
- `--style`：AnimeGANv2 整图风格，6 种风格可选，风景和全身照也能用。
- `--upscale anime|general`：Real-ESRGAN 超分放大，头像和整图更清晰（安卓有开关）。
- **Windows 图形界面**：双击 `ptoart.exe` 打开窗口，选模式、风格、尺寸、强度、超分，原图与效果并排预览，支持多张批量。
- 新增[小白使用教程](https://github.com/sincalaway/ptoart/blob/refactor/ptoart-20261023/docs/QUICKSTART.md)；Windows 包内附拖拽即用的 `.bat`。
- 所有模型直接附在 Release 中，无需再去网盘或自行转换。

`SHA256SUMS-all.txt` 可用于校验下载文件。

## 重要说明

- 模型不打进 EXE/APK，单独附在本 Release：Windows 把 `.pt`、`.pb` 放进 EXE 旁的 `weights\`；安卓导入 `.onnx`。
- Windows 版需要 `ptoart_weights.pt` + `seg_model_384.pb`；安卓版只需 `ptoart_weights.onnx`（人脸检测与人像分割使用内置 ML Kit 模型）。
- 整图风格：解压 `ptoart-styles.zip`，Windows 放到 `weights\styles\`；安卓在 App 中「导入模型」多选导入。
- 安卓 APK 使用调试签名，安装时需允许“未知来源”；EXE 未签名。
- 这是预览版：自动构建与自检已通过，真实照片 + 官方权重的出图效果未做完整验收。
