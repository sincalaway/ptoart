## 下载

| 平台 | 文件 | 说明 |
| --- | --- | --- |
| Windows 10/11 x64 | `ptoart-windows-x64.zip` | 便携版，仅 CPU。**双击 `ptoart.exe` 打开图形界面**；也可命令行或拖拽 `.bat` 使用。把模型放进 EXE 旁的 `weights` 文件夹即可自动识别 |
| Android 7.0+ | `ptoart-android.apk` | 图形界面 App：卡通头像、整张照片人脸卡通化、整图风格三种模式，离线运行 |
| 头像生成器 | `ptoart_weights.pt` | Windows 默认后端（MIT） |
| 人像分割 | `seg_model_384.pb` | Windows 头像/整张照片模式必需 |
| 头像生成器 ONNX | `ptoart_weights.onnx` | 安卓必需；Windows `--backend onnx` 可选 |
| DCT-Net 画风 | `ptoart-dctnet.zip` | DCT 日漫、DCT 3D、DCT 手绘 3 种头像/整张照片画风（阿里达摩院 DCT-Net，Apache-2.0，可商用）。Windows 解压到 `weights\styles\`，安卓解压后多选导入 |
| 风格模型 | `ptoart-styles.zip` | 11 种 AnimeGANv2/v3 风格 ONNX，含动漫脸、日漫二次元、吉卜力人像、黑白素描 4 种头像画风（仅限非商业用途）。Windows 解压到 `weights\styles\`，安卓解压后多选导入 |
| 超分模型 | `ptoart-upscalers-realesrgan.zip` | Real-ESRGAN `anime`、`general` x4（BSD-3） |

## 本版更新（v0.3.9）

**新画风：DCT-Net（宝宝、儿童照片推荐）**。来自阿里巴巴达摩院开源的 [DCT-Net](https://github.com/menyifang/DCT-Net)（[ModelScope](https://modelscope.cn/models/damo/cv_unet_person-image-cartoon_compound-models)，Apache-2.0），纯 CPU、模型小（每个约 6 MB）。下载 `ptoart-dctnet.zip`，解压到 `weights\styles\` 后在「头像画风」里出现：

| 画风 | 模型文件 | 适合 |
| --- | --- | --- |
| DCT 日漫 | `dct_anime.onnx` + `dct_anime_bg.onnx` | 清爽日漫脸，宝宝、儿童不变形 |
| DCT 3D | `dct_3d.onnx` | 3D 动画电影质感，最像本人 |
| DCT 手绘 | `dct_handdrawn.onnx` + `dct_handdrawn_bg.onnx` | 手绘插画感 |

- 卡通头像：人脸对齐后用 DCT-Net 头部模型（288×288）推理。
- 整张照片：先用 DCT-Net 背景模型把整张图卡通化，再贴回 DCT 头部结果，和 DCT-Net 官方流程一致（安卓暂时只换脸、不换背景）。
- 整图风格里也多了「DCT 日漫 / DCT 3D / DCT 手绘」。
- 原有画风（动漫脸默认、日漫二次元、吉卜力人像、黑白素描、经典卡通）不变。

**Windows 界面**：
- 窗口顶部新增工具栏：**「↶ 撤销」Ctrl+Z、「↷ 重做」Ctrl+Y**（没有可撤销/重做时变灰）、**「保存」Ctrl+S**（确认当前结果已存盘并显示路径）、**「另存为…」**、**「打开输出文件夹」**；仍然每次自动保存。
- 风格和画风改成点选列表：选中项蓝底并带 ✓，顶部常驻「✓ 头像画风：… · 模式：…」，切换时短暂提示「已选择…」。
- 出图后再改设置，会出现黄色提示「设置已更改：当前效果是用…生成的」和「用新设置重新生成」按钮。
- 左侧面板可滚动，小屏幕也能看到全部选项。

**命令行**：新增 `--look paint|dct_anime|dct_3d|dct_handdrawn|jp|ghibli|sketch|classic`；`--output photo` 配合 DCT 画风会自动用背景模型（也可用 `--scene-style` 指定）。

## 全部功能一览

| 功能 | Windows 界面 | 命令行 | 安卓 |
| --- | --- | --- | --- |
| 卡通头像：裁出人脸生成头像，可透明背景 | 模式选「卡通头像」 | `ptoart.exe convert 照片.jpg 头像.png` | 模式「卡通头像 · 画风」 |
| 整张照片：所有人脸卡通化后贴回原图，原尺寸 | 模式选「整张照片」 | `--output photo`（`--max-faces` 默认 10） | 模式「整张照片 · 画风」 |
| 整图风格：整张图变动漫，风景、全身照都行，11 种 | 模式选「整图风格」+ 风格 | `--style hayao` 等 | 模式「整图风格 · 风格名」 |
| 头像画风：8 种（动漫脸、DCT 日漫 / 3D / 手绘、日漫二次元、吉卜力人像、黑白素描、经典卡通） | 「头像画风」点选列表，下方有说明 | `--look dct_3d` 等 | 模式下拉框 |
| 高清尺寸：512 / 1024 / 2048 | 「尺寸」 | `--size 1024` | 头像默认 512 |
| 卡通强度 0–100% | 「强度」滑块 | `--strength 0.8` | 滑块 |
| 超分放大：Real-ESRGAN x4 | 「超分辨率增强」开关 | `--upscale anime\|general` | 开关 |
| 自动保存 / 撤销 / 重做 / 保存 / 另存为 | 每次自动存到 `ptoart输出`；顶部工具栏 + Ctrl+Z / Ctrl+Y / Ctrl+S | 指定输出路径 | 「保存 PNG」 |
| 批量处理 | 一次选多张 | `ptoart.exe batch 照片文件夹 输出文件夹` | — |
| 拖拽即用 | 把照片拖到包内 `.bat` | — | — |

小白教程：[QUICKSTART](https://github.com/sincalaway/ptoart/blob/refactor/ptoart-20261023/docs/QUICKSTART.md)

`SHA256SUMS-all.txt` 可用于校验下载文件。

## 重要说明

- 头像、分割、风格、超分模型单独附在本 Release：Windows 把 `.pt`、`.pb` 放进 EXE 旁的 `weights\`；安卓导入 `.onnx`。Windows 包已内置人脸检测模型（`weights\torch`），无需联网。
- Windows 版需要 `ptoart_weights.pt` + `seg_model_384.pb`；安卓版只需 `ptoart_weights.onnx`（人脸检测与人像分割使用内置 ML Kit 模型）。
- 风格包（整图风格 + 4 种头像画风）：解压 `ptoart-styles.zip`，Windows 放到 `weights\styles\`；安卓在 App 中「导入模型」多选导入。`ptoart-dctnet.zip`（DCT 画风）同样处理。
- 许可：头像生成器 MIT（minivision-ai/photo2cartoon）；DCT 画风 Apache-2.0（阿里达摩院 DCT-Net，menyifang/DCT-Net，ModelScope；仅转换为 ONNX，权重未改动，zip 内附许可）；超分 BSD-3；AnimeGANv2/v3 仅限非商业用途。
- 安卓 APK 使用调试签名，安装时需允许“未知来源”；EXE 未签名。
- 这是预览版：自动构建与自检已通过，真实照片 + 官方权重的出图效果未做完整验收。
