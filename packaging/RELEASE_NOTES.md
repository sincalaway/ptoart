## 下载

| 平台 | 文件 | 说明 |
| --- | --- | --- |
| Windows 10/11 x64 | `ptoart-windows-x64.zip` | 便携版，仅 CPU。**双击 `ptoart.exe` 打开图形界面**；也可命令行或拖拽 `.bat` 使用。把模型放进 EXE 旁的 `weights` 文件夹即可自动识别 |
| Android 7.0+ | `ptoart-android.apk` | 图形界面 App：卡通头像、整张照片人脸卡通化、整图风格三种模式，离线运行 |
| 头像生成器 | `ptoart_weights.pt` | Windows 默认后端（MIT） |
| 人像分割 | `seg_model_384.pb` | Windows 头像/整张照片模式必需 |
| 头像生成器 ONNX | `ptoart_weights.onnx` | 安卓必需；Windows `--backend onnx` 可选 |
| 风格模型 | `ptoart-styles.zip` | 11 种 AnimeGANv2/v3 风格 ONNX，含动漫脸、日漫二次元、吉卜力人像、黑白素描 4 种头像画风（仅限非商业用途）。Windows 解压到 `weights\styles\`，安卓解压后多选导入 |
| 超分模型 | `ptoart-upscalers-realesrgan.zip` | Real-ESRGAN `anime`、`general` x4（BSD-3） |

## 本版更新（v0.3.8）

**画风（头像 / 整张照片）**：新增 5 种，可在 Windows「画风」下拉框或安卓模式下拉框里选择：

| 画风 | 模型文件 | 适合 |
| --- | --- | --- |
| 动漫脸（推荐，默认） | `face_paint_512_v2.onnx` | 五官自然不变形，儿童、老人都好看 |
| 日漫二次元 | `jp_face.onnx` | 大眼睛二次元脸 |
| 吉卜力人像 | `ghibli_face.onnx` | 柔和手绘油彩 |
| 黑白素描 | `portrait_sketch.onnx` | 黑白线稿 |
| 经典卡通 | `ptoart_weights.pt/.onnx` | 原版线条，儿童照片易变形 |

**整图风格**：新增 宫崎骏 v3、新海诚 v3，共 11 种。

**Windows 界面**：
- 人脸检测模型已打包（`weights\torch`），首次使用不再联网下载，不会卡在「正在加载」。
- 启动时后台加载一次模型（底部显示「模型已就绪」）；切换画风、尺寸、强度不再重新加载。
- 每次结果**自动保存**到照片旁的 `ptoart输出` 文件夹，底部显示完整路径；按钮：「↶ 撤销」「↷ 重做」「另存为…」「打开输出文件夹」。
- 效果图上方标明用的模式、画风、尺寸和强度；改了设置未转换时按钮变成「应用新设置」。
- 「动漫风格」改名为「整图风格」。

**命令行**：`--style jp_face|ghibli_face|portrait_sketch|face_paint_512_v2` 配合 `--output face/photo` 换画风。

## 全部功能一览

| 功能 | Windows 界面 | 命令行 | 安卓 |
| --- | --- | --- | --- |
| 卡通头像：裁出人脸生成头像，可透明背景 | 模式选「卡通头像」 | `ptoart.exe convert 照片.jpg 头像.png` | 模式「卡通头像 · 画风」 |
| 整张照片：所有人脸卡通化后贴回原图，原尺寸 | 模式选「整张照片」 | `--output photo`（`--max-faces` 默认 10） | 模式「整张照片 · 画风」 |
| 整图风格：整张图变动漫，风景、全身照都行，11 种 | 模式选「整图风格」+ 风格 | `--style hayao` 等 | 模式「整图风格 · 风格名」 |
| 头像画风：5 种（见上表） | 「画风」下拉框，下方有说明 | `--style jp_face` 等 | 模式下拉框 |
| 高清尺寸：512 / 1024 / 2048 | 「尺寸」 | `--size 1024` | 头像默认 512 |
| 卡通强度 0–100% | 「强度」滑块 | `--strength 0.8` | 滑块 |
| 超分放大：Real-ESRGAN x4 | 「超分辨率增强」开关 | `--upscale anime\|general` | 开关 |
| 自动保存 / 撤销 / 重做 / 另存为 | 每次自动存到 `ptoart输出` | 指定输出路径 | 「保存 PNG」 |
| 批量处理 | 一次选多张 | `ptoart.exe batch 照片文件夹 输出文件夹` | — |
| 拖拽即用 | 把照片拖到包内 `.bat` | — | — |

小白教程：[QUICKSTART](https://github.com/sincalaway/ptoart/blob/refactor/ptoart-20261023/docs/QUICKSTART.md)

`SHA256SUMS-all.txt` 可用于校验下载文件。

## 重要说明

- 头像、分割、风格、超分模型单独附在本 Release：Windows 把 `.pt`、`.pb` 放进 EXE 旁的 `weights\`；安卓导入 `.onnx`。Windows 包已内置人脸检测模型（`weights\torch`），无需联网。
- Windows 版需要 `ptoart_weights.pt` + `seg_model_384.pb`；安卓版只需 `ptoart_weights.onnx`（人脸检测与人像分割使用内置 ML Kit 模型）。
- 风格包（整图风格 + 4 种头像画风）：解压 `ptoart-styles.zip`，Windows 放到 `weights\styles\`；安卓在 App 中「导入模型」多选导入。
- 安卓 APK 使用调试签名，安装时需允许“未知来源”；EXE 未签名。
- 这是预览版：自动构建与自检已通过，真实照片 + 官方权重的出图效果未做完整验收。
