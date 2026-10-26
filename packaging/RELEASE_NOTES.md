## 下载

**所有模型都已内置在 App 里**：Windows 只下载 `ptoart-windows-x64.zip`，安卓只下载 `ptoart-android.apk`，不需要再下载任何模型，安卓也不需要导入。

| 平台 | 文件 | 说明 |
| --- | --- | --- |
| Windows 10/11 x64 | `ptoart-windows-x64.zip`（约 1.05 GB） | 便携版，仅 CPU。**双击 `ptoart.exe` 打开图形界面**。全部模型已在 `weights` 文件夹里：卡通生成器和人像分割、全部风格与画风（含 DCT-Net）、Real-ESRGAN 超分、人脸检测 |
| Android 7.0+ | `ptoart-android.apk`（约 270 MB） | 图形界面 App：卡通头像、整张照片人脸卡通化、整图风格三种模式，离线运行。全部模型已内置，打开即用；「导入模型」只用于换成自己的模型 |

以下单独的模型文件**只给命令行/源码用户**，Windows 和安卓用户不需要下载：

| 内容 | 文件 | 许可 |
| --- | --- | --- |
| 头像生成器 | `ptoart_weights.pt`、`ptoart_weights.onnx` | MIT |
| 人像分割 | `seg_model_384.pb` | 上游资源包 |
| 风格模型 | `ptoart-styles.zip`（11 种 AnimeGANv2/v3 风格，含动漫脸、日漫二次元、吉卜力人像、黑白素描 4 种头像画风） | **仅限非商业用途** |
| DCT-Net 画风 | `ptoart-dctnet.zip`（DCT 日漫、DCT 3D、DCT 手绘，阿里达摩院 DCT-Net） | Apache-2.0 |
| 超分模型 | `ptoart-upscalers-realesrgan.zip`（Real-ESRGAN `anime`、`general` x4） | BSD-3 |

## 本版更新（v0.3.11）

**所有模型内置，下载 App 即可用**
- Windows 包的 `weights` 文件夹里已经有 `ptoart_weights.pt`、`seg_model_384.pb`、`ptoart_weights.onnx`、全部风格和画风（`styles\`，含 DCT-Net）、Real-ESRGAN 超分（`upscale\`）和人脸检测（`torch\`），各自的许可文件也在里面。首次启动直接识别，不用放文件、不用导入。
- 安卓 APK 内置 `ptoart_weights.onnx`、全部风格和 DCT-Net 画风、两个超分模型。首次打开，所有模式和画风就出现在下拉框里，不用导入；模型直接从 APK 读取，不额外占用手机存储。「超分辨率增强」下面可选动漫（快）或通用（照片）。「导入模型」保留为可选：导入同名模型会替换内置模型。

**Windows 界面修复**
- 「整张照片」+ DCT 画风之前会复用同一画风「卡通头像」的已加载模型，背景没有被卡通化；现在背景会正确地一起变成卡通。
- 结果标题会写明「透明背景」和所用的超分模型（如「超分：动漫（快）」）。
- 出图后切换超分模型，也会提示「设置已更改」。
- 修复连续出图时偶发的 TclError（图片清空方式错误）。
- 某个事件处理出错时，界面不再卡在「处理中…」。
- 降低内存占用：风格和超分模型不再保留 ONNX Runtime 的 CPU 内存池，多试几种风格也不会越来越占内存。

## 本版更新（v0.3.10）

- **DCT 画风直接内置进 Windows 包**：解压 `ptoart-windows-x64.zip` 后 `weights\styles\` 里已经有 DCT 日漫 / DCT 3D / DCT 手绘，打开就能选，不用再单独下载 `ptoart-dctnet.zip`（安卓仍需单独下载导入）。

## v0.3.9 更新

**新画风：DCT-Net（宝宝、儿童照片推荐）**。来自阿里巴巴达摩院开源的 [DCT-Net](https://github.com/menyifang/DCT-Net)（[ModelScope](https://modelscope.cn/models/damo/cv_unet_person-image-cartoon_compound-models)，Apache-2.0），纯 CPU、模型小（每个约 6 MB）。Windows 包已内置（安卓下载 `ptoart-dctnet.zip` 导入），在「头像画风」里出现：

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
| 超分放大：Real-ESRGAN x4 | 「超分辨率增强」开关 | `--upscale anime\|general` | 开关（动漫 / 通用） |
| 自动保存 / 撤销 / 重做 / 保存 / 另存为 | 每次自动存到 `ptoart输出`；顶部工具栏 + Ctrl+Z / Ctrl+Y / Ctrl+S | 指定输出路径 | 「保存 PNG」 |
| 批量处理 | 一次选多张 | `ptoart.exe batch 照片文件夹 输出文件夹` | — |
| 拖拽即用 | 把照片拖到包内 `.bat` | — | — |

小白教程：[QUICKSTART](https://github.com/sincalaway/ptoart/blob/refactor/ptoart-20261023/docs/QUICKSTART.md)

`SHA256SUMS-all.txt` 可用于校验下载文件。

## 重要说明

- 所有模型都已内置：Windows 包在 EXE 旁的 `weights\`（含 `styles\`、`upscale\`、`torch\`），安卓在 APK 内；都不需要联网或导入。安卓的人脸检测与人像分割使用内置 ML Kit 模型。
- 单独的模型文件（`.pt`、`.onnx`、`.pb` 和三个 zip）继续附在本 Release，供命令行和源码用户放进自己的 `weights` 目录。
- 许可：头像生成器 MIT（minivision-ai/photo2cartoon）；DCT 画风 Apache-2.0（阿里达摩院 DCT-Net，menyifang/DCT-Net，ModelScope；仅转换为 ONNX，权重未改动，zip 内附许可）；超分 BSD-3；AnimeGANv2/v3 仅限非商业用途。
- 安卓 APK 使用调试签名，安装时需允许“未知来源”；EXE 未签名。
- 这是预览版：自动构建与自检已通过，真实照片 + 官方权重的出图效果未做完整验收。
