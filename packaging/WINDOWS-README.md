# ptoart Windows 版使用教程

适用 Windows 10/11 x64，仅 CPU，**无需安装 Python，也不需要命令行**：双击 `ptoart.exe` 就是图形界面，全程鼠标操作。

## 1. 下载

打开 [Release 下载页](https://github.com/sincalaway/ptoart/releases/latest)，在 Assets 里下载：

| 文件 | 是否必需 | 放到哪里 |
| --- | --- | --- |
| `ptoart-windows-x64.zip` | 必需 | 解压，例如到 `C:\ptoart` |
| `ptoart_weights.pt` | 必需（卡通头像、整张照片） | `weights\` |
| `seg_model_384.pb` | 必需（卡通头像、整张照片） | `weights\` |
| `ptoart-styles.zip` | 可选（动漫风格） | 解压到 `weights\styles\` |
| `ptoart-upscalers-realesrgan.zip` | 可选（更清晰） | 解压到 `weights\upscale\` |

右键 ZIP →「全部解压缩」。**请保留整个 `ptoart` 文件夹**（`ptoart.exe` 必须和 `_internal` 在一起），不要在压缩包里直接运行。放好后像这样：

```text
ptoart\
├── ptoart.exe            ← 双击打开
├── _internal\
└── weights\
    ├── ptoart_weights.pt
    ├── seg_model_384.pb
    ├── styles\hayao.onnx、shinkai.onnx …
    └── upscale\anime.onnx、general.onnx
```

`weights\README.txt` 只是说明，不是模型。

## 2. 使用（图形界面）

1. **双击 `ptoart.exe`**（首次启动约 10–30 秒）。Windows 提示“已保护你的电脑”时，点「更多信息 → 仍要运行」（程序未签名）。
2. 看左下角模型状态：✓ 表示已就绪。缺模型时点「打开模型文件夹」放入文件，再点「重新检测模型」。
3. 点「＋ 选择照片」，可一次多选。
4. 选效果：
   - **卡通头像**：最大一张脸的方形头像，可选尺寸 512 / 1024 / 2048、卡通强度、透明背景；
   - **整张照片**：所有人脸卡通化后贴回原图，身体和背景保持原样；
   - **动漫风格**：整张图变成宫崎骏、新海诚等画风（风景、全身照都可以，不需要人脸）。
5. 头像和整张照片可选「画风」：**动漫脸（推荐）**、**日漫二次元**、**吉卜力人像**、**黑白素描**五官自然不变形，适合小朋友和长辈；**经典卡通**线条更夸张。前四种来自风格包 `ptoart-styles.zip`。
6. 放了超分模型时，「超分辨率增强」开关会自动打开，结果更清晰（稍慢）。
7. 点「开始转换 →」，右侧对比原图和效果，点「保存结果」。多张照片会自动保存到照片旁的 `ptoart输出` 文件夹。

首次处理人脸请**保持联网**：程序会自动下载一次人脸检测数据（约几十 MB），之后可离线使用。

**不想开窗口？** 把照片直接拖到包里的 `1-avatar.bat`（卡通头像）、`2-whole-photo.bat`（整张照片）或 `3-style-hayao.bat`（宫崎骏风格）上，结果在 `output` 文件夹。

## 3. 常见问题

- **双击没反应或闪退**：确认已完整解压、`_internal` 文件夹在 EXE 旁边，且不在压缩包里运行。
- **提示缺少模型**：文件名必须一字不差（下载时别变成 `ptoart_weights (1).pt`），放在 `weights` 里，然后点「重新检测模型」。
- **没检测到人脸**：换清晰、正面、光线好的照片；只想要动漫画风可以选「动漫风格」。
- **很慢**：纯 CPU 运行，大图 + 超分可能十几秒到一分钟，属正常。
- **能商用吗**：卡通头像（MIT）和超分（Real-ESRGAN，BSD-3）可以；**动漫风格（AnimeGANv2）仅限非商业用途**。

<details>
<summary><b>进阶：命令行用法（可选，普通用户不需要）</b></summary>

### 3. 打开 PowerShell 并检查

以下假设文件夹为 `C:\ptoart`，请换成自己的实际位置；含空格的路径加双引号：

```powershell
cd "C:\ptoart"
.\ptoart.exe --version
.\ptoart.exe --help
.\ptoart.exe doctor --model-dir .\weights --device cpu
```

`doctor` 的 `dependencies`、`models` 各项和 `ready` 应为 `true`。缺项退出码为 2，用 `$LASTEXITCODE` 查看。它只检查模块及文件存在，不验证模型格式、实际加载或出图质量。

双击 EXE 后窗口关闭不代表程序有图形界面；请在 PowerShell 中运行，才能看到错误。相对路径按当前 PowerShell 目录解析。

### 4. 单图转换

```powershell
.\ptoart.exe convert .\portrait.jpg .\output\avatar.png --model-dir .\weights --device cpu
```

成功会打印结果路径，输出目录自动创建。结果为 **256×256 人脸裁剪头像**，不是整张照片的高清卡通化；多脸默认选择最大脸。建议清晰、正面、单人照片。

```powershell
# 透明背景（输出支持 PNG/WebP/TIFF，不支持 JPEG）
.\ptoart.exe convert .\portrait.jpg .\output\avatar-transparent.png --model-dir .\weights --device cpu --background transparent

# 浅蓝背景，70% 卡通结果与 30% 原始裁剪头像混合
.\ptoart.exe convert .\portrait.jpg .\output\avatar-soft.png --model-dir .\weights --device cpu --background DCEEFF --strength 0.7
```

强度范围 0–1，默认 1，只是后处理混合，不改变学习到的风格。透明边缘取决于分割模型。已有图片默认不覆盖；确需覆盖结果可加 `--overwrite`，但输入和输出永远不能是同一路径。

### 5. 批量转换

将图片放进 `photos`；`--recursive` 包含子目录：

```powershell
.\ptoart.exe batch .\photos .\output-batch --recursive --model-dir .\weights --device cpu --report .\batch-report.json
```

输出保留相对目录，如 `person.jpg` → `person.jpg.png`。模型初始化成功后，单图损坏或无脸会记录失败并继续。报告包含总数、成功数、失败数及逐图错误。输出目录不得等于或包含输入目录；嵌在输入目录内的输出目录会被跳过。

重复运行时换输出目录，或加 `--overwrite` 覆盖图片。**报告文件始终拒绝覆盖**，即使指定 `--overwrite`，仍要改报告名，如 `batch-report-2.json`。

退出码：**0** 命令成功；**1** 批次中有失败；**2** 参数、初始化、空输入等命令级错误，或 `doctor` 未就绪。

### 6. 可选模式

高清头像 / 整张照片 / 整图风格：

```powershell
.\ptoart.exe convert .\portrait.jpg .\output\avatar-1024.png --size 1024 --model-dir .\weights --device cpu
.\ptoart.exe convert .\group.jpg .\output\group.jpg --output photo --model-dir .\weights --device cpu
.\ptoart.exe convert .\street.jpg .\output\street-hayao.jpg --style hayao --model-dir .\weights
```

加 `--upscale anime` 用 Real-ESRGAN 超分放大，例如 `--size 1024 --upscale anime`。`--output photo` 把每张人脸卡通化后贴回原图（`--max-faces` 默认 10）。`--style` 需要把风格模型放到 `weights\styles\`：hayao（宫崎骏）、shinkai（新海诚）、paprika、celeba_distill、face_paint_512_v1、face_paint_512_v2，Release 中的 `ptoart-styles.zip` 已包含全部风格（仅限非商业用途）。

ONNX：准备 `ptoart_weights.onnx`，检查与转换都加 `--backend onnx`。普通照片仍需要 `seg_model_384.pb` 及人脸预处理：

```powershell
.\ptoart.exe doctor --backend onnx --model-dir .\weights --device cpu
.\ptoart.exe convert .\portrait.jpg .\output\avatar-onnx.png --backend onnx --model-dir .\weights --device cpu
```

已对齐头像：**仅用于已经对齐、紧裁切的正方形人像**，不要拿完整照片加此参数绕过错误：

```powershell
.\ptoart.exe doctor --aligned --model-dir .\weights --device cpu
.\ptoart.exe convert .\aligned-face.png .\output\avatar-aligned.png --aligned --model-dir .\weights --device cpu
```

`--aligned` 仍需生成器权重，但跳过检测/分割，不需要分割权重，也不会自动抠图；整个输入视为不透明。

### 遇到问题

- 找不到命令：确认完整解压两层 ZIP，进入 EXE 目录，并使用 `.\ptoart.exe`。
- 缺少模型：核对模型文件名与 `--model-dir`；文件存在仍可能格式错误，`doctor` 不校验内容。
- 首次下载失败：查看终端中的第三方下载错误并检查网络；仅复制生成器/分割模型不保证离线首启。
- 无脸：换清晰、正面、单人照片，不要随意改用 `--aligned`。
- 输出/报告已存在：换名称；`--overwrite` 只覆盖图片，不覆盖报告。
- 提示 CUDA/dlib 缺失：本包仅 CPU + SFD，不要指定 `--device cuda` 或 `--detector dlib`。

支持 JPG/JPEG、PNG、WebP、BMP、TIF/TIFF，默认上限 2500 万像素；过大时可先缩图。更多 API、源码安装和限制见[仓库 README](https://github.com/sincalaway/ptoart#readme)。更新时请解压到新目录，不要混用不同构建的 `_internal`。


</details>
