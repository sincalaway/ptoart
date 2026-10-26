# ptoart Windows 版使用教程

适用 Windows 10/11 x64，仅 CPU，**无需安装 Python，也不需要命令行**：双击 `ptoart.exe` 就是图形界面，全程鼠标操作。

## 1. 下载（只要一个文件）

打开 [Release 下载页](https://github.com/sincalaway/ptoart/releases/latest)，在 Assets 里只下载 **`ptoart-windows-x64.zip`**（约 1 GB）。**所有模型都已内置在里面，不需要再下载任何模型。**

右键 ZIP →「全部解压缩」，例如解压到 `C:\ptoart`。**请保留整个 `ptoart` 文件夹**（`ptoart.exe` 必须和 `_internal` 在一起），不要在压缩包里直接运行。解压后像这样：

```text
ptoart\
├── ptoart.exe            ← 双击打开
├── _internal\
└── weights\              ← 所有模型，已内置，不要移动、改名或删除
    ├── ptoart_weights.pt、seg_model_384.pb、ptoart_weights.onnx   （卡通生成器、人像分割）
    ├── styles\hayao.onnx、jp_face.onnx、dct_3d.onnx …            （全部风格和画风，含 DCT-Net）
    ├── upscale\anime.onnx、general.onnx                           （超分）
    └── torch\                                                    （人脸检测）
```

`weights\README.txt` 是说明，列出每个文件夹的内容；各模型的许可文件也在对应文件夹里。Release 页上单独的模型文件（`ptoart_weights.pt`、`ptoart-styles.zip` 等）是给命令行/源码用户的，Windows 版用不到。

## 2. 使用（图形界面）

1. **双击 `ptoart.exe`**（首次启动约 10–30 秒）。Windows 提示“已保护你的电脑”时，点「更多信息 → 仍要运行」（程序未签名）。
2. 看左下角模型状态：全部是 ✓，不需要放任何文件。
3. 点「＋ 选择照片」，可一次多选。
4. 选效果：
   - **卡通头像**：最大一张脸的方形头像，可选尺寸 512 / 1024 / 2048、卡通强度、透明背景；
   - **整张照片**：所有人脸卡通化后贴回原图，身体和背景保持原样；
   - **整图风格**：整张图变成宫崎骏、新海诚等画风（风景、全身照都可以，不需要人脸）。
5. 头像和整张照片在「头像画风」列表里点一下即选中：选中项变成蓝底并带 ✓，窗口顶部常驻显示「✓ 头像画风：…」，切换时还会短暂提示「已选择…」。可选 **动漫脸（推荐）**、**DCT 日漫**、**DCT 3D**、**DCT 手绘**、**日漫二次元**、**吉卜力人像**、**黑白素描**（五官自然不变形，适合小朋友和长辈）和 **经典卡通**（线条更夸张）。**宝宝、儿童照片优先试 DCT 3D 或 DCT 日漫**；DCT 画风在「整张照片」模式下连背景一起卡通化。
6. 「超分辨率增强」默认打开，可选动漫（快）或通用，结果更清晰（稍慢）。
7. 点「开始转换 →」。每次的结果都会**自动保存**到照片旁的 `ptoart输出` 文件夹（窗口底部显示完整路径）。窗口**顶部工具栏**：**「↶ 撤销」Ctrl+Z**、**「↷ 重做」Ctrl+Y** 在几次结果间切换（没有可撤销/重做时变灰）；**「保存」Ctrl+S** 确认当前结果已保存并显示保存位置；**「另存为…」** 存到别处；**「打开输出文件夹」**。效果图上方写明用的是哪个模式、画风和强度；出图后再改画风或设置，会出现黄色提示「设置已更改」和 **「用新设置重新生成」** 按钮。

所有模型（包括 `weights\torch` 里的人脸检测模型）都已打包，**无需联网**；启动时后台加载一次（底部显示「模型已就绪」），之后切换画风、强度都不再重新加载。

**不想开窗口？** 把照片直接拖到包里的 `1-avatar.bat`（卡通头像）、`2-whole-photo.bat`（整张照片）或 `3-style-hayao.bat`（宫崎骏风格）上，结果在 `output` 文件夹。

## 3. 常见问题

- **双击没反应或闪退**：确认已完整解压、`_internal` 文件夹在 EXE 旁边，且不在压缩包里运行。
- **提示缺少模型**：模型都内置在 `weights` 文件夹里。请确认 ZIP 是完整解压的、`weights` 没被移动或改名；重新解压一次即可。
- **没检测到人脸**：换清晰、正面、光线好的照片；只想要动漫画风可以选「整图风格」。
- **很慢**：纯 CPU 运行，大图 + 超分可能十几秒到一分钟，属正常。
- **能商用吗**：卡通头像（MIT）、DCT 画风（阿里达摩院 [DCT-Net](https://github.com/menyifang/DCT-Net)，[ModelScope](https://modelscope.cn/models/damo/cv_unet_person-image-cartoon_compound-models)，Apache-2.0）和超分（Real-ESRGAN，BSD-3）可以；**动漫风格和画风（AnimeGANv2/v3）仅限非商业用途**。

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

加 `--upscale anime` 用 Real-ESRGAN 超分放大，例如 `--size 1024 --upscale anime`。`--output photo` 把每张人脸卡通化后贴回原图（`--max-faces` 默认 10）。`--style` 使用 `weights\styles\` 里已内置的风格模型：hayao（宫崎骏）、shinkai（新海诚）、paprika、celeba_distill、face_paint_512_v1、face_paint_512_v2、hayao_v3、shinkai_v3、jp_face（日漫二次元）、ghibli_face（吉卜力人像）、portrait_sketch（黑白素描），（AnimeGAN 风格仅限非商业用途）。头像/整张照片加 `--style jp_face` 等即换画风，例如 `ptoart.exe convert 宝宝.jpg 宝宝头像.png --style face_paint_512_v2 --size 1024`；也可以按画风名用 `--look`（`paint`、`dct_anime`、`dct_3d`、`dct_handdrawn`、`jp`、`ghibli`、`sketch`、`classic`），例如 `ptoart.exe convert 宝宝.jpg 宝宝3D.png --look dct_3d --size 1024`。

ONNX：`weights\ptoart_weights.onnx` 已内置，检查与转换都加 `--backend onnx`。普通照片仍需要 `seg_model_384.pb` 及人脸预处理：

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
- 首次下载失败：所有模型（含 `weights\torch` 人脸检测）都已内置，正常不需要联网；出现下载说明 `weights` 不完整，请重新解压。
- 无脸：换清晰、正面、单人照片，不要随意改用 `--aligned`。
- 输出/报告已存在：换名称；`--overwrite` 只覆盖图片，不覆盖报告。
- 提示 CUDA/dlib 缺失：本包仅 CPU + SFD，不要指定 `--device cuda` 或 `--detector dlib`。

支持 JPG/JPEG、PNG、WebP、BMP、TIF/TIFF，默认上限 2500 万像素；过大时可先缩图。更多 API、源码安装和限制见[仓库 README](https://github.com/sincalaway/ptoart#readme)。更新时请解压到新目录，不要混用不同构建的 `_internal`。


</details>
