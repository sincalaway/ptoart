# ptoart

将人像照片转换为卡通头像的命令行工具，基于 [minivision-ai/photo2cartoon](https://github.com/minivision-ai/photo2cartoon) 重构。支持 Windows 便携 EXE、Python CLI 和 Python API，以及单图、批量、透明背景和强度混合。

[English](README_EN.md) · [Windows 便携版说明](packaging/WINDOWS-README.md) · [模型准备](docs/MODELS.md) · [模型下载（光鸭云盘）](https://www.guangyapan.com/s/1961237335383875651_alCCcsK8loJ3qRN9)

> **当前为 0.2.0 工程预览版，不是开箱即用的完整模型包。** Windows 自动构建和打包自检已通过，但**官方模型权重不随包提供，真实照片的完整检测 → 分割 → 卡通出图尚未验证**。请先准备可信权重再尝试；构建成功、`doctor` 通过或随机权重自检成功，都不等于真实出图质量已验收。

## 先了解能做什么

默认输出 **256×256 的人脸裁剪头像**，不是整张照片的高清卡通化。多张脸沿用上游的最大脸选择策略；建议先用清晰、正面、单人的照片尝试。重构保留原生成器和训练算法，不会自动增加新绘画风格，也未消除上游训练数据的人群偏差。

| 你想怎么用 | 从哪里开始 |
| --- | --- |
| Windows 上不安装 Python，直接运行 | 下方「Windows 使用教程」 |
| 从源码安装，或在 Linux/macOS 上尝试 | 「Python 源码安装」 |
| 在自己的 Python 程序里调用 | 「Python API」 |
| 安卓手机上点选照片生成头像 | 「安卓 App」 |

## 安卓 App

1. 在 [Releases](https://github.com/sincalaway/ptoart/releases) 下载 `ptoart-android.apk` 安装（Android 7.0+，需允许安装未知来源应用；APK 为调试签名）。
2. 从[光鸭云盘模型分享](https://www.guangyapan.com/s/1961237335383875651_alCCcsK8loJ3qRN9)下载 **`photo2cartoon_weights.onnx`** 保存到手机，在 App 中点「导入模型」选中它（模型不随 App 分发）。
3. 点「选择人像照片」，可调卡通强度、透明背景，结果为 256×256 头像，可「保存到相册」（Pictures/ptoart）。也可以在相册里把照片“分享”到本 App。

安卓版全程离线：人脸检测与人像分割使用打包进 APK 的 Google ML Kit 模型，卡通生成使用 ONNX Runtime。裁剪与对齐策略与桌面版近似但不完全一致，效果可能略有差异。

## Windows 使用教程

适用 Windows 10/11 x64，**仅 CPU、命令行界面**，无需另装 Python。不含 CUDA、dlib 或官方权重；不要把它当作双击打开的图形软件。

### 1. 下载并完整解压

1. 打开 [Releases 页面](https://github.com/sincalaway/ptoart/releases)，在最新版本的 Assets 里下载 **`ptoart-windows-x64-cpu.zip`**（无需登录，长期保留）。
2. 解压得到包含 `ptoart.exe` 和 `_internal` 的 `ptoart` 文件夹，放在方便的位置，例如 `C:\ptoart`。**不要只复制 EXE，也不要在 ZIP 内直接运行。**
3. 可用同一页面的 `SHA256SUMS-all.txt` 核对下载文件；包内 `BUILD.txt` 标明源码提交。EXE 未签名，请确认来自本仓库的 Release。

开发版构建仍可在 [Windows EXE 工作流](https://github.com/sincalaway/ptoart/actions/workflows/windows-exe.yml) 的 Artifacts 中下载（保留 30 天）。

### 2. 准备模型和照片

从[光鸭云盘模型分享](https://www.guangyapan.com/s/1961237335383875651_alCCcsK8loJ3qRN9)下载模型（其他来源见[模型准备说明](docs/MODELS.md)）。普通照片使用默认 Torch 后端时，需要以下两个真实模型文件，放入 EXE 旁的 `weights` 文件夹：

```text
C:\ptoart\
├── ptoart.exe
├── _internal\                        # 保留整个文件夹
├── weights\
│   ├── photo2cartoon_weights.pt       # 卡通生成器
│   └── seg_model_384.pb               # 人像分割模型
└── portrait.jpg                      # 你自己的测试照片
```

`weights` 中自带的 `README.txt` 只是提示，不是模型。不要把下载网页或随机测试权重重命名后当模型使用。普通推理不需要训练用的 `model_mobilefacenet.pth`。

**首次处理普通照片请保持联网。** Windows 包默认使用 SFD 检测器；face-alignment 仍可能下载人脸检测/关键点资产到用户缓存。仅放入上面两个文件不能保证离线首启，自动下载链路尚未完成实际验收。

### 3. 在 PowerShell 检查程序

打开 PowerShell，进入**实际存放 EXE 的目录**。以下用 `C:\ptoart` 为例，若位置不同请替换：

```powershell
cd "C:\ptoart"
.\ptoart.exe --version
.\ptoart.exe --help
.\ptoart.exe doctor --model-dir .\weights --device cpu
```

`doctor` 输出 JSON：`dependencies` 和 `models` 的各项应为 `true`，`ready` 应为 `true`。未就绪返回退出码 **2**，可紧接着运行 `$LASTEXITCODE` 查看。

`doctor` **只检查模块和文件存在**，不加载模型、不验证格式、CUDA 或图片效果。即使 `ready: true`，下一步实际加载仍可能失败。

> 从 0.2.0 起，EXE 会自动使用同目录下的 `weights` 文件夹（未指定 `--model-dir`/`--weights` 且未设置 `PTOART_MODEL_DIR` 时）。示例中保留 `--model-dir .\weights` 以便旧版本也能运行。相对路径基于当前 PowerShell 所在目录，含空格的路径请加双引号。

### 4. 转换第一张照片

把自己的照片保存为 `portrait.jpg`，然后运行：

```powershell
.\ptoart.exe convert .\portrait.jpg .\output\avatar.png --model-dir .\weights --device cpu
```

成功后会打印输出路径，结果位于 `output\avatar.png`，输出目录会自动创建。CPU 首次启动可能较慢；若失败，请保留终端错误文字并参考下方排错表。

常用变化可以分别运行：

```powershell
# 透明背景：输出请使用 PNG（也支持 WebP/TIFF）
.\ptoart.exe convert .\portrait.jpg .\output\avatar-transparent.png --model-dir .\weights --device cpu --background transparent

# 浅蓝背景，70% 卡通输出 + 30% 原始裁剪头像
.\ptoart.exe convert .\portrait.jpg .\output\avatar-soft.png --model-dir .\weights --device cpu --background DCEEFF --strength 0.7

# 仅当你希望替换已有结果时，加 --overwrite
.\ptoart.exe convert .\portrait.jpg .\output\avatar.png --model-dir .\weights --device cpu --overwrite
```

`--strength` 范围为 0–1，默认 1；它只是像素混合，不是模型风格控制。默认白色背景，透明度来自分割掩码，发丝边缘质量仍取决于分割模型。输入与输出不得是同一路径，即使加 `--overwrite` 也不能覆盖原照片。

### 5. 批量处理文件夹

把照片放进 `photos` 文件夹。下面命令会包含子目录，并生成逐图 JSON 报告：

```powershell
.\ptoart.exe batch .\photos .\output-batch --recursive --model-dir .\weights --device cpu --report .\batch-report.json
```

输出保留相对目录，`photos\person.jpg` 会变成 `output-batch\person.jpg.png`，避免与同名 PNG 冲突。批次顺序复用模型，单图损坏或无脸等错误会记录后继续；模型初始化失败则无法开始处理。

已有结果默认记为该图失败。重复执行时，可换输出目录，或明确加 `--overwrite` 覆盖图片；**报告文件永不覆盖**，即使有 `--overwrite` 也要换成新的名称（例如 `batch-report-2.json`）。输出目录不得等于或包含输入目录；位于输入目录内部的输出目录会被排除。

### 6. 可选：ONNX 或已对齐头像

Windows 包也包含 ONNX Runtime。准备 `photo2cartoon_weights.onnx` 后，用同一后端检查和转换；普通照片仍需要分割模型及人脸预处理，不是纯 ONNX 依赖链：

```powershell
.\ptoart.exe doctor --backend onnx --model-dir .\weights --device cpu
.\ptoart.exe convert .\portrait.jpg .\output\avatar-onnx.png --backend onnx --model-dir .\weights --device cpu
```

**只有已经对齐、紧裁切的正方形人像**才使用 `--aligned`：

```powershell
.\ptoart.exe doctor --aligned --model-dir .\weights --device cpu
.\ptoart.exe convert .\aligned-face.png .\output\avatar-aligned.png --aligned --model-dir .\weights --device cpu
```

此模式仍需要生成器权重，但跳过人脸检测和分割，不需要 `seg_model_384.pb`。它把整张输入视为不透明，**不会自动抠图**；不要为绕过普通照片的错误而随意加此参数，也不要把「正方形」误当成「已对齐」。

### 常见问题与退出码

| 现象 | 处理方法 |
| --- | --- |
| 双击后窗口闪退 | 这是命令行工具，请在 PowerShell 中运行，保留错误信息 |
| PowerShell 找不到命令 | 先 `cd` 到 EXE 所在目录，使用 `.\ptoart.exe`；检查是否仍在 ZIP 内或少解压了一层 |
| `doctor` 中模型为 `false` / 缺少模型 | 核对文件名、真实文件和 `--model-dir`；不要只放 `README.txt` |
| `doctor` 通过但模型加载失败 | 文件存在不代表格式或内容有效；确认可信权重和后端匹配，不要用改名的网页文件 |
| 首次下载失败 / 无法联网 | 检查终端给出的第三方下载错误及网络；SFD/关键点缓存需另行准备，不能保证离线首启 |
| `No face detected` | 换清晰、正面、单人照片；不是随意加 `--aligned` 的理由 |
| `Output already exists` / `Cannot write` | 换输出名、检查目录权限；明确希望覆盖图片时才加 `--overwrite` |
| `Report already exists` | 换 JSON 报告名；`--overwrite` 不会覆盖报告 |
| 透明输出失败或透明效果不明显 | 使用 PNG/WebP/TIFF；`--aligned` 不产生前景分割，透明像素不会凭空出现 |
| 提示 CUDA 或 dlib 缺失 | 此 Windows 包使用 CPU + SFD，不要指定 `--device cuda` 或 `--detector dlib` |
| 图片过大 / 格式不支持 | 支持 JPG、JPEG、PNG、WebP、BMP、TIF/TIFF；默认上限 2500 万像素，可先缩图或谨慎调整 `--max-pixels` |

退出码 **0**：命令成功（`doctor` 表示存在性检查通过）；**1**：批次中有图片失败；**2**：参数、模型/依赖初始化、空输入等命令级错误，或 `doctor` 未就绪。批处理请同时查看报告中的 `total`、`succeeded`、`failed` 和逐图错误。

## Python 源码安装

建议 Python 3.11 独立虚拟环境。基础包要求 Python 3.10+；CI 配置覆盖 3.10–3.12，不代表所有机器学习依赖组合均已验收。下面从**本仓库的当前默认分支**开始，不需要照搬上游旧版 Python/Torch 安装说明。

```bash
git clone --branch refactor/ptoart-20261023 https://github.com/sincalaway/ptoart.git
cd ptoart
python -m venv .venv
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

在已激活环境中安装：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[torch,face]"
```

源码 CLI 默认检测器为 **dlib**（与 Windows EXE 的 SFD 默认值不同）。dlib 可能需要 C++ 编译器和 CMake；如只想在 Windows 上使用，优先选择便携包。GPU 依赖需按平台选择，不要盲目安装冲突的 CPU/GPU 包。

只处理已对齐头像可安装 `.[torch]` 并显式加 `--aligned`。ONNX 改装 `.[onnx,face]`；完整预处理仍依赖 face-alignment/Torch 和 TensorFlow。

取得[模型权重](docs/MODELS.md)，放入自己创建的 `weights` 目录后：

```bash
python -m ptoart doctor --model-dir ./weights --device cpu
python -m ptoart convert ./images/photo_test.jpg ./output/avatar.png --model-dir ./weights --device cpu
python -m ptoart batch ./photos ./output-batch --recursive --model-dir ./weights --device cpu --report ./batch-report.json
```

也可以将 `python -m ptoart` 换成安装后的 `ptoart` 命令。背景、强度、ONNX、对齐模式和防覆盖规则与上述 EXE 示例相同；子命令参数放在 `convert` / `batch` / `doctor` 后，可用 `python -m ptoart convert --help` 查看。

### 模型路径规则

不指定目录时默认使用 `~/.cache/ptoart/models/`；也可设置环境变量 `PTOART_MODEL_DIR`。优先级：单文件 `--weights` / `--segmentation` > `--model-dir` > 环境变量 > 默认目录。相对路径基于当前工作目录。

```bash
ptoart convert portrait.jpg avatar.png --weights ./weights/photo2cartoon_weights.pt --segmentation ./weights/seg_model_384.pb --device cpu
```

读取支持 EXIF 方向、灰度和透明图片；输入透明区域先合成为白色。透明输出的 alpha 来自人像分割，不是保留输入文件原有透明度。

## Python API

```python
from ptoart import ArtConfig, ArtPipeline

config = ArtConfig(model_dir="./weights", device="cpu", strength=0.85)
with ArtPipeline(config) as pipeline:
    pipeline.convert_file("portrait.jpg", "avatar.png")
    # 重复调用复用已加载的模型
    pipeline.convert_file("another.jpg", "another.png")
```

`convert()` 接收 `uint8 H×W×3 RGB`，返回 `uint8 256×256×3 RGB`；透明背景（`background=None`）返回 RGBA。通过 `NoFaceError` / `ImageError` / `ModelError` 区分失败。一个管线用于一个顺序工作者，不要跨线程共享。上下文管理器或 `close()` 释放会话和网络引用，不保证底层框架立即归还所有缓存显存。

## 自动构建与更新

[Windows EXE 工作流](.github/workflows/windows-exe.yml)在分支 `refactor/ptoart-20261023` 推送、`v*` 标签推送或手动运行时构建。需要重新生成时，可在 **Actions → Windows EXE → Run workflow** 选择该分支；有相应仓库权限才会看到操作入口。若未来更改默认分支，需同步更新工作流中的分支过滤条件。

更新便携包时，建议解压到新目录，核对 `BUILD.txt` 后复制自己的权重和照片，不要将不同构建的 `_internal` 文件混用。便携版教程随包保存为 `README-WINDOWS.md`。

## 重构、兼容与开发

本次重构统一了 Torch/ONNX 的 `ArtPipeline`，增加结构化错误、批量逐项结果、模型存在性诊断、RGB/RGBA 输出和回归测试。具体设计见[重构说明](docs/REFACTOR.md)。

旧命令保留，权重位置沿用原项目（生成器在 `models/`，分割模型在 `utils/`）：

```bash
python test.py --photo_path ./images/photo_test.jpg --save_path ./output/avatar.png
python test_onnx.py --photo_path ./images/photo_test.jpg --save_path ./output/avatar.png
```

旧 `Photo2Cartoon.inference(RGB)` 返回 BGR，新 `ArtPipeline` 返回 RGB。旧入口不再默认覆盖目标；需要覆盖时使用新版 CLI 的 `--overwrite`。保留旧模型文件名以兼容官方 checkpoint。

训练代码在 `train.py` / `models/`，另需 `.[train]` 和 MobileFaceNet 权重，仍需 GPU，未完成现代依赖下的训练验收。数据可放在 `dataset/ptoart/{trainA,trainB,testA,testB}` 后运行 `python train.py --dataset ptoart`。`predict.py` / `cog.yaml` 是可选 Cog 部署适配，容器构建和线上部署尚未验证。

开发检查：

```bash
python -m pip install -e ".[dev]"
ruff check ptoart tests test.py test_onnx.py data_process.py predict.py
pytest -q
python -m build
```

未安装对应框架时部分冒烟测试会跳过，请同时查看 passed 和 skipped。Windows 工作流另做冻结 EXE 的版本/帮助、缺模型退出码和 `--self-test` 检查；自检使用随机 Torch 权重与合成 TensorFlow 分割图，不能评价卡通效果。[早期源码验证记录](docs/VALIDATION.md)记录的是当时的本地重构验收，远程 Windows 构建以 Actions 对应运行日志为准。

官方权重端到端推理、Torch/ONNX 视觉与数值一致性、首次资产下载、GPU、持续运行性能和训练仍需实际验证。

## 来源与许可

保留上游 MIT [LICENSE](LICENSE)，上游基线为 `6204e27a71620500c2f5b2ffd36646b0e2e5c110`。原中英文说明存档于 `docs/UPSTREAM_README_*.md`，其中相对链接是上游历史记录，当前使用以本文为准。

模型和数据的下载、再分发及商用权利需单独核实，不能仅凭代码的 MIT 许可推断权重及数据也已获授权。上游说明训练数据以亚洲年轻女性为主，其他人群覆盖不足；重构未消除该偏差，不应用于身份核验。
