# ptoart Windows 便携版使用教程

适用 Windows 10/11 x64，**仅 CPU，无需安装 Python**。这是命令行工具，不是图形界面。请保留整个文件夹，让 `ptoart.exe` 和 `_internal` 始终在一起，不要只复制 EXE。

> **官方模型权重不随包提供，真实照片的完整检测、分割和卡通出图尚未验证。** 自动构建及随机权重自检通过，不代表真实模型效果已经验收。

## 1. 下载并解压两层 ZIP

打开 [Windows EXE 构建列表](https://github.com/sincalaway/ptoart/actions/workflows/windows-exe.yml)，选择分支 `refactor/ptoart-20261023` 上成功的运行，在底部 **Artifacts** 下载 **`ptoart-windows-x64-cpu`**。网页下载通常需要登录 GitHub。

先解压下载的 artifact，再解压里面的 `ptoart-windows-x64-cpu.zip`，直到看到 `ptoart/ptoart.exe` 和 `ptoart/_internal`。将整个 `ptoart` 文件夹移动到例如 `C:\ptoart`，不要在压缩包内运行。

产物保留 30 天。过期后选择新的成功运行，或在有仓库权限时使用 **Actions → Windows EXE → Run workflow** 重新构建。工作流也在上述分支推送及 `v*` 标签推送时触发；未来改默认分支需同步修改工作流过滤条件。

`SHA256SUMS.txt` 用于核对内层 ZIP；`BUILD.txt` 标记源码提交，`build-dependencies.txt` 记录构建依赖。EXE 未签名，请只运行可信仓库及运行中的产物，不要为来源不明的程序关闭系统防护。

## 2. 准备权重和照片

权重不在包内，也不会在构建时下载。按[模型准备说明](https://github.com/sincalaway/ptoart/blob/refactor/ptoart-20261023/docs/MODELS.md)取得可信模型，推荐从[光鸭云盘模型分享](https://www.guangyapan.com/s/1961237335383875651_alCCcsK8loJ3qRN9)下载。包内也附有 `MODELS.md`。

默认 Torch 后端处理普通照片时，将这两个模型放入 `weights`，再准备自己的测试照片：

```text
C:\ptoart\
├── ptoart.exe
├── _internal\
├── weights\
│   ├── photo2cartoon_weights.pt
│   └── seg_model_384.pb
└── portrait.jpg
```

`weights/README.txt` 不是权重。普通推理不需要训练用的 `model_mobilefacenet.pth`。所有示例都显式指定 `--model-dir .\weights`，程序不会自动选择 EXE 旁的 `weights`。

本包默认用 **SFD** 检测器，不包含 dlib 或 CUDA。首次处理普通照片时，face-alignment 仍可能联网下载 SFD/关键点资产到用户缓存；请保持联网，不保证离线首启。该下载链路和 SFD 下真实照片效果尚未实际验收。

## 3. 打开 PowerShell 并检查

以下假设文件夹为 `C:\ptoart`，请换成自己的实际位置；含空格的路径加双引号：

```powershell
cd "C:\ptoart"
.\ptoart.exe --version
.\ptoart.exe --help
.\ptoart.exe doctor --model-dir .\weights --device cpu
```

`doctor` 的 `dependencies`、`models` 各项和 `ready` 应为 `true`。缺项退出码为 2，用 `$LASTEXITCODE` 查看。它只检查模块及文件存在，不验证模型格式、实际加载或出图质量。

双击 EXE 后窗口关闭不代表程序有图形界面；请在 PowerShell 中运行，才能看到错误。相对路径按当前 PowerShell 目录解析。

## 4. 单图转换

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

## 5. 批量转换

将图片放进 `photos`；`--recursive` 包含子目录：

```powershell
.\ptoart.exe batch .\photos .\output-batch --recursive --model-dir .\weights --device cpu --report .\batch-report.json
```

输出保留相对目录，如 `person.jpg` → `person.jpg.png`。模型初始化成功后，单图损坏或无脸会记录失败并继续。报告包含总数、成功数、失败数及逐图错误。输出目录不得等于或包含输入目录；嵌在输入目录内的输出目录会被跳过。

重复运行时换输出目录，或加 `--overwrite` 覆盖图片。**报告文件始终拒绝覆盖**，即使指定 `--overwrite`，仍要改报告名，如 `batch-report-2.json`。

退出码：**0** 命令成功；**1** 批次中有失败；**2** 参数、初始化、空输入等命令级错误，或 `doctor` 未就绪。

## 6. 可选模式

ONNX：准备 `photo2cartoon_weights.onnx`，检查与转换都加 `--backend onnx`。普通照片仍需要 `seg_model_384.pb` 及人脸预处理：

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

## 遇到问题

- 找不到命令：确认完整解压两层 ZIP，进入 EXE 目录，并使用 `.\ptoart.exe`。
- 缺少模型：核对模型文件名与 `--model-dir`；文件存在仍可能格式错误，`doctor` 不校验内容。
- 首次下载失败：查看终端中的第三方下载错误并检查网络；仅复制生成器/分割模型不保证离线首启。
- 无脸：换清晰、正面、单人照片，不要随意改用 `--aligned`。
- 输出/报告已存在：换名称；`--overwrite` 只覆盖图片，不覆盖报告。
- 提示 CUDA/dlib 缺失：本包仅 CPU + SFD，不要指定 `--device cuda` 或 `--detector dlib`。

支持 JPG/JPEG、PNG、WebP、BMP、TIF/TIFF，默认上限 2500 万像素；过大时可先缩图。更多 API、源码安装和限制见[仓库 README](https://github.com/sincalaway/ptoart#readme)。更新时请解压到新目录，不要混用不同构建的 `_internal`。

## 自检能证明什么

Windows 工作流运行回归测试，并从源码目录外检查冻结 EXE 的版本、帮助、缺模型退出码及 `--self-test`。自检导入 Torch / TensorFlow / ONNX Runtime / 人脸处理模块，用随机权重运行对齐头像推理，再用合成 TensorFlow 图测试分割接口。

**自检不代表官方权重和真实照片完整出图已通过。** 官方模型、首次自动下载、真实检测/分割、视觉质量仍需实际验证；随机权重结果不能用于评价卡通效果。
