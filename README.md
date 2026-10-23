# ptoart

将人像照片转换为卡通头像的 Python 工具，基于 [minivision-ai/photo2cartoon](https://github.com/minivision-ai/photo2cartoon) 重构。保留原始生成器与训练算法，在其外层建立可复用、可测试的推理管线。

**当前版本：0.1.0，工程重构预览版。** 已提供无权重回归测试和随机权重网络冒烟测试；未完成官方预训练权重、完整人脸检测/分割链路、GPU 或训练的端到端验收。不能将测试通过理解为真实出图质量或生产稳定性承诺。模型权重不在代码包内。

## 这次改了什么

| 原有问题 | ptoart 的处理 |
| --- | --- |
| Torch、ONNX、部署脚本重复推理逻辑 | 统一 `ArtPipeline`，后端通过 `run()` 接口扩展 |
| 导入测试脚本就解析命令并执行工作 | 安全导入，命令从 `main()` 启动 |
| 每个请求重新加载模型 | 一个管线顺序复用模型，显式关闭资源 |
| 路径依赖当前工作目录 | 可指定模型目录/权重路径；兼容入口相对脚本定位 |
| 坏图、无脸或保存失败难以定位 | 分类异常、批次逐项结果、非零退出码 |
| 没有扩展输出方式 | RGB 背景色、透明 PNG、风格混合强度 |
| 没有回归保障 | pytest、静态检查、多 Python 版本 CI 配置 |

这些是工程与输出控制改进，不是新训练模型。默认仍输出 **256×256 的人脸裁剪头像**；不会自动获得高清效果、多人合照完整输出或新绘画风格。多张脸沿用上游的最大脸选择策略。

## Windows exe 自动构建

仓库的 **Actions → Windows EXE** 会在默认分支推送时自动生成 Windows x64 CPU 便携包，也可手动点击 **Run workflow**。成功运行后，在 **Artifacts** 下载 `ptoart-windows-x64-cpu`，解压下载包及其中的 ZIP；保留 `ptoart.exe` 与 `_internal` 文件夹在一起，无需安装 Python。产物保留 30 天。

这是命令行 exe，不是图形界面；包含 Torch、ONNX Runtime、TensorFlow 和使用 SFD 的人脸处理依赖，不包含 dlib、CUDA 或官方模型权重。打包冒烟测试不等于官方模型真实出图验收。使用方法及限制见 [Windows 便携版说明](packaging/WINDOWS-README.md)。

## 安装

推荐 Python 3.11 的独立虚拟环境。基础包适配 Python 3.10+，CI 配置覆盖 3.10–3.12；这不表示所有机器学习依赖组合都已验收。

```bash
cd ptoart
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell 请改用：.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[torch,face]'
```

dlib 可能需要 C++ 编译器和 CMake。GPU 版 Torch、TensorFlow、ONNX Runtime 需按各自平台选择，不应盲目安装多个冲突的 CPU/GPU 包。仅处理已经对齐的头像，可以只安装 `.[torch]` 并显式使用 `--aligned`。ONNX 推理改装 `.[onnx,face]`；完整预处理仍使用 face-alignment/Torch 和 TensorFlow，**并非纯 ONNX 依赖链**。

## 模型准备

从[模型说明](docs/MODELS.md)中的上游入口取得权重，默认放在：

```text
~/.cache/ptoart/models/
├── photo2cartoon_weights.pt      # Torch 后端
├── photo2cartoon_weights.onnx    # ONNX 后端，二选一
└── seg_model_384.pb              # 普通照片预处理必需
```

也可设置 `PTOART_MODEL_DIR`，或传 `--model-dir /path/to/models`。参数优先于环境变量，单文件 `--weights` / `--segmentation` 优先于目录。相对路径基于当前工作目录解析。

```bash
ptoart doctor --model-dir ./weights
```

`doctor` 只检查依赖模块和文件是否存在，不加载模型，不测试 CUDA、模型完整性或出图质量。缺项返回退出码 2；检查通过返回 0。

## 单图转换

```bash
ptoart convert ./images/photo_test.jpg ./output/avatar.png --model-dir ./weights
# 切换 ONNX 后端
ptoart convert portrait.jpg avatar.png --backend onnx --model-dir ./weights
# 透明背景（仅支持 PNG、WebP 或 TIFF 输出）
ptoart convert portrait.jpg avatar.png --background transparent --model-dir ./weights
# 浅蓝背景，卡通/原始裁剪头像混合
ptoart convert portrait.jpg avatar.png --background DCEEFF --strength 0.7 --model-dir ./weights
```

`strength` 在 0–1 之间：1 为完整模型输出，0 为原始裁剪头像；这是后处理混合，不是模型条件控制。默认白色背景；透明度来自原有分割掩码，发丝边缘质量仍由分割模型决定。

若你已经准备好**对齐、裁剪的正方形头像**，可使用 `--aligned` 跳过检测/分割。此模式将整个头像视为不透明，不会自动抠图；把完整照片当作对齐头像会降低效果。

读取支持 EXIF 方向、灰度和透明输入；输入透明区域先合成为白色。默认限制 2500 万像素，可用 `--max-pixels` 调整。输出默认不覆盖，使用 `--overwrite` 才可替换目标；CLI 始终拒绝输入与输出为同一路径。

## 批量转换

```bash
ptoart batch ./photos ./output --recursive --model-dir ./weights --report ./batch-report.json
```

批处理顺序复用同一模型；单图损坏、无脸等错误不会阻断其他图片。保留相对目录，`person.jpg` 输出为 `person.jpg.png`，避免与 `person.png` 冲突。嵌套在输入目录内的输出目录会被跳过；不允许输出目录等于或包含输入目录。已有图片默认记为失败，不悄悄覆盖。JSON 报告同样拒绝覆盖已有文件。

退出码：0 全部成功；1 批次中有失败；2 参数、初始化、空输入或其他命令级错误。JSON 包含每项输入/输出、状态、错误类型及汇总。

## Python API

```python
from ptoart import ArtConfig, ArtPipeline

config = ArtConfig(model_dir="./weights", strength=0.85)
with ArtPipeline(config) as pipeline:
    pipeline.convert_file("portrait.jpg", "avatar.png")
    # 重复调用复用已加载模型
    pipeline.convert_file("another.jpg", "another.png")
```

`convert()` 接收 `uint8 H×W×3 RGB`，返回 `uint8 256×256×3 RGB`，透明背景则返回 RGBA。使用 `NoFaceError` / `ImageError` / `ModelError` 分别处理失败。一个管线用于一个顺序工作者；不要在多个线程中共享实例。`close()` 或上下文管理器释放会话及网络引用，不保证立即归还底层框架缓存的所有显存。

## 旧项目迁移

旧调用仍保留，权重位置沿用原项目：

```bash
python test.py --photo_path ./images/photo_test.jpg --save_path ./output/avatar.png
python test_onnx.py --photo_path ./images/photo_test.jpg --save_path ./output/avatar.png
```

这两个入口的 `Photo2Cartoon.inference(RGB)` 仍返回 BGR；新版 `ArtPipeline` 返回 RGB。原有覆盖目标文件的行为改为默认拒绝；需要覆盖时请使用新版 CLI 的 `--overwrite`。旧资源文件名没有强行改名，以继续兼容官方 checkpoint。

训练与损失函数保留在 `train.py` / `models/`。原数据目录名 `photo2cartoon` 是兼容默认值，也可以将数据放到 `dataset/ptoart/{trainA,trainB,testA,testB}` 并执行 `python train.py --dataset ptoart`。训练额外依赖 `.[train]` 和 MobileFaceNet 权重，仍需要 GPU；未完成现代依赖下的训练验收。旧辅助代码的已知遗留限制见[重构分析](docs/REFACTOR.md)。

`predict.py` 与 `cog.yaml` 是可选 Cog 部署适配，已改为 `setup()` 复用管线；候选容器配置未经过构建和线上验证。

## 开发与验证

```bash
pip install -e '.[dev]'
ruff check ptoart tests test.py test_onnx.py data_process.py predict.py
pytest -q
python -m build
```

没有 Torch 时，网络冒烟测试会明确跳过。真实权重验收前，还需要比对上游/ptoart 结果、Torch/ONNX 输出误差、CPU/GPU 行为、峰值内存、连续处理和不同人群效果；不应以随机权重测试代替这些工作。交付时的实际验证结果见 [VALIDATION.md](docs/VALIDATION.md)。

## 来源与许可

保留上游 MIT [LICENSE](LICENSE)，上游基线为 `6204e27a71620500c2f5b2ffd36646b0e2e5c110`。原中文和英文说明存档于 `docs/UPSTREAM_README_*.md`，其中的相对链接是上游历史记录，使用以本文为准。模型和数据的下载、再分发及商用权利应单独核实；不能仅凭代码的 MIT 许可推断所有数据和权重均获授权。

上游说明指出训练数据以亚洲年轻女性为主，其他人群覆盖不足。重构未消除该偏差，不应用作身份核验系统。
