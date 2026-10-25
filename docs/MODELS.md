# 模型准备

从 v0.3.0 起，所有推理所需模型都附在 [GitHub Release](https://github.com/sincalaway/ptoart/releases/latest) 中（由发布工作流从上游资源包下载、转换并校验后上传，附 `SHA256SUMS-all.txt`）。EXE/APK 本身不含模型。不要把随机权重测试生成的文件当作可用模型。

## 到哪里取得

**备用（国内直连）：[光鸭云盘 · ptoart 模型](https://www.guangyapan.com/s/1961241077160239155_alCCcsK8loJ3qRN9)**，由仓库维护者分享，包含 Windows/桌面版所需的 `photo2cartoon_weights.pt`、`seg_model_384.pb`，安卓 App 所需的 `photo2cartoon_weights.onnx`，以及仅训练用的 `model_mobilefacenet.pth`。下载后按下表放置文件；建议用 `sha256sum`/`Get-FileHash` 自行留档校验。

其他来源：

[上游项目说明](https://github.com/minivision-ai/photo2cartoon#start)提供了以下资源入口。这些链接来自上游，不代表本次已成功下载或验证文件：

[Google Drive 资源包](https://drive.google.com/open?id=1lsQS8hOCquMFKJFhK_z-n03ixWGkjT2P)；[百度网盘资源包](https://pan.baidu.com/s/1MsT3-He3UGipKhUi4OcCJw)，提取码 `y2ch`；[ONNX 单文件](https://drive.google.com/file/d/1PhwKDUhiq8p-UqrfHCqj257QnqBWD523/view)。

| 文件 | 用途 | 新 CLI 默认位置 |
| --- | --- | --- |
| `photo2cartoon_weights.pt` | Torch 生成器，checkpoint 中含 `genA2B` | `~/.cache/ptoart/models/` |
| `photo2cartoon_weights.onnx` | ONNX 生成器 | 同上 |
| `seg_model_384.pb` | 已裁剪人像分割 | 同上 |
| `model_mobilefacenet.pth` | 训练时的人脸身份损失，不用于普通推理 | 训练脚本沿用 `models/` |

旧 `test.py` / `test_onnx.py` 仍从仓库的 `models/` 读生成器、`utils/` 读分割模型。新 CLI 可用 `--weights`、`--segmentation` 显式指向这些文件。

完整的人脸预处理还会依赖 face-alignment 的检测/关键点资产，其首次运行可能触发第三方下载。离线部署前，应准备并验证这些资产和依赖缓存，不能仅复制上述三个文件就保证离线可用。

## 自行生成 ONNX（可选）

```bash
pip install "torch>=2.5" onnx onnxscript onnxruntime
python tools/export_onnx.py photo2cartoon_weights.pt photo2cartoon_weights.onnx
```

脚本只导出 `genA2B` 生成器（输入 `input`、输出 `output`，均为 `[1,3,256,256]`），并用 ONNX Runtime 与 Torch 结果做数值比对后打印误差。

## 整图风格模型（AnimeGANv2）

```bash
pip install "torch>=2.5" onnx onnxscript onnxruntime tensorflow   # tensorflow 仅 hayao/shinkai 需要
python tools/export_styles.py -o weights/styles          # 全部风格
python tools/export_styles.py hayao paprika -o weights/styles
```

| 风格 | 来源 |
| --- | --- |
| `hayao` 宫崎骏、`shinkai` 新海诚 | [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2) TF checkpoint，脚本直接读取变量转换 |
| `paprika`、`celeba_distill`、`face_paint_512_v1`、`face_paint_512_v2` | [animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)（MIT） |

导出的 ONNX 输入 `input`、输出 `output`，NCHW RGB，取值 [-1,1]，高宽可变（建议 4 的倍数）；脚本会在两种尺寸下与 PyTorch 比对。`face_paint_*` 针对人像训练，对风景效果一般。**AnimeGANv2 权重仅限非商业用途**，商用需联系原作者授权。

## 超分辨率模型（Real-ESRGAN）

```bash
python tools/export_upscalers.py -o weights/upscale      # anime.onnx、general.onnx
```

`anime` 为 realesr-animevideov3（约 2.5 MB，适合卡通输出），`general` 为 realesr-general-x4v3（约 4.9 MB）。输入 [0,1] NCHW RGB，输出 4 倍尺寸；文件带 `ptoart_kind=upscaler` 元数据，安卓据此识别。桌面端按 256 像素分块推理，大图内存可控。BSD-3 许可。

## 安全与完整性

只使用可信来源的模型。新 Torch 后端使用 `torch.load(..., weights_only=True)` 和严格的 state_dict 匹配，不会因装载失败回退到任意 pickle 执行。历史训练代码仍保留旧的权重加载方式；不要用它装载不可信 checkpoint。

上游没有在已读说明中提供可核验的 SHA-256 清单，本项目不会虚构校验值。可在可信下载完成后运行 `sha256sum 文件路径`（Windows 可用 `Get-FileHash`），将哈希与具体部署版本一同留档；自算哈希只能检测后续变更，不能证明来源。

`doctor` 的 ready 只代表文件与模块存在。请进一步实际加载模型，转换测试人像，并验证输出和资源占用。文件存在并不代表格式正确、权重可信或效果合格。

## ONNX 注意事项

输入必须为单一 float32 NCHW `[1,3,256,256]` 张量，输出也须为同一形状的有限数值。输入名从模型元数据读取，优先使用名为 `output` 的输出，否则使用第一个输出。不同导出模型需要自行验证语义和数值一致性。

基础 ONNX extra 安装 CPU Runtime。使用 GPU 需要适合本机 CUDA 的 `onnxruntime-gpu`，避免与 CPU Runtime 冲突。显式要求 CUDA 而 provider 缺失会报错，不静默降级。检测/分割链路仍非纯 ONNX。
