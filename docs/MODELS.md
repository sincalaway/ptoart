# 模型准备

ptoart 不包含、不自动下载，也不重新分发上游模型权重。不要把随机权重测试生成的文件当作可用模型。

## 到哪里取得

**推荐（国内直连）：[光鸭云盘 · ptoart 模型](https://www.guangyapan.com/s/1961237335383875651_alCCcsK8loJ3qRN9)**，由仓库维护者分享。下载后按下表放置文件；建议用 `sha256sum`/`Get-FileHash` 自行留档校验。

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

## 安全与完整性

只使用可信来源的模型。新 Torch 后端使用 `torch.load(..., weights_only=True)` 和严格的 state_dict 匹配，不会因装载失败回退到任意 pickle 执行。历史训练代码仍保留旧的权重加载方式；不要用它装载不可信 checkpoint。

上游没有在已读说明中提供可核验的 SHA-256 清单，本项目不会虚构校验值。可在可信下载完成后运行 `sha256sum 文件路径`（Windows 可用 `Get-FileHash`），将哈希与具体部署版本一同留档；自算哈希只能检测后续变更，不能证明来源。

`doctor` 的 ready 只代表文件与模块存在。请进一步实际加载模型，转换测试人像，并验证输出和资源占用。文件存在并不代表格式正确、权重可信或效果合格。

## ONNX 注意事项

输入必须为单一 float32 NCHW `[1,3,256,256]` 张量，输出也须为同一形状的有限数值。输入名从模型元数据读取，优先使用名为 `output` 的输出，否则使用第一个输出。不同导出模型需要自行验证语义和数值一致性。

基础 ONNX extra 安装 CPU Runtime。使用 GPU 需要适合本机 CUDA 的 `onnxruntime-gpu`，避免与 CPU Runtime 冲突。显式要求 CUDA 而 provider 缺失会报错，不静默降级。检测/分割链路仍非纯 ONNX。
