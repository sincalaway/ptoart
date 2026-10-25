# ptoart 验证结果

## 已经通过

**31 项 pytest 测试全部通过，耗时约 4 秒。** 这证明当前测试覆盖的工程行为符合预期，不代表已验证官方模型的视觉效果。

| 验证项 | 实际结果 |
| --- | --- |
| 统一管线：RGB、白底数学流程、强度、透明度、输出范围 | 通过注入式组件回归测试 |
| 错误处理：无脸、损坏图、超限、非法模型输出 | 通过 |
| I/O：中文路径、EXIF、灰度/透明输入、防覆盖、透明 JPEG 拒绝 | 通过 |
| 批处理：递归、扩展名冲突规避、坏图继续、嵌套输出排除 | 通过 |
| Torch 实际网络 | 原生成器共 3,831,079 参数；随机 checkpoint 可受限加载，eval CPU 前向输出有限的 1×3×256×256 |
| 不可信 Torch checkpoint | 含 pickle 副作用的测试输入被拒绝，没有执行副作用 |
| ONNX Runtime | 合成 Identity 图实际运行；动态读取张量名、错误输入形状及损坏模型处理通过 |
| 轻量导入 | help/doctor 在阻断所有 ML 框架导入的测试进程中正常执行 |
| 初始化前预检 | 缺失/损坏/超限图片、无效输出、透明 JPEG、空目录和非正方形 aligned 输入均在模型初始化前拒绝 |
| 框架初始化异常 | 模拟设备初始化失败被包装为 ModelError 并保留原始异常链 |
| 静态与打包 | Ruff、git diff --check、pip check 通过；wheel 与源码分发构建成功 |
| 独立 wheel 安装 | 独立虚拟环境安装 wheel，从仓库外目录导入成功；不是复用源码路径 |
| wheel 安装后的 CLI | 合成 ONNX 模型运行 doctor、单图、拒绝覆盖、批处理、部分失败；退出码分别符合 0/2/1 约定 |

本次只实际运行了 Python 3.12.3 / Linux CPU 环境。CI 文件配置了 Python 3.10、3.11、3.12 的核心测试及独立 Torch、ONNX 冒烟任务，**尚未在远程 GitHub Actions 执行**。

## 仍存在的警告

保留上游网络实现产生 4 条 `torch.nn.functional.upsample` 弃用警告。本轮没有为清除警告而修改生成器计算代码或参数层级。构建工具对旧式许可证元数据也可能发出弃用提示，不影响此次构建。

## 尚未验证

没有下载或随包附带官方 `ptoart_weights.pt`、ONNX 生成器及 `seg_model_384.pb`。随机 Torch 权重与合成 ONNX 图只能验证结构、装载和工程链路，**不能产生可用的卡通效果证明**。

完整 face-alignment + TensorFlow 人脸预处理没有实际运行；官方权重下的原版/新版结果误差、Torch/ONNX 一致性、GPU、性能基准、持续显存占用、跨平台、Cog 容器和训练尚未验收。现代依赖范围和 Cog 依赖固定组合属于候选配置，不是完整推理环境锁文件。

## 如何复验

```bash
pip install -e '.[dev,torch,onnx]' onnx
pytest -q
ruff check ptoart tests test.py test_onnx.py data_process.py predict.py
python -m build
```

不安装 Torch 或 ONNX 测试依赖时，对应用例会明确跳过，应同时查看 passed 与 skipped，不能只看退出码。官方模型验收需要先按 [MODELS.md](MODELS.md) 准备可信权重，再在普通照片模式下运行转换与上游结果比较。
