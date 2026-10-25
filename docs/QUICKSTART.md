# ptoart 小白使用教程

照片 → 卡通头像 / 整张照片卡通化 / 宫崎骏等动漫风格。全部在你自己的电脑或手机上离线处理，照片不上传。

所有文件都在同一个下载页：**https://github.com/sincalaway/ptoart/releases/latest**（往下拉到 **Assets**，点文件名即可下载）。

---

## 一、安卓手机（最简单）

**要下载的文件**（用手机浏览器打开上面的下载页）：

| 文件 | 必需？ | 作用 |
| --- | --- | --- |
| `ptoart-android.apk` | 必需 | App 安装包 |
| `photo2cartoon_weights.onnx` | 必需（做卡通头像） | 卡通头像模型 |
| `ptoart-styles-animeganv2.zip` | 可选 | 宫崎骏、新海诚等 6 种整图风格 |
| `ptoart-upscalers-realesrgan.zip` | 可选 | 让结果更清晰的超分模型 |

**步骤：**

1. 点开下载好的 `ptoart-android.apk` 安装。手机提示“禁止安装未知来源应用”时，按提示允许浏览器/文件管理器安装即可（安卓 7.0 及以上）。
2. 两个 zip 先用手机自带的“文件管理”解压（长按 → 解压）。
3. 打开 App，点 **「导入模型（.onnx，可多选）」**，选中 `photo2cartoon_weights.onnx`；想要风格和高清，再把解压出来的 `hayao.onnx`、`shinkai.onnx`… 和 `anime.onnx` 一起多选导入。导入一次即可，以后不用再导。
4. 在下拉框选模式：
   - **卡通头像**：最大那张脸，输出 512×512 头像；
   - **整张照片**：照片里每张脸都卡通化，再贴回原图，身体和背景不变；
   - **整图风格：宫崎骏 / 新海诚…**：整张图变动漫风，风景、全身照都能用。
5. 点 **「选择照片」→「开始转换」**，满意后点 **「保存 PNG」** 选个位置保存。
6. 想更清晰就打开 **「超分辨率增强」** 开关（会慢几秒）。

---

## 二、Windows 电脑

Windows 版没有图形界面，但准备了 **拖拽即用** 的小工具，不用敲命令。

**要下载的文件**（电脑浏览器打开下载页）：

| 文件 | 必需？ |
| --- | --- |
| `ptoart-windows-x64-cpu.zip` | 必需 |
| `photo2cartoon_weights.pt` | 必需（头像、整张照片） |
| `seg_model_384.pb` | 必需（头像、整张照片） |
| `ptoart-styles-animeganv2.zip` | 可选（整图风格） |
| `ptoart-upscalers-realesrgan.zip` | 可选（更清晰） |

**步骤：**

1. 右键 `ptoart-windows-x64-cpu.zip` → **全部解压缩**，比如解压到 `C:\ptoart`。不要在压缩包里直接运行。
2. 把 `photo2cartoon_weights.pt` 和 `seg_model_384.pb` 放进 `C:\ptoart\ptoart\weights` 文件夹。
3. （可选）把风格 zip 解压到 `weights\styles`，把超分 zip 解压到 `weights\upscale`。最终像这样：

   ```text
   ptoart\
   ├── ptoart.exe
   ├── 1-avatar.bat          ← 拖照片到这里：卡通头像
   ├── 2-whole-photo.bat     ← 拖照片到这里：整张照片卡通化
   ├── 3-style-hayao.bat     ← 拖照片到这里：宫崎骏风格
   ├── _internal\
   └── weights\
       ├── photo2cartoon_weights.pt
       ├── seg_model_384.pb
       ├── styles\hayao.onnx、shinkai.onnx …
       └── upscale\anime.onnx、general.onnx
   ```

4. **把照片拖到对应的 .bat 文件上**（可以一次拖多张）。黑色窗口跑完后会自动打开 `output` 文件夹，结果就在里面。
   - 放了超分模型会自动用超分，头像为 1024×1024 高清图。
5. 第一次处理人脸时请**保持联网**：程序会自动下载人脸检测数据（约几十 MB，只下一次）。之后可离线使用。
6. Windows 提示“已保护你的电脑”时，点 **更多信息 → 仍要运行**（程序未做数字签名）。

想用其他风格（新海诚等）或调参数，见 [README](../README.md) 的命令行说明，例如：

```powershell
cd C:\ptoart\ptoart
.\ptoart.exe convert .\me.jpg .\output\me-shinkai.jpg --style shinkai --model-dir .\weights
```

---

## 三、常见问题

- **提示找不到模型**：检查文件名没被改（如变成 `photo2cartoon_weights (1).pt`），且放在 `weights` 文件夹里。
- **没检测到人脸**：用正脸、清晰、光线好的照片；侧脸、遮挡、太小的脸可能检测不到。只想要动漫风可以用“整图风格”，它不需要人脸。
- **太慢**：纯 CPU 运行，大图 + 超分可能要十几秒到一分钟，属正常。
- **能商用吗**：卡通头像（photo2cartoon，MIT）和超分（Real-ESRGAN，BSD-3）可以；**AnimeGANv2 整图风格仅限非商业用途**。
