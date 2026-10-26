DCT-Net portrait cartoon models for ptoart (looks: DCT 日漫 / DCT 3D / DCT 手绘)

Upstream: DCT-Net, Alibaba DAMO Academy (Men Yifang et al., "DCT-Net: Domain-Calibrated
Translation for Portrait Stylization", SIGGRAPH 2022).
  Code:    https://github.com/menyifang/DCT-Net
  Weights: ModelScope damo/cv_unet_person-image-cartoon_compound-models (anime),
           damo/cv_unet_person-image-cartoon-3d_compound-models (3d),
           damo/cv_unet_person-image-cartoon-handdrawn_compound-models (handdrawn)
Licence: Apache License 2.0 (see LICENSE-DCT-Net-Apache-2.0.txt). Commercial use is allowed
under its terms.

Modifications (NOTICE): the upstream TensorFlow frozen graphs cartoon_h.pb (head) and
cartoon_bg.pb (background) were converted to ONNX with tf2onnx and wrapped so the input and
output are NCHW RGB float in [-1,1], like every other ptoart style. Weights are unchanged;
outputs match TensorFlow to within 1 uint8 level. Converter: tools/export_dctnet.py.

Files
  dct_anime.onnx        head model, 日漫 look (faces)
  dct_anime_bg.onnx     background model, 日漫 look (whole photo / 整图风格 "DCT 日漫")
  dct_3d.onnx           3D look (upstream head and background models are identical)
  dct_handdrawn.onnx    head model, 手绘 look
  dct_handdrawn_bg.onnx background model, 手绘 look

Windows: unzip into weights\styles\ beside ptoart.exe, then click 「重新检测模型」.
Android: unzip, then 「导入模型」 and multi-select the .onnx files.
