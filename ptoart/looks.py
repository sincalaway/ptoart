"""Face looks shared by the GUI and the CLI; imports no ML frameworks.

A look picks the model that paints faces in avatar ("face") and whole-photo ("photo")
modes. ``None`` means the original ptoart generator (ptoart_weights.pt / .onnx).
DCT-Net looks also name a scene model: in photo mode the whole picture is stylized
with it first and the DCT-Net faces are pasted on top, as upstream DCT-Net does.
"""

# key -> (label, face painter style stem or None for the ptoart generator).
# Order is the menu order; the first available look is the default.
FACE_LOOKS = {
    "paint": ("动漫脸（推荐）", "face_paint_512_v2"),
    "dct_anime": ("DCT 日漫", "dct_anime"),
    "dct_3d": ("DCT 3D", "dct_3d"),
    "dct_handdrawn": ("DCT 手绘", "dct_handdrawn"),
    "jp": ("日漫二次元", "jp_face"),
    "ghibli": ("吉卜力人像", "ghibli_face"),
    "sketch": ("黑白素描", "portrait_sketch"),
    "classic": ("经典卡通", None),
}
LOOK_HINTS = {
    "paint": "五官自然不变形，儿童、老人都好看",
    "dct_anime": "阿里达摩院 DCT-Net：清爽日漫脸，宝宝、儿童不变形；整张照片连背景一起变",
    "dct_3d": "DCT-Net：3D 动画电影质感，圆润可爱",
    "dct_handdrawn": "DCT-Net：手绘插画感，线条柔和",
    "jp": "大眼睛二次元脸",
    "ghibli": "柔和手绘油彩感",
    "sketch": "黑白线稿",
    "classic": "原版卡通，线条夸张；儿童照片可能变形",
}
# Whole-photo background model for DCT-Net looks. When the *_bg file is absent the
# head model itself is used (for 3d the upstream head and background models are identical).
LOOK_SCENES = {
    "dct_anime": "dct_anime_bg",
    "dct_3d": "dct_3d",
    "dct_handdrawn": "dct_handdrawn_bg",
}
# Face-only DCT-Net head models; not offered as whole-image styles (their *_bg twin is).
HEAD_ONLY_STYLES = {"dct_anime", "dct_handdrawn"}


def scene_for(look, available):
    """Scene style stem for ``look`` among ``available`` style stems, or None."""
    scene = LOOK_SCENES.get(look)
    if scene is None:
        return None
    if scene in available:
        return scene
    painter = FACE_LOOKS[look][1]
    return painter if painter in available else None
