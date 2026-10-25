"""Console entry for the Windows CPU distribution (no model downloads at build time)."""
import multiprocessing
import os
import sys
from pathlib import Path


def default_weights_dir():
    """Return the ``weights`` folder next to the frozen EXE (or None if absent)."""
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else None
    if base is None:
        return None
    folder = base / "weights"
    return folder if folder.is_dir() else None


def frozen_self_test():
    """Exercise packaged native libraries and random-weight inference, not real model quality."""
    import importlib
    import tempfile
    from pathlib import Path

    import numpy as np
    import torch
    import tensorflow as tf
    import onnxruntime as ort
    from PIL import Image
    from models.networks import ResnetGenerator
    from ptoart.cli import main
    from utils.face_seg import FaceSeg

    for module in ("face_alignment", "face_alignment.detection.sfd", "utils.preprocess"):
        importlib.import_module(module)
    assert "CPUExecutionProvider" in ort.get_available_providers()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="ptoart-smoke-") as folder:
        root = Path(folder)
        source, target = root / "input.png", root / "output.png"
        Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(source)
        torch.manual_seed(0)
        network = ResnetGenerator(ngf=32, img_size=256, light=True)
        checkpoint = root / "random.pt"
        torch.save({"genA2B": network.state_dict()}, checkpoint)
        del network
        result = main(["convert", str(source), str(target), "--aligned", "--device", "cpu",
                       "--weights", str(checkpoint)])
        assert result == 0 and target.is_file(), "Frozen Torch inference failed"
        with Image.open(target) as image:
            assert image.size == (256, 256)
        graph = tf.Graph()
        with graph.as_default():
            image = tf.compat.v1.placeholder(tf.float32, [None, 384, 384, 3], name="input_1")
            tf.identity(tf.reduce_mean(image, axis=3, keepdims=True), name="sigmoid/Sigmoid")
        segmentation = root / "synthetic.pb"
        segmentation.write_bytes(graph.as_graph_def().SerializeToString())
        with FaceSeg(segmentation) as segmenter:
            mask = segmenter.get_mask(np.full((32, 32, 3), 128, dtype=np.uint8))
            assert mask.shape == (32, 32)
    print("Frozen self-test passed: imports, CPU random-weight Torch inference, synthetic TF segmentation.")
    print("Official weights and real-photo end-to-end quality are NOT validated.")
    return 0


def run(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--self-test"]:
        return frozen_self_test()
    if args == ["gui"] or (not args and getattr(sys, "frozen", False)):
        # Double-clicked ptoart.exe: open the window instead of a bare console.
        from ptoart.gui import hide_own_console, run_gui
        hide_own_console()
        return run_gui()
    if len(args) == 2 and args[0] == "--gui-screenshot":
        import traceback
        try:
            from ptoart.gui import run_gui
            return run_gui(screenshot=args[1])
        except BaseException:
            Path(os.environ.get("PTOART_GUI_LOG", "gui.log")).write_text(traceback.format_exc(), encoding="utf-8")
            raise
    # Avoid a Windows C++/dlib dependency. Explicit CLI selections are respected.
    if args and args[0] in {"convert", "batch", "doctor"}:
        if not any(arg == "--detector" or arg.startswith("--detector=") for arg in args):
            args.extend(["--detector", "sfd"])
        # Portable layout: use the "weights" folder beside ptoart.exe when the user
        # gave no explicit model location and PTOART_MODEL_DIR is unset.
        explicit = ("--model-dir", "--weights")
        if (not any(a in explicit or a.startswith(tuple(e + "=" for e in explicit)) for a in args)
                and not os.environ.get("PTOART_MODEL_DIR")):
            portable = default_weights_dir()
            if portable is not None:
                args.extend(["--model-dir", str(portable)])
    from ptoart.cli import main
    return main(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(run())
