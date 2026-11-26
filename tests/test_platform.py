from pathlib import Path

import wm.platform as P


def test_local_by_default():
    assert P.detect_platform() == "local"


def test_kaggle_detected(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert P.detect_platform() == "kaggle"


def test_colab_detected(monkeypatch):
    monkeypatch.setattr(P, "_colab_importable", lambda: True)
    assert P.detect_platform() == "colab"


def test_out_dirs(monkeypatch):
    monkeypatch.delenv("WM_OUT_DIR")
    assert P.out_dir("kaggle") == Path("/kaggle/working/outputs")
    assert P.out_dir("colab") == Path("/content/outputs")
    assert P.out_dir("local") == Path("./outputs")


def test_device_never_mps():
    dev = P.get_device()
    assert dev.type in ("cuda", "cpu")
    if dev.type == "cpu":
        assert P.device_name(dev) == "cpu"
