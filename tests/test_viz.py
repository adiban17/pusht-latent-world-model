import imageio.v2 as imageio
import numpy as np

from wm.viz.media import filmstrip, save_gif


def test_filmstrip_shape():
    frames = np.random.default_rng(0).integers(0, 256, (101, 64, 64, 3), dtype=np.uint8)
    strip = filmstrip(frames, n=8, scale=2, gap=0)
    assert strip.shape == (128, 8 * 128, 3) and strip.dtype == np.uint8
    np.testing.assert_array_equal(strip[::2, :128:2], frames[0])
    np.testing.assert_array_equal(strip[::2, -128::2], frames[-1])


def test_gif_written(tmp_path):
    frames = np.random.default_rng(0).integers(0, 256, (5, 64, 64, 3), dtype=np.uint8)
    path = save_gif(frames, tmp_path / "x.gif")
    back = imageio.mimread(path)
    assert len(back) == 5 and back[0].shape[:2] == (128, 128)
