import numpy as np
import pytest

from head_tracking.detector import Face, FaceFinder, default_model_path, opencv_version

FRAME = np.zeros((360, 640, 3), np.uint8)


class FakeYuNet:
    """Reports one face at a fixed full-frame position, if it's inside the image it gets.
    The image is assumed to be a crop starting at `offset`, scaled down by `scale`."""

    def __init__(self, x, y, w=40, h=50):
        self.box = (x, y, w, h)
        self.offset, self.scale = (0, 0), 1.0
        self.sizes = []

    def detect(self, img):
        self.sizes.append((img.shape[1], img.shape[0]))
        x, y, w, h = self.box
        s = self.scale
        fx, fy = (x - self.offset[0]) / s, (y - self.offset[1]) / s
        if fx < 0 or fy < 0 or fx + w / s > img.shape[1] or fy + h / s > img.shape[0]:
            return []
        return [Face(fx, fy, w / s, h / s, 0.9, np.zeros((5, 2)))]


def finder(full, crop, **kw):
    f = FaceFinder(**kw)
    f._full, f._crop = full, crop
    return f


def test_crop_search_maps_back_to_full_frame():
    crop = FakeYuNet(400, 150)
    crop.offset = (420 - 96, 175 - 96)  # 192 px square centered on the face
    f = finder(FakeYuNet(400, 150), crop, search_hz=0)
    f.focus((400, 150, 40, 50))
    faces = f.find(FRAME, now=0)
    assert f.mode == "roi" and crop.sizes[-1] == (192, 192)
    assert (faces[0].x, faces[0].y) == pytest.approx((400, 150))


def test_crop_miss_falls_back_to_full_frame():
    f = finder(FakeYuNet(100, 100), FakeYuNet(-999, -999), search_hz=0)
    f.focus((400, 150, 40, 50))
    assert len(f.find(FRAME, now=0)) == 1 and f.mode == "full"


def test_full_frame_check_every_n_frames():
    crop = FakeYuNet(0, 0)
    f = finder(FakeYuNet(0, 0), crop, full_every=3, search_hz=0)
    f.focus((0, 0, 40, 50))
    modes = []
    for i in range(6):
        f.find(FRAME, now=i)
        modes.append(f.mode)
    assert modes == ["roi", "roi", "full", "roi", "roi", "full"]


def test_idle_search_rate_is_capped():
    f = finder(FakeYuNet(-999, -999), FakeYuNet(-999, -999), search_hz=5)
    modes = []
    for i in range(15):  # 1 s at 15 fps, nobody in view
        f.find(FRAME, now=i / 15)
        modes.append(f.mode)
    assert modes.count("full") == 5


def test_model_matches_opencv_version():
    name = default_model_path()
    assert name.endswith("2026may.onnx" if opencv_version() >= (5, 0) else "2023mar.onnx")
