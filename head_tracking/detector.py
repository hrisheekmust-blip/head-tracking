"""Face detection with YuNet, a small pre-trained face detector that ships with
OpenCV's model zoo (~230 KB, trained on the WIDER FACE dataset). No training needed.

To save CPU, once we know where the listener is we only search a small square
around their last position instead of the whole frame.
"""

import os
import re
import time
from dataclasses import dataclass

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def opencv_version():
    m = re.match(r"(\d+)\.(\d+)", cv2.__version__)
    return int(m.group(1)), int(m.group(2))


def default_model_path():
    """OpenCV 5 needs the 2026 model (dynamic input size); OpenCV 4.8+ uses the 2023 one."""
    v = opencv_version()
    if v < (4, 8):
        raise RuntimeError(
            f"OpenCV {cv2.__version__} is too old for YuNet (need 4.8+). On the Pi run: "
            "pip install --break-system-packages opencv-python-headless")
    name = "face_detection_yunet_2026may.onnx" if v >= (5, 0) else "face_detection_yunet_2023mar.onnx"
    return os.path.join(MODEL_DIR, name)


@dataclass
class Face:
    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: np.ndarray  # 5x2: right eye, left eye, nose, right mouth, left mouth

    @property
    def area(self):
        return self.w * self.h

    @property
    def head_point(self):
        """Midpoint between the eyes. Ear height is about eye height."""
        (rx, ry), (lx, ly) = self.landmarks[0], self.landmarks[1]
        return (rx + lx) / 2.0, (ry + ly) / 2.0

    @property
    def bbox(self):
        return self.x, self.y, self.w, self.h

    def moved(self, scale, dx, dy):
        """Map from a cropped/resized image back to full-frame pixels."""
        return Face(self.x * scale + dx, self.y * scale + dy, self.w * scale, self.h * scale,
                    self.score, self.landmarks * scale + np.array([dx, dy]))


class YuNet:
    def __init__(self, model_path=None, score_threshold=0.5):
        model_path = model_path or default_model_path()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Run: python3 download_models.py")
        self._size = (320, 320)
        self._net = cv2.FaceDetectorYN.create(model_path, "", self._size, score_threshold, 0.3, 20)

    def detect(self, img):
        h, w = img.shape[:2]
        if (w, h) != self._size:
            self._size = (w, h)
            self._net.setInputSize(self._size)
        _, rows = self._net.detect(img)
        if rows is None:
            return []
        return [Face(*map(float, r[:4]), float(r[14]), r[4:14].reshape(5, 2).astype(np.float64))
                for r in rows]


class FaceFinder:
    """Decides where to search each frame.

    - Tracking: search a roi_size square around the last face (about 7x cheaper
      than the whole frame). If that misses, search the whole frame.
    - Every full_every frames: search the whole frame anyway, as a check.
    - Nobody in view: search the whole frame at most search_hz times a second.

    Call find(frame), then focus(bbox) with the face the tracker is following
    (or None once it is lost).
    """

    def __init__(self, model_path=None, score_threshold=0.5, roi=True, roi_size=192,
                 full_every=30, search_hz=5.0):
        self._full = YuNet(model_path, score_threshold)
        # Separate instance so switching between crop and full frame doesn't
        # force the network to be resized every frame.
        self._crop = YuNet(model_path, score_threshold)
        self.roi = roi
        self.roi_size = roi_size
        self.full_every = max(int(full_every), 1)
        self.search_interval = 1.0 / search_hz if search_hz > 0 else 0.0
        self._focus = None
        self._frames_since_full = 0
        self._last_full_t = -1e9
        self.mode = "full"  # what happened on the last frame: "roi", "full" or "skip"

    def focus(self, bbox):
        self._focus = bbox

    def _search_roi(self, frame):
        H, W = frame.shape[:2]
        x, y, w, h = self._focus
        side = int(min(max(self.roi_size, 3 * max(w, h)), W, H))
        x0 = int(min(max(x + w / 2 - side / 2, 0), W - side))
        y0 = int(min(max(y + h / 2 - side / 2, 0), H - side))
        crop = frame[y0:y0 + side, x0:x0 + side]
        scale = 1.0
        if side > self.roi_size:
            scale = side / self.roi_size
            crop = cv2.resize(crop, (self.roi_size, self.roi_size), interpolation=cv2.INTER_AREA)
        return [f.moved(scale, x0, y0) for f in self._crop.detect(crop)]

    def find(self, frame, now=None):
        now = time.monotonic() if now is None else now
        self._frames_since_full += 1

        if self._focus is not None:
            if self.roi and self._frames_since_full < self.full_every:
                faces = self._search_roi(frame)
                if faces:
                    self.mode = "roi"
                    return faces
        elif now - self._last_full_t < self.search_interval:
            self.mode = "skip"
            return []

        self._frames_since_full = 0
        self._last_full_t = now
        self.mode = "full"
        return self._full.detect(frame)
