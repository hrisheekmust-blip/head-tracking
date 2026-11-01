"""Frame sources: the Pi Camera Module 3, or a webcam / video file for testing."""

import time

import cv2


class PiCamera:
    """Pi Camera Module 3 through Picamera2, with autofocus locked.

    Autofocus must stay off: refocusing changes the focal length and silently
    breaks the calibration. lens_position is in dioptres (1 / metres), so 0.5
    focuses at about 2 m. Use the same value you calibrated with.
    """

    def __init__(self, width=640, height=360, fps=15.0, lens_position=0.5, camera_num=0):
        try:
            from picamera2 import Picamera2
            from libcamera import controls
        except ImportError as e:
            raise RuntimeError("picamera2 missing. On Raspberry Pi OS: "
                               "sudo apt install -y python3-picamera2") from e

        self._cam = Picamera2(camera_num)
        # The Pi's image processor (not the CPU) scales down to width x height,
        # and FrameRate makes the sensor itself run at fps.
        # raw size = the full-field-of-view binned mode, so the image isn't cropped.
        config = dict(main={"size": (width, height), "format": "RGB888"},
                      controls={"FrameRate": float(fps)},
                      raw={"size": (2304, 1296)}, buffer_count=3, queue=False)
        try:
            self._cam.configure(self._cam.create_video_configuration(**config))
        except Exception:
            del config["raw"]  # a camera without that sensor mode
            self._cam.configure(self._cam.create_video_configuration(**config))
        self._cam.start()
        try:
            self._cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                                    "LensPosition": float(lens_position)})
        except Exception:
            pass  # camera has no autofocus, nothing to lock
        time.sleep(0.5)  # let exposure settle

    def read(self):
        # Picamera2's "RGB888" is laid out B,G,R in memory, which is what OpenCV expects.
        return self._cam.capture_array("main")

    def close(self):
        self._cam.stop()
        self._cam.close()


class OpenCVCamera:
    def __init__(self, source, width=640, height=360):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source {source!r}")
        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or None

    def read(self):
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        self._cap.release()


def open_camera(source, width=640, height=360, fps=15.0, lens_position=0.5):
    """'picam' (or 'picam:1') -> Pi camera, '0' -> webcam 0, anything else -> video file."""
    if source.startswith("picam"):
        num = int(source.split(":")[1]) if ":" in source else 0
        return PiCamera(width, height, fps, lens_position, num)
    if source.isdigit():
        return OpenCVCamera(int(source), width, height)
    return OpenCVCamera(source)
