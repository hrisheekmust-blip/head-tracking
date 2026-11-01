"""Turns per-frame face detections into one smoothed head measurement."""

import math
import threading
import time
from dataclasses import asdict, dataclass

from .geometry import head_height, ray_angles


@dataclass
class HeadMeasurement:
    robot_id: int
    detected: bool                 # face seen in THIS frame
    yaw_error_deg: float = None    # head right of camera axis (+)  -> pan servo
    pitch_error_deg: float = None  # head above camera axis (+)     -> tilt servo
    elevation_deg: float = None    # head angle above level from the camera
    head_height_m: float = None    # needs a floor distance
    distance_m: float = None
    confidence: float = 0.0
    head_px: tuple = None
    bbox: tuple = None

    def to_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


class HeadTracker:
    """Picks which face is the listener and smooths its position.

    Target choice: keep following the face nearest the last position (within
    gate_frac of the image width); otherwise take the largest face, i.e. the
    closest person. After a miss, the last estimate is held for lost_after_s
    (with detected=False) before reporting the listener as lost.
    """

    def __init__(self, intrinsics, robot_id=1, camera_height_m=0.95, camera_pitch_deg=0.0,
                 smoothing=0.5, gate_frac=0.25, lost_after_s=0.5, distance_timeout_s=1.0):
        self.K = intrinsics
        self.robot_id = robot_id
        self.camera_height_m = camera_height_m
        self.alpha = 1.0 - min(max(smoothing, 0.0), 0.95)
        self.gate_frac = gate_frac
        self.lost_after_s = lost_after_s
        self.distance_timeout_s = distance_timeout_s
        self.fixed_distance_m = None

        self._lock = threading.Lock()  # inputs arrive from the UDP thread
        self._camera_pitch = camera_pitch_deg
        self._live_distance = None     # (metres, time.monotonic() when received)

        self._pt = None
        self._face = None
        self._last_seen = None

    # Inputs from other subsystems (safe to call from another thread).
    def set_distance(self, metres):
        """Floor distance from THIS robot's camera to the listener (overhead CV)."""
        with self._lock:
            self._live_distance = (float(metres), time.monotonic())

    def set_camera_pitch(self, deg):
        """Camera tilt up from level (mount angle + tilt servo)."""
        with self._lock:
            self._camera_pitch = float(deg)

    def _distance(self):
        with self._lock:
            live = self._live_distance
        if live and time.monotonic() - live[1] <= self.distance_timeout_s:
            return live[0]
        return self.fixed_distance_m

    def _pick(self, faces, width):
        if not faces:
            return None
        if self._pt is not None:
            nearest = min(faces, key=lambda f: math.dist(f.head_point, self._pt))
            if math.dist(nearest.head_point, self._pt) <= self.gate_frac * width:
                return nearest
        return max(faces, key=lambda f: f.area)

    def update(self, faces, frame_shape, t):
        """t: frame time in seconds (any clock, just consistent between calls)."""
        h, w = frame_shape[:2]
        if (w, h) != (self.K.width, self.K.height):
            self.K = self.K.scaled_to(w, h)

        face = self._pick(faces, w)
        if face is not None:
            new = face.head_point
            if self._pt is None:
                self._pt = new
            else:
                a = self.alpha
                self._pt = (a * new[0] + (1 - a) * self._pt[0], a * new[1] + (1 - a) * self._pt[1])
            self._face, self._last_seen = face, t
        elif self._pt is not None and t - self._last_seen > self.lost_after_s:
            self._pt = self._face = None

        if self._pt is None:
            return HeadMeasurement(self.robot_id, detected=False)

        with self._lock:
            cam_pitch = self._camera_pitch
        xn, yn = self.K.normalize(*self._pt)
        yaw, pitch = ray_angles(xn, yn)
        _, elev = ray_angles(xn, yn, cam_pitch)
        dist = self._distance()
        return HeadMeasurement(
            robot_id=self.robot_id, detected=face is not None,
            yaw_error_deg=yaw, pitch_error_deg=pitch, elevation_deg=elev,
            head_height_m=head_height(self.camera_height_m, dist, elev) if dist else None,
            distance_m=dist, confidence=face.score if face else 0.0,
            head_px=self._pt, bbox=self._face.bbox,
        )
