"""Pixel position -> angles -> head height.

Conventions (all angles in degrees):
  yaw   > 0 : head is to the RIGHT of the camera axis
  pitch > 0 : head is ABOVE the camera axis
  camera_pitch_deg : how far the camera is tilted UP from level

Head height (proposal, CV section):
  height = camera_height + floor_distance * tan(elevation)
"""

import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# Horizontal field of view of the full-sensor 16:9 modes (datasheet).
FOV_PRESETS = {"cam3": 66.0, "cam3_wide": 102.0}


@dataclass
class Intrinsics:
    """Pinhole camera model: focal lengths, image center, lens distortion."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @classmethod
    def from_fov(cls, width, height, hfov_deg):
        """Rough model from the datasheet FOV, no distortion. Fine for aiming,
        not accurate enough for the 5 cm height target (calibrate for that)."""
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(width, height, f, f, width / 2.0, height / 2.0)

    @classmethod
    def load(cls, path):
        """Load a calibration file written by calibrate.py."""
        with open(path) as fh:
            d = json.load(fh)
        K = np.asarray(d["camera_matrix"], dtype=np.float64)
        return cls(int(d["width"]), int(d["height"]), K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                   np.asarray(d["dist_coeffs"], dtype=np.float64).ravel())

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def scaled_to(self, width, height):
        """Same camera at a different resolution (same aspect ratio)."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy,
                          self.cx * sx, self.cy * sy, self.dist.copy())

    def normalize(self, u, v):
        """Pixel -> undistorted normalized coordinates (X/Z, Y/Z)."""
        if np.any(self.dist):
            x, y = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), self.K, self.dist)[0, 0]
            return float(x), float(y)
        return (u - self.cx) / self.fx, (v - self.cy) / self.fy


def ray_angles(xn, yn, camera_pitch_deg=0.0):
    """Yaw and elevation of the ray through a normalized image point.

    With camera_pitch_deg=0 this is the offset from the camera axis (what the
    servos should null). With the real camera tilt, the second value is the
    elevation above level (what the height formula needs).
    """
    right, up, fwd = xn, -yn, 1.0
    t = math.radians(camera_pitch_deg)
    fwd_level = fwd * math.cos(t) - up * math.sin(t)
    up_level = fwd * math.sin(t) + up * math.cos(t)
    yaw = math.degrees(math.atan2(right, fwd_level))
    elev = math.degrees(math.atan2(up_level, math.hypot(fwd_level, right)))
    return yaw, elev


def head_height(camera_height_m, floor_distance_m, elevation_deg):
    return camera_height_m + floor_distance_m * math.tan(math.radians(elevation_deg))
