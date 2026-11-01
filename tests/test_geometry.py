import math

import numpy as np
import pytest

from head_tracking.detector import Face
from head_tracking.geometry import Intrinsics, head_height, ray_angles
from head_tracking.tracker import HeadTracker

SHAPE = (360, 640, 3)


def cam():
    return Intrinsics.from_fov(640, 360, 66.0)


def face_at(u, v, size=60):
    eyes = np.array([[u - 12, v], [u + 12, v], [u, v + 15], [u - 8, v + 30], [u + 8, v + 30]], float)
    return Face(u - size / 2, v - size / 3, size, size * 1.2, 0.9, eyes)


def test_center_pixel_is_on_axis():
    assert ray_angles(*cam().normalize(320, 180)) == pytest.approx((0, 0), abs=1e-9)


def test_image_edge_is_half_the_fov():
    yaw, _ = ray_angles(*cam().normalize(640, 180))
    assert yaw == pytest.approx(33.0)


def test_signs_right_and_up_are_positive():
    yaw, pitch = ray_angles(*cam().normalize(500, 50))
    assert yaw > 0 and pitch > 0


def test_camera_tilt_adds_to_elevation():
    assert ray_angles(0, 0, camera_pitch_deg=10)[1] == pytest.approx(10)


def test_height_formula():
    elev = math.degrees(math.atan2(0.70, 2.0))
    assert head_height(0.95, 2.0, elev) == pytest.approx(1.65)


def test_recovers_known_head_height_with_tilted_camera():
    """Put a head at 1.20 m, 2 m away, project it into a camera tilted up 5 deg,
    then check the tracker gets 1.20 m back."""
    k = cam()
    cam_h, dist, true_h, tilt = 0.95, 2.0, 1.20, 5.0
    t = math.radians(tilt)
    up, fwd = true_h - cam_h, dist
    fwd_c = fwd * math.cos(t) + up * math.sin(t)
    up_c = -fwd * math.sin(t) + up * math.cos(t)
    u, v = k.cx, k.cy - k.fy * up_c / fwd_c

    tr = HeadTracker(k, camera_height_m=cam_h, camera_pitch_deg=tilt, smoothing=0)
    tr.fixed_distance_m = dist
    m = tr.update([face_at(u, v)], SHAPE, t=0)
    assert m.head_height_m == pytest.approx(true_h, abs=1e-6)
    assert m.yaw_error_deg == pytest.approx(0, abs=1e-6)


def test_sticks_with_the_same_person():
    tr = HeadTracker(cam(), smoothing=0)
    listener, bigger_stranger = face_at(320, 180), face_at(60, 180, size=120)
    tr.update([listener], SHAPE, t=0)
    m = tr.update([listener, bigger_stranger], SHAPE, t=0.07)
    assert m.head_px == pytest.approx(listener.head_point)


def test_holds_briefly_then_reports_lost():
    tr = HeadTracker(cam(), lost_after_s=0.5)
    tr.update([face_at(320, 180)], SHAPE, t=0)
    held = tr.update([], SHAPE, t=0.2)
    assert not held.detected and held.yaw_error_deg is not None and held.bbox is not None
    lost = tr.update([], SHAPE, t=0.8)
    assert not lost.detected and lost.yaw_error_deg is None and lost.bbox is None


def test_live_distance_overrides_fixed_then_expires(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("head_tracking.tracker.time.monotonic", lambda: clock[0])
    tr = HeadTracker(cam(), distance_timeout_s=1.0)
    tr.fixed_distance_m = 3.0
    tr.set_distance(2.0)
    assert tr.update([face_at(320, 180)], SHAPE, t=0).distance_m == 2.0
    clock[0] += 1.5
    assert tr.update([face_at(320, 180)], SHAPE, t=0.1).distance_m == 3.0


def test_calibration_rescales_with_resolution():
    k = Intrinsics.from_fov(1280, 720, 66.0).scaled_to(640, 360)
    assert k.fx == pytest.approx(cam().fx) and k.cx == 320
