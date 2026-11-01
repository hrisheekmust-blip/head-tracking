#!/usr/bin/env python3
"""Camera calibration with a printed checkerboard (needed for the 5 cm height target).

Live (SPACE = capture, C = calibrate and save, Q = quit):
    python3 calibrate.py --source picam --out calib/robot1.json --square 0.025
From saved photos:
    python3 calibrate.py --images "calib_imgs/*.jpg" --out calib/robot1.json --square 0.025

--cols/--rows count INNER corners. Measure one square with calipers for --square (metres).
Use the same resolution and --lens-position as the tracker. Take ~20 shots at different
angles and distances, covering the corners of the frame. RMS under 0.5 px is good.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

from head_tracking.camera import open_camera


def find_corners(img, pattern):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(
        gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def calibrate_and_save(corner_sets, size, pattern, square, out, lens_position):
    if len(corner_sets) < 8:
        print(f"Need at least 8 good images, have {len(corner_sets)}")
        return
    board = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    rms, K, dist, _, _ = cv2.calibrateCamera([board] * len(corner_sets), corner_sets, size, None, None)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"width": size[0], "height": size[1], "camera_matrix": K.tolist(),
                   "dist_coeffs": dist.ravel().tolist(), "rms_px": rms,
                   "num_images": len(corner_sets), "lens_position": lens_position}, fh, indent=2)
    print(f"Saved {out}: RMS {rms:.3f} px from {len(corner_sets)} images"
          + ("  (high: retake blurry shots)" if rms > 1.0 else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="picam")
    p.add_argument("--images", help="Glob of saved photos instead of a live camera")
    p.add_argument("--out", required=True)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--square", type=float, required=True, help="Square size in metres")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--lens-position", type=float, default=0.5)
    a = p.parse_args()
    pattern = (a.cols, a.rows)

    if a.images:
        sets, size = [], None
        for path in sorted(glob.glob(a.images)):
            img = cv2.imread(path)
            if img is None:
                continue
            size = (img.shape[1], img.shape[0])
            c = find_corners(img, pattern)
            print(f"{os.path.basename(path)}: {'ok' if c is not None else 'board not found'}")
            if c is not None:
                sets.append(c)
        if size is None:
            sys.exit("No images found")
        calibrate_and_save(sets, size, pattern, a.square, a.out, a.lens_position)
        return

    cam = open_camera(a.source, a.width, a.height, lens_position=a.lens_position)
    os.makedirs("calib_imgs", exist_ok=True)
    sets = []
    print("SPACE = capture, C = calibrate & save, Q = quit")
    try:
        while (frame := cam.read()) is not None:
            c = find_corners(frame, pattern)
            vis = frame.copy()
            if c is not None:
                cv2.drawChessboardCorners(vis, pattern, c, True)
            cv2.putText(vis, f"captured {len(sets)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("calibrate", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and c is not None:
                sets.append(c)
                cv2.imwrite(f"calib_imgs/calib_{len(sets):02d}.jpg", frame)
            elif key == ord("c"):
                calibrate_and_save(sets, (frame.shape[1], frame.shape[0]), pattern,
                                   a.square, a.out, a.lens_position)
            elif key == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
