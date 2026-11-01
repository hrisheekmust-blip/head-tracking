#!/usr/bin/env python3
"""Download the YuNet face detector models (~230 KB each) from OpenCV's model zoo into models/.
Run once after cloning:  python3 download_models.py
"""

import hashlib
import os
import urllib.request

BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
MODELS = {  # file name -> start of its sha256, to catch a bad download
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4",  # OpenCV 4.8 - 4.x
    "face_detection_yunet_2026may.onnx": "ebafce4e3c118d65",  # OpenCV 5.x
}

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(here, exist_ok=True)
for name, sha in MODELS.items():
    path = os.path.join(here, name)
    if not os.path.isfile(path):
        print(f"downloading {name}")
        urllib.request.urlretrieve(BASE + name, path)
    with open(path, "rb") as fh:
        ok = hashlib.sha256(fh.read()).hexdigest().startswith(sha)
    print(f"{name}: {'ok' if ok else 'CHECKSUM MISMATCH, delete it and re-run'}")
