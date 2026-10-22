"""Live webcam spot-check: 5 frames, YuNet + FER, metrics only (no images saved).

Consent: run only with the informed consent of the person in front of the camera.
"""

import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "backend/app/face")
from detector import detect_faces
from fer_model import FER7, load_fer, preprocess, predict_fer

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("camera unavailable")
w = load_fer()
rows = []
for i in range(5):
    ok, frame = cap.read()
    if not ok:
        rows.append({"i": i, "error": "frame-grab-failed"})
        continue
    faces = detect_faces(frame)
    if not faces:
        rows.append({"i": i, "face_found": False})
    else:
        box = max(faces, key=lambda b: b["w"] * b["h"])
        x, y, bw, bh = (max(0, box["x"]), max(0, box["y"]), box["w"], box["h"])
        p = predict_fer(preprocess(frame[y : y + bh, x : x + bw]), w)
        top = int(np.argmax(p))
        rows.append(
            {
                "i": i,
                "face_found": True,
                "det_conf": round(box["confidence"], 3),
                "box_wh": [bw, bh],
                "top1": FER7[top],
                "top1_prob": round(float(p[top]), 3),
                "uncertainty": round(float(1 - p.max()), 3),
            }
        )
    time.sleep(0.5)
cap.release()
json.dump(rows, open("eval_results/webcam_spotcheck.json", "w"), indent=1)
for r in rows:
    print(r)
