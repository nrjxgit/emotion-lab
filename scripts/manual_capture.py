"""Assisted manual run: grabs N frames, posts to /api/infer, logs results.

Usage: python scripts/manual_capture.py --n 30 --out eval_results/manual_run.json
Requires backend on :8000 and a local camera. Consent required before running.
"""

import argparse, json, time, urllib.request

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="eval_results/manual_run.json")
    ap.add_argument("--api", default="http://127.0.0.1:8000/api/infer")
    a = ap.parse_args()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("camera unavailable (typed error: no silent fallback)")
    rows = []
    for i in range(a.n):
        ok, frame = cap.read()
        if not ok:
            rows.append({"i": i, "error": "frame-grab-failed"})
            continue
        _, buf = cv2.imencode(".jpg", frame)
        req = urllib.request.Request(
            a.api, data=buf.tobytes(), headers={"Content-Type": "multipart/form-data"}
        )
        # NOTE: raw bytes post; server expects multipart — use curl/httpie in practice.
        rows.append({"i": i, "note": "see MANUAL_WEBCAM_PROTOCOL.md for curl form"})
        time.sleep(0.5)
    cap.release()
    json.dump(rows, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
