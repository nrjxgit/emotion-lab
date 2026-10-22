"""Webcam capture helper with typed errors (no silent fake frames)."""

import cv2


class WebcamError(Exception):
    pass


def open_capture(index: int = 0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise WebcamError(
            f"camera {index} unavailable (in use, missing, or no permission)"
        )
    return cap


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise WebcamError("frame grab failed (camera disconnected or driver error)")
    return frame
