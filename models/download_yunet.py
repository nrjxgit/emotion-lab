"""Download YuNet face detector (Apache-2.0, opencv_zoo)."""

import urllib.request, os

URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]


def main(dest="models/face_detection_yunet_2023mar.onnx"):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 100000:
        print("exists", dest)
        return dest
    for u in URLS:
        try:
            print("trying", u)
            urllib.request.urlretrieve(u, dest)
            if os.path.getsize(dest) > 100000:
                print("saved", dest)
                return dest
        except Exception as e:
            print("failed:", e)
    raise SystemExit("YuNet download failed - check network")


if __name__ == "__main__":
    main()
