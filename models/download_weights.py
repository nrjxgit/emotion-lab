"""Download genuine pretrained weights (run manually; never fake)."""

import urllib.request, os

URLS = {
    # oarriaga FER2013 mini-Xception (real trained, ~66% FER2013). Convert to ONNX separately.
    "fer_mini_xception_h5": "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5",
}


def main():
    os.makedirs("models", exist_ok=True)
    for k, u in URLS.items():
        dest = f"models/{k}.hdf5"
        if os.path.exists(dest):
            print(f"exists {dest}")
            continue
        print(f"downloading {u} ...")
        urllib.request.urlretrieve(u, dest)
        print(f"saved {dest}")


if __name__ == "__main__":
    main()
