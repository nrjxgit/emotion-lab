# Third-party models and datasets

None of these are included in this repository. Each keeps its own licence; check it before you download or use it.

## Models (downloaded by scripts in `models/`)

| Model | Source | Notes |
|---|---|---|
| mini-Xception (FER2013) | [oarriaga/face_classification](https://github.com/oarriaga/face_classification) | Arriaga, Valdenegro-Toro and Plöger, [arXiv:1710.07557](https://arxiv.org/abs/1710.07557) |
| YuNet face detector | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | |
| ViT facial emotions (optional, experimental) | [dima806/facial_emotions_image_detection](https://huggingface.co/dima806/facial_emotions_image_detection) | Training data not confirmed to be separate from the FER2013 test set, so its scores are not trusted |

## Datasets used in evaluation

| Dataset | Source | Licence |
|---|---|---|
| FER2013 | [Goodfellow et al., 2013](https://arxiv.org/abs/1307.0414), via Kaggle | Research use under Kaggle's terms; do not redistribute |
| CREMA-D | [Cao et al., 2014](https://github.com/CheyneyComputerScience/CREMA-D) | Open Database Licence (ODbL) and Database Contents Licence |
| RAVDESS (planned) | [Livingstone and Russo, 2018](https://zenodo.org/records/1188976) | CC BY-NC-SA 4.0 (non-commercial) |
