# Emotion Lab

Emotion Lab is a research prototype that runs on your own computer. It looks at a webcam image, estimates which **facial-expression pattern** it sees, shows every probability, and is allowed to answer **"not enough evidence"** instead of guessing.

It was built for my TKS Replicate project. This repository is the mini-Xception version whose results are reported in the article; the screenshots in the article come from a separate dashboard version that uses HSEmotion. The full story, including what went wrong, is in my article: [What I Learned Building Emotion Lab, an Emotion AI That Admits When It’s Not Sure](https://medium.com/@achunrjneeraj/what-i-learned-building-emotion-lab-an-emotion-ai-that-admits-when-its-not-sure-8adbc57bb1f1).

> **What it is not:** Emotion Lab does not read feelings. It recognises patterns in faces that strangers have labelled with emotion words. A yawn can look like "surprise", and a scowl can mean concentration. Treat every output as a guess about an expression, never as a fact about how someone feels.

## How it was built

I didn't write the code myself. AI coding agents (OpenCode, Claude, Codex and ChatGPT), working across several sessions, wrote it from my instructions. I chose the research it is based on, designed how it should work, tested every version, sent feedback, and made the decisions about which results to trust, which model to use by default, and which data could be used.

## What it does

- **Face detection:** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) finds the face in each frame.
- **Expression model:** [mini-Xception](https://github.com/oarriaga/face_classification) (trained on FER2013) scores seven expression patterns. A vision transformer can be selected as an experimental alternative.
- **Honest output:** every probability is shown, with a confidence level. Below a chosen confidence threshold, it reports "insufficient evidence".
- **Local only:** frames are processed on your machine. Nothing is uploaded or stored by default.
- **Research extras (experimental):** upper-body movement cues and opt-in personal calibration. Neither has been validated on real people.

## Quick start

```bash
pip install -r requirements.txt
pip install python-multipart httpx
python models/download_weights.py   # mini-Xception weights
python models/download_yunet.py     # YuNet face detector
python -m pytest -q                 # optional: run the tests
```

Start the backend (Windows: double-click `run_backend.bat`), or:

```bash
PYTHONPATH=backend/app uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir backend/app
```

Then open `frontend-standalone/index.html` in a browser and press **Start webcam**. The full dashboard is in `frontend/` (`npm install`, then `npm run dev`). See [docs/SETUP.md](docs/SETUP.md) for details.

## Results

A summary is in [RESULTS.md](RESULTS.md), and the full reports for each version are in [eval_results/](eval_results/). In short:

| Test | Result |
|---|---|
| FER2013 test set, original version (7,178 faces) | 56.0% accuracy |
| Fixing preprocessing (same 350 faces) | 49.7% → 65.7% |
| Answering only when ≥50% sure (FER2013) | answers 59.7% of faces, 69.3% correct |
| CREMA-D actors, unfamiliar dataset (546 clips) | 29.7% accuracy; two models agreed on only 47.8% |

None of these are results on real-world, everyday emotions. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Data

No datasets are included. FER2013, CREMA-D and RAVDESS each have their own licence and must be obtained from their original sources. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Responsible use

Please read [docs/ETHICS.md](docs/ETHICS.md) before running it on anyone. In short:

- Only analyse people who have agreed to it, with a visible recording indicator. Get a guardian's consent for anyone under 18.
- Never use it to make decisions about people at work, school, in healthcare or in policing. The EU AI Act bans emotion recognition in workplaces and schools except for medical or safety reasons.
- Do not use it to monitor, judge or profile anyone.

## Licence

The code is released under the MIT Licence (see [LICENSE](LICENSE)). Models, weights and datasets downloaded by the scripts keep their own licences.
