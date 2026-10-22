# Setup
```powershell
cd emotion-recognition-system
pip install -r requirements.txt
pip install python-multipart httpx
python -m pytest -q
$env:PYTHONPATH="backend/app"
uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir backend/app
# open frontend-standalone/index.html → Start webcam
```
```cmd
:: npm MUST run via cmd (PowerShell blocks npm.ps1 under default ExecutionPolicy):
cd frontend && npm install && npm run build
```
- Optional real weights: `python models/download_weights.py` (oarriaga FER hdf5, convert to ONNX) or set HF `dima806/facial_emotions_image_detection` with torch+transformers.
- Export: each of `face/`, `body/`, `fusion/`, `training/` imports standalone with `predict`/`predict_proba` signatures.
