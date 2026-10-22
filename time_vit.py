import sys, time
import numpy as np

sys.path.insert(0, "backend/app/face")
sys.path.insert(0, "training")
from fer_images import load_split

X, y, _ = load_split("eval_data/fer2013/test")
print("timing ViT on 10 images ...")
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

proc = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained(
    "dima806/facial_emotions_image_detection"
).eval()
t0 = time.time()
with torch.no_grad():
    for im in X[:10]:
        inp = proc(images=np.stack([im] * 3, -1), return_tensors="pt")
        model(**inp)
dt = (time.time() - t0) / 10
print(f"{dt:.2f} s/img -> full 7178 ~= {dt * 7178 / 3600:.1f} h")
