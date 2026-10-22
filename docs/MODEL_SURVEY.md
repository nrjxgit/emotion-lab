# Stronger FER models — survey (2026-10-16, integration candidates)

All need torch + labeled data; none integrated yet. Selection rule: public code
AND downloadable weights AND same-7-class (or convertible) head, compared on
the SAME dataset + preprocessing as mini-Xception.

| Model | Why | Code/weights (verify at integration) |
|---|---|---|
| EAC (ECCV'22, Zhao et al.) | SOTA on RAF-DB/AffectNet; erasing-attention, robust to occlusion | GitHub `Our309/EAC` reported public |
| POSTER / POSTER++ (pyramid + landmark streams) | top AffectNet/RAF-DB numbers; uses landmarks (fits our pipeline) | GitHub `TouchCtrl/POSTER-DAN` family reported public |
| DAN — Distract Your Attention (Wen et al.) | strong AffectNet-7/8 + RAF-DB; published confusion matrices to compare against | author repo reported public |
| DDAMFN++ | lightweight, good cross-dataset | reported public |

Integration plan: `training/compare_models.py` (to build) loads each ONNX/torch
checkpoint, runs `benchmark()` on the same labeled rows, reports ΔmacF1 + ECE.
No comparison claimed until then. mini-Xception stays the deployed baseline.
