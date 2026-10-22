from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import math
import threading
import asyncio
from typing import Literal

from face.detector import detect_faces, model_available as yunet_ok
from face.landmarks import get_landmarks
from face.aus import estimate_aus, measure_geometry
from face.expression import ExpressionPredictor, observable_feature_analysis, CLASSES
from body.pose import estimate_pose
from body.pose_payload import PosePayload, validate_pose
from upload_policy import MAX_UPLOAD_BYTES, MAX_IMAGE_PIXELS, MAX_IMAGE_DIMENSION, check_image_header
from body.movement import movement_features, predict_body_emotion
from fusion.fuse import fuse

app = FastAPI(title="Emotion Recognition Research Platform")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
_predictors = {}


def get_predictor(profile: str):
    from face.expression import ExpressionPredictor

    if profile not in ExpressionPredictor.PROFILES:
        raise HTTPException(
            status_code=422,
            detail=f"unknown profile {profile!r}; choices {list(ExpressionPredictor.PROFILES)}",
        )
    if profile not in _predictors:
        _predictors[profile] = ExpressionPredictor(profile=profile)
    return _predictors[profile]


_track: list = []  # compatibility name only; no request writes or reads this former shared history
MAX_INFER_REQUESTS = 2
_INFER_SLOTS = threading.BoundedSemaphore(MAX_INFER_REQUESTS)
MAX_UPLOAD_REQUESTS = 4
UPLOAD_TIMEOUT_S = 5.0
_UPLOAD_SLOTS = threading.BoundedSemaphore(MAX_UPLOAD_REQUESTS)
_research_adapters: dict = {}

ETHICS = (
    "Apparent-expression hypothesis only — not proof of internal feeling. "
    "Do not use for employment/school/healthcare decisions. Local-first; no retention without consent."
)


@app.get("/api/health")
def health():
    def has(m):
        try:
            __import__(m)
            return True
        except Exception:
            return False

    return {
        "status": "ok",
        "mediapipe": has("mediapipe"),
        "torch": has("torch"),
        "onnxruntime": has("onnxruntime"),
        "yunet": yunet_ok(),
    }


@app.get("/api/model-info")
def model_info():
    try:
        from face.fer_model import weights_available, FER7

        fer = {
            "loaded": get_predictor("corrected").source
            == ExpressionPredictor.TRAINED_SOURCE,
            "source": get_predictor("corrected").source,
            "default_profile": "corrected (YuNet margin25 crop, [-1,1] upstream normalization)",
            "legacy_profile": "legacy ([0,1], tight box) via ?profile=legacy",
            "classes7": FER7 if weights_available() else [],
            "weights": "models/fer_mini_xception_h5.hdf5 (oarriaga face_classification; research use)",
            "licence": "oarriaga/face_classification repository MIT; source-image rights separate (see docs/LICENCE_CORRECTIONS_v0.5.md)",
        }
    except Exception as e:
        fer = {"loaded": False, "error": str(e)}
    return {
        "face_detector": {
            "name": "YuNet-2023mar",
            "available": yunet_ok(),
            "licence": "Apache-2.0 (opencv_zoo)",
            "landmarks": "5pt",
        },
        "expression": fer,
        "body": {
            "name": "motion-energy",
            "type": "motion-only",
            "note": "No licensed body-emotion weights integrated; movement tracked, emotion NOT predicted from body",
        },
        "fusion": {"mode": "late_quality_weighted"},
    }


@app.get("/api/models")
def available_models():
    from face.adapters import VIT_REVISION

    return {
        "default": "mini-xception",
        "models": {
            "mini-xception": {
                "available": True,
                "experimental": False,
                "profile": "corrected (margin25, [-1,1]); legacy via ?profile=legacy",
            },
            "vit": {
                "available": True,
                "experimental": True,
                "revision": VIT_REVISION,
                "warning": "possible FER training overlap; opt-in only",
            },
            "hsemotion": {
                "available": False,
                "experimental": True,
                "warning": "separate implementation; no merge performed",
            },
        },
        "confidence_status": "raw neural-network scores, not calibrated personal feelings",
    }


@app.post("/api/infer")
async def infer(
    image: UploadFile = File(...),
    model: Literal["mini-xception", "vit", "hsemotion"] = "mini-xception",
    profile: str = "corrected",
    calibrated: bool = False,
    abstain: float | None = None,
    socialcue: str = "0",
    x_socialcue_session: str | None = Header(default=None),
):
    if model == "hsemotion":
        raise HTTPException(
            status_code=501, detail="HSEmotion not integrated; separate app preserved"
        )
    if calibrated and (model != "mini-xception" or profile != "corrected"):
        raise HTTPException(
            status_code=422,
            detail="calibration validated only for mini-xception + corrected profile",
        )
    if abstain is not None and abstain not in (0.5, 0.7, 0.9, 0.95):
        raise HTTPException(
            status_code=422,
            detail="abstain threshold must be a validated value: 0.5, 0.7, 0.9, 0.95",
        )
    import socialcue_session
    store = socialcue_session.STORE
    # Only the upload budget and a read-only generation snapshot cross this await.
    if not _UPLOAD_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="upload busy; retry later")
    try:
        captured = store.capture_generation(x_socialcue_session)
        try:
            data = await asyncio.wait_for(image.read(MAX_UPLOAD_BYTES + 1), timeout=UPLOAD_TIMEOUT_S)
        except TimeoutError:
            raise HTTPException(status_code=408, detail="image upload read timed out")
    finally:
        _UPLOAD_SLOTS.release()
    if not data or len(data) < 100:
        raise HTTPException(status_code=422, detail="empty or truncated upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds 5 MB limit")
    # Busy-slot 503 never creates or refreshes a generation entry. CPU slots exclude upload waits.
    if not _INFER_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="inference busy; retry later")
    try:
        check_image_header(data)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=422, detail="undecodable image (not JPEG/PNG)")
        height, width = img.shape[:2]
        if width * height > MAX_IMAGE_PIXELS or max(width, height) > MAX_IMAGE_DIMENSION:
            raise HTTPException(status_code=413, detail="decoded image exceeds pixel/dimension limit")
        # Profile validation must not mint tokens on a rejected request. Resolve
        # once here and pass the predictor through, preserving default responses.
        predictor = get_predictor(profile) if model == "mini-xception" else None
        try:
            generation = store.admit(x_socialcue_session, captured=captured)
        except socialcue_session.SessionInvalidated:
            raise HTTPException(status_code=409, detail="session invalidated during upload")
        # CPU work remains synchronous on this event-loop thread: explicit prototype limit.
        result = _infer_image(img, model, profile, calibrated, abstain, socialcue,
                              x_socialcue_session, store, generation, predictor=predictor)
        if generation is not None and not store.generation_current(x_socialcue_session, generation):
            raise HTTPException(status_code=409, detail="session invalidated during inference")
        return result
    finally:
        _INFER_SLOTS.release()


def _infer_image(img, model, profile, calibrated, abstain, socialcue, x_socialcue_session, store, generation, *, predictor=None):
    try:
        faces = detect_faces(img)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    H, W = img.shape[:2]

    def _usable(b) -> bool:
        # YuNet boxes may overflow the frame; keep only boxes with a non-empty
        # intersection so downstream crops can never be empty (else 500).
        x0, y0 = max(0, b["x"]), max(0, b["y"])
        return x0 < min(W, b["x"] + b["w"]) and y0 < min(H, b["y"] + b["h"])

    faces = [b for b in faces if _usable(b)]
    face_out = None
    fq = 0.0
    if faces:
        box = max(faces, key=lambda b: b["w"] * b["h"])
        pts = get_landmarks(img, box)
        aus = estimate_aus(pts)
        geom = measure_geometry(pts)
        x, y, w, h = (max(0, box["x"]), max(0, box["y"]), box["w"], box["h"])
        if model == "mini-xception" and profile == "corrected":
            from face.preprocessing import crop_pixels

            crop = crop_pixels(img, box, margin=0.25)
        else:
            crop = img[y : y + h, x : x + w]
        face_res = (
            predictor.predict(crop, aus, calibrated=calibrated)
            if predictor is not None
            else get_predictor("legacy").predict(crop, aus)
        )
        face_res = dict(face_res)
        face_res["profile"] = profile if model == "mini-xception" else "n/a"
        if model == "vit":
            from face.adapters import get_adapter, FER7

            try:
                if model not in _research_adapters:
                    _research_adapters[model] = get_adapter(model)
                adapter = _research_adapters[model]
                p7 = adapter.predict_proba(crop)
            except Exception as e:
                raise HTTPException(
                    status_code=503, detail=f"requested ViT unavailable: {e}"
                )
            class_map = {"angry": "anger", "happy": "happiness", "sad": "sadness"}
            probs = {c: 0.0 for c in CLASSES}
            for c, p in zip(FER7, p7):
                probs[class_map.get(c, c)] = float(p)
            face_res = {
                "probs": probs,
                "source": adapter.source,
                "experimental": True,
                "revision": adapter.revision,
                "confidence": float(p7.max()),
                "uncertainty": float(1 - p7.max()),
                "calibrated": False,
                "valence": None,
                "arousal": None,
                "expressiveness": None,
                "warning": "apparent-expression scores; possible training overlap",
            }
        fq = float(box.get("confidence", 0.5))
        from fusion.evidence import evidence_status

        ev = evidence_status(True, fq, face_res.get("confidence", 0.0))
        abstained = False
        if abstain is not None:
            abstained = face_res.get("confidence", 0.0) < abstain
            if abstained and "below-abstention-threshold" not in ev["reasons"]:
                ev["reasons"] = ev["reasons"] + ["below-abstention-threshold"]
                ev["insufficient"] = True
        face_out = {
            "face_found": True,
            "box": {k: box[k] for k in ("x", "y", "w", "h")},
            "aus": aus,
            "expression": face_res,
            "observable_analysis": observable_feature_analysis(aus),
            "geometry": geom,
            "quality": fq,
            "insufficient_evidence": ev,
            "abstained": abstained,
            "abstain_threshold": abstain,
        }
    else:
        from fusion.evidence import evidence_status

        face_out = {
            "face_found": False,
            "box": None,
            "aus": [],
            "expression": {
                "probs": {c: (1.0 if c == "unknown" else 0.0) for c in CLASSES},
                "confidence": 0.15,
                "uncertainty": 0.85,
                "source": get_predictor("corrected").source
                if model == "mini-xception"
                else "hf-vit-dima806-not-run-no-face",
                "note": "no face detected; not an emotion reading",
            },
            "quality": 0.0,
            "insufficient_evidence": evidence_status(False, 0.0, 0.0),
        }
    body_res = None
    bq = 0.0
    try:
        pose = validate_pose(estimate_pose(img))  # bounded ONCE, before either body consumer
    except Exception:
        pose = PosePayload(reason="internal_error")
    try:
        points, identity = None, None
        if pose.visible:  # legacy movement is session-keyed, independent of face count
            points = tuple((x, y) for x, y, _ in pose.keypoints)
            left, right = pose.keypoints[11], pose.keypoints[12]
            scale = math.hypot((left[0] - right[0]) * W, (left[1] - right[1]) * H)
            if scale > 0:
                identity = ((W, H), scale, (left[0] + right[0]) * W / 2, (left[1] + right[1]) * H / 2)
        track = store.movement(x_socialcue_session, generation, points, identity)
        if points is not None:
            feats = movement_features(track if len(track) >= 5 else [])
            body_res = predict_body_emotion(feats)
            bq = pose.quality
    except Exception as exc:
        import socialcue_session
        if isinstance(exc, socialcue_session.SessionInvalidated):
            raise HTTPException(status_code=409, detail="session invalidated during inference")
        pose = PosePayload(reason="internal_error")
        body_res, bq = None, 0.0
        try:
            store.movement(x_socialcue_session, generation, None)
        except socialcue_session.SessionInvalidated:
            raise HTTPException(status_code=409, detail="session invalidated during inference")
    fused = fuse(
        {
            "probs": face_out["expression"]["probs"],
            "confidence": face_out["expression"]["confidence"],
            "uncertainty": face_out["expression"]["uncertainty"],
        }
        if faces
        else None,
        (
            {
                "probs": body_res["probs"],
                "confidence": body_res["confidence"],
                "uncertainty": body_res["uncertainty"],
            }
            if body_res
            else None
        ),
        {"face_q": fq, "body_q": bq},
    )
    resp = {
        "selected_model": model,
        "face": face_out,
        "body": body_res
        or {"visible": False, "type": "motion-only", "uncertainty": 0.9},
        "fusion": fused,
        "ethics_notice": ETHICS,
    }
    # Only the exact value "1" enables the research block; any other or malformed value behaves as 0
    # (never a 422), so the default path is unchanged for every existing caller.
    if socialcue == "1":
        # Additive research block; any failure inside returns insufficient/internal_error.
        try:
            import socialcue_session

            resp["socialcue"] = socialcue_session.socialcue_block(
                x_socialcue_session, faces, box if faces else None, face_out, fq,
                pose=pose, width=img.shape[1], height=img.shape[0], store=store, generation=generation
            )
        except Exception:
            try:
                import socialcue_session

                resp["socialcue"] = socialcue_session.insufficient_block("internal_error")
            except Exception:  # last resort when even the block builder cannot be used
                try:
                    from socialcue_session import BANNER as _cue_banner
                except Exception:
                    _cue_banner = "Social cues (research prototype): not a reading of feelings; hypothesis heuristic (HB-1), not validated."
                resp["socialcue"] = {
                    "states": [],
                    "insufficient": {"flag": True, "reasons": ["internal_error"]},
                    "window": {"seconds": 3.0, "usable_frames": 0, "fps": 0.0},
                    "heuristic": "HB-1 hypothesis, unvalidated",
                    "proxy_eval_ref": None,
                    "banner": _cue_banner,
                }
    return resp


@app.delete("/api/socialcue/session")
def socialcue_session_clear(x_socialcue_session: str | None = Header(default=None)):
    """Drop one in-memory social-cue session (id from the X-Socialcue-Session header).

    Unauthenticated: the id is a bearer secret (single-user local prototype; see socialcue_session.py).
    """
    try:
        import socialcue_session

        cleared = bool(x_socialcue_session) and socialcue_session.STORE.clear(x_socialcue_session)
    except Exception:
        cleared = False
    return {"ok": True, "cleared": cleared}


@app.post("/api/calibrate")
async def calibrate(payload: dict):
    from calibration.baseline import PersonalBaseline

    if not payload.get("consent"):
        raise HTTPException(
            status_code=422,
            detail="explicit opt-in consent required to store personal baseline data",
        )
    b = PersonalBaseline()
    if "aus" in payload:
        b.update(payload["aus"])
    return {"ok": True, "baseline": b.data}


_sessions: dict = {}


@app.post("/api/session/start")
async def session_start(payload: dict):
    from calibration.sessions import CalibrationSession

    if not payload.get("consent"):
        raise HTTPException(status_code=422, detail="explicit opt-in consent required")
    s = CalibrationSession()
    s.start(consent=True, participant=payload.get("participant", "anon"))
    _sessions[payload.get("participant", "anon")] = s
    return {"ok": True}


@app.post("/api/session/record")
async def session_record(payload: dict):
    s = _sessions.get(payload.get("participant", "anon"))
    if s is None:
        raise HTTPException(status_code=422, detail="no active session")
    return s.record(payload["ai_label"], payload["self_report"], payload.get("probs"))


@app.get("/api/session/report")
def session_report(participant: str = "anon"):
    s = _sessions.get(participant)
    if s is None:
        raise HTTPException(status_code=422, detail="no active session")
    return s.report()


@app.delete("/api/session")
def session_delete(participant: str = "anon"):
    from calibration.sessions import CalibrationSession

    s = _sessions.pop(participant, None)
    if s is None:
        raise HTTPException(status_code=422, detail="no active session")
    store = getattr(s, "store", "configs/calibration_sessions.json")
    try:
        import json
        import os

        if os.path.exists(store):
            rows = json.load(open(store))
            rows = [r for r in rows if r.get("participant") != participant]
            json.dump(rows, open(store, "w"))
    except Exception:
        pass
    return {"ok": True, "deleted": participant}


@app.get("/api/anatomy/stats")
def anatomy_stats():
    return {
        "hypothesis": "nose_width_ratio vs AU09 intensity",
        "method": "Pearson r with 95% CI; control for camera angle, age, lighting; never causal",
        "status": "awaiting labeled data — use training/anatomy_study.py",
    }


@app.get("/api/fairness/report")
def fairness():
    return {
        "metrics": ["macroF1", "per-class PR", "calibration-error", "abstention-rate"],
        "slices": ["age", "gender", "skin-tone", "lighting"],
        "status": "run training/evaluate.py",
    }
