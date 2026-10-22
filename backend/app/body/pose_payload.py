"""One bounded estimator snapshot shared by both body consumers; no model or IO."""
from dataclasses import dataclass
import math
from numbers import Real
import numpy as np


@dataclass(frozen=True)
class PosePayload:
    visible: bool = False
    keypoints: tuple = ()
    quality: float = 0.0
    reason: str | None = "invalid_pose_payload"


def validate_pose(pose):
    if type(pose) is PosePayload:
        # Exact instances can be hand-built by internal callers too. Keep only
        # canonical empty refusals; visible payloads take the bounded validator.
        if (pose.visible is False and type(pose.keypoints) is tuple and not pose.keypoints
                and type(pose.quality) is float and pose.quality == 0.0
                and pose.reason in ("invalid_pose_payload", "no_pose_model", "pose_not_visible", "internal_error")):
            return pose
        pose = {"visible": pose.visible, "keypoints": pose.keypoints, "quality": pose.quality}
    try:
        if not isinstance(pose, dict):
            return PosePayload()
        available = pose.get("estimator_available", True)
        visible = pose.get("visible")
        if available is False:
            return PosePayload(reason="no_pose_model")
        if available is not True:
            return PosePayload()
        if visible is False:
            return PosePayload(reason="pose_not_visible")
        if visible is not True:
            return PosePayload()
        raw, quality = pose.get("keypoints", ()), pose.get("quality")
        if type(raw) in (tuple, list) and len(raw) == 0:
            return PosePayload(reason="pose_not_visible")
        if (isinstance(quality, bool) or not isinstance(quality, Real)
                or not math.isfinite(quality) or not 0 <= quality <= 1):
            return PosePayload()
        if quality <= 0.3:
            return PosePayload(reason="pose_not_visible")
        # Check ndarray shape/dtype BEFORE any conversion: at most 99 numeric values.
        if type(raw) is np.ndarray:
            if raw.shape != (33, 3) or raw.dtype.kind not in "fiu":
                return PosePayload()
            raw = raw.tolist()
        # Check length before traversing: never copy/consume an oversized point list.
        if type(raw) not in (tuple, list) or len(raw) != 33:
            return PosePayload()
        points = []
        for point in raw:
            if type(point) not in (tuple, list) or len(point) != 3:
                return PosePayload()
            if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in point):
                return PosePayload()
            x, y, visibility = map(float, point)
            # Structural overflow guard, not a cue threshold or image-position validity claim.
            if abs(x) > 1_000_000 or abs(y) > 1_000_000 or not 0 <= visibility <= 1:
                return PosePayload()
            points.append((x, y, visibility))
        return PosePayload(True, tuple(points), float(quality), None)
    except Exception:
        return PosePayload()
