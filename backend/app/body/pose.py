"""Pose: MediaPipe if present else stub returning not-visible (optical flow in frontend)."""


def estimate_pose(img):
    available = False
    try:
        import mediapipe as mp

        # lazy singleton
        global _pose
        try:
            _pose
        except NameError:
            _pose = None
        if _pose is None:
            _pose = mp.solutions.pose.Pose(static_image_mode=True)
        available = True
        import cv2

        res = _pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return {"visible": False, "keypoints": [], "quality": 0.0, "estimator_available": True}
        kps = [(l.x, l.y, l.visibility) for l in res.pose_landmarks.landmark]
        vis = sum(1 for _, _, v in kps if v > 0.5) / len(kps)
        return {"visible": vis > 0.3, "keypoints": kps, "quality": float(vis), "estimator_available": True}
    except Exception:
        return {"visible": False, "keypoints": [], "quality": 0.0, "estimator_available": available}
