import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app" / "face"))


def neutral_pts():
    return {
        "brow_inner_L": (0.42, 0.35),
        "brow_inner_R": (0.58, 0.35),
        "brow_outer_L": (0.35, 0.34),
        "brow_outer_R": (0.65, 0.34),
        "eye_top_L": (0.42, 0.41),
        "eye_bot_L": (0.42, 0.44),
        "eye_top_R": (0.58, 0.41),
        "eye_bot_R": (0.58, 0.44),
        "nose_tip": (0.5, 0.52),
        "nose_L": (0.472, 0.52),
        "nose_R": (0.528, 0.52),
        "mouth_L": (0.42, 0.66),
        "mouth_R": (0.58, 0.66),
        "mouth_top": (0.5, 0.645),
        "mouth_bot": (0.5, 0.675),
        "chin": (0.5, 0.85),
        "cheek_L": (0.3, 0.55),
        "cheek_R": (0.7, 0.55),
    }


def test_smile_gives_au12():
    from aus import estimate_aus

    pts = neutral_pts()
    pts["mouth_L"] = (0.38, 0.65)
    pts["mouth_R"] = (0.62, 0.65)
    aus = {a["code"]: a for a in estimate_aus(pts)}
    assert aus["AU12"]["intensity"] > 0.5


def test_brow_lower_gives_au04():
    from aus import estimate_aus

    pts = neutral_pts()
    pts["brow_inner_L"] = (0.42, 0.39)
    pts["brow_inner_R"] = (0.58, 0.39)
    aus = {a["code"]: a for a in estimate_aus(pts)}
    assert aus["AU04"]["intensity"] > 0.5


def test_geometry_ratios_bounded():
    from aus import measure_geometry

    g = measure_geometry(neutral_pts())
    for k in ("nose_width_ratio", "mouth_width_ratio"):
        assert 0.0 < g[k] < 2.0
