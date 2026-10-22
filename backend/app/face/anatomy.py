from .aus import measure_geometry as _mg


def measure_anatomy(points: dict) -> dict:
    g = _mg(points)
    g["notes"] = (
        "normalized ratios, not labels; use for hypothesis tests with confounders"
    )
    return g
