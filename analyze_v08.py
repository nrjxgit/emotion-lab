import sys, json
import pathlib
import numpy as np

sys.path.insert(0, "training")
from compare_clips import paired_stats, actor_split, audit_clips
from cremad_science import score_six
from calibrate_temp import apply_temperature, ece_from_proba, fit_temperature

OUT = "eval_results/v0.8"
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
man = json.load(open("eval_results/v0.7/cremad/manifest.json"))
clips = [{"file": c, "actor": c[:4]} for c in man["clips"]]
audit = audit_clips(clips, train_files=set(), selected_actors=set(man["actors"]))
print("audit:", {k: (len(v) if isinstance(v, list) else v) for k, v in audit.items()})

rows = []
for c in man["clips"]:
    p = pathlib.Path("eval_results/v0.7/cremad/clips") / (
        c.replace(".flv", "") + ".json"
    )
    rows.append(json.loads(p.read_text()))
emo = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "SAD": 4, "NEU": 6}
y = np.array([emo[r["file"].split("_")[2]] for r in rows])
pv = [
    np.mean([f["vit"] for f in r["frames"]], axis=0) if r["frames"] else np.zeros(7)
    for r in rows
]


# frames store dicts? handle both
def get_probs(r, key):
    fr = r["frames"]
    if fr and isinstance(fr[0], dict):
        return np.mean([f[key] for f in fr], axis=0)
    return np.mean(fr, axis=0) if len(fr) else np.zeros(7)


Mm = np.array([get_probs(r, "mini") for r in rows])
Mv = np.array([get_probs(r, "vit") for r in rows])
det = np.array([len(r["frames"]) > 0 for r in rows])
print("clips with frames:", det.sum(), "/", len(rows))
ps = paired_stats(y[det], Mm[det].argmax(1), Mv[det].argmax(1))
print(
    "paired:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ps.items()}
)
for name, P in (("mini", Mm), ("vit", Mv)):
    s = score_six(y[det], P[det], det[det])
    print(
        name,
        "acc:",
        round(s["accuracy"], 4),
        "F1:",
        round(s["macro_f1_six"], 4),
        "recall:",
        [round(v, 3) for v in s["recall"]],
        "surprise_preds:",
        s["surprise_predictions"],
    )
actors = sorted({r["file"][:4] for r, d in zip(rows, det) if d})
cal_a, tst_a = actor_split(actors, n_cal=30)
tr = np.array([r["file"][:4] in cal_a for r in rows])
cf = np.array([r["file"][:4] in tst_a for r in rows])
T = fit_temperature(np.log(np.clip(Mm[det & tr], 1e-12, 1)), y[det & tr])
Pcal = apply_temperature(np.log(np.clip(Mm[det & cf], 1e-12, 1)), T)
print(
    "actor-split T:",
    round(T, 3),
    "test ECE:",
    round(ece_from_proba(Mm[det & cf], y[det & cf]), 4),
    "->",
    round(ece_from_proba(Pcal, y[det & cf]), 4),
)
json.dump(
    {
        "audit": audit,
        "paired": ps,
        "T_actor_split": T,
        "n_cal_actors": len(cal_a),
        "n_test_actors": len(tst_a),
    },
    open(f"{OUT}/paired_calibration.json", "w"),
    indent=2,
    default=lambda o: o.tolist() if hasattr(o, "tolist") else o,
)
print("wrote", OUT + "/paired_calibration.json")
