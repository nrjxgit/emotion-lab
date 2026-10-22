"""Paired same-clip model comparison, actor-disjoint splits, manifest audits."""


def paired_stats(y_true, pred_a, pred_b) -> dict:
    y_true = list(y_true)
    a = list(pred_a)
    b = list(pred_b)
    assert len(y_true) == len(a) == len(b) and y_true
    ca = [int(x == y) for x, y in zip(a, y_true)]
    cb = [int(x == y) for x, y in zip(b, y_true)]
    return {
        "n": len(y_true),
        "acc_a": sum(ca) / len(ca),
        "acc_b": sum(cb) / len(cb),
        "agreement": sum(int(x == y) for x, y in zip(a, b)) / len(a),
        "both_correct": sum(int(x and y) for x, y in zip(ca, cb)),
        "both_wrong": sum(int(not x and not y) for x, y in zip(ca, cb)),
        "a_only": sum(int(x and not y) for x, y in zip(ca, cb)),
        "b_only": sum(int((not x) and y) for x, y in zip(ca, cb)),
    }


def actor_split(actors: list, n_cal: int, seed: int = 505) -> tuple:
    import hashlib

    ordered = sorted(
        actors, key=lambda a: hashlib.sha256(f"{seed}:{a}".encode()).hexdigest()
    )
    return ordered[:n_cal], ordered[n_cal:]


def audit_clips(clips: list, train_files: set, selected_actors: set) -> dict:
    seen, dups = set(), []
    for c in clips:
        if c["file"] in seen:
            dups.append(c["file"])
        seen.add(c["file"])
    in_train = [c["file"] for c in clips if c["file"] in train_files]
    outside = [c["actor"] for c in clips if c["actor"] not in selected_actors]
    return {
        "n": len(clips),
        "n_unique": len(seen),
        "duplicates": sorted(set(dups)),
        "also_in_train": in_train,
        "outside_selected_actors": sorted(set(outside)),
    }
