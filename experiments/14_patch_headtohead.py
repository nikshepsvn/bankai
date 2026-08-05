"""Bankai Experiment 14: head-to-head patch comparison on one yardstick.

Experiment 6 reported 4 held-out fixes and Experiment 12 reported 5, but under
different probe sets and different metrics, so the two numbers cannot be compared
directly. This scores every patch on the *same* 30 held-out probes with the
*same* corrected metric, which is the only comparison that settles whether the
corrected search actually produces a better patch.

Evaluation only; no search runs.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankai.backends import get_backend
from bankai.eval import evaluate_generation
from bankai.patch import Patch, apply_patch, remove_patch
from bankai.probes import get_metric
from experiments_exp12_data import ALL_VAL, VAL_BY_CATEGORY

MODEL_PATH = "models/bonsai-8b-mlx"
PATCHES = [
    ("exp6  (token_gap search)", "patches/calculus_generalized_v1.json"),
    ("exp12 (corrected, base layers)", "patches/calculus_corrected_baseline.json"),
    ("exp12 (corrected, ext layers)", "patches/calculus_corrected_extended.json"),
]


def measure(backend, probes):
    prepare, measure_fn = get_metric("seq_logprob")
    return measure_fn(backend, prepare(backend, probes), [p.name for p in probes])


def main():
    backend = get_backend("mlx")
    backend.load(MODEL_PATH)

    print("Scoring baseline...", flush=True)
    base_gen = evaluate_generation(backend, ALL_VAL)
    base_lp = measure(backend, ALL_VAL)
    base_ok = {n: base_gen[n]["correct"] for n in base_gen}

    rows, detail = [], {}
    for label, path in PATCHES:
        patch = Patch.load(path)
        apply_patch(backend, patch)
        print(f"Scoring {label} ({len(patch.flips)} flips)...", flush=True)
        gen = evaluate_generation(backend, ALL_VAL)
        lp = measure(backend, ALL_VAL)
        remove_patch(backend, patch)

        ok = {n: gen[n]["correct"] for n in gen}
        fixed = [n for n in ok if not base_ok[n] and ok[n]]
        broke = [n for n in ok if base_ok[n] and not ok[n]]
        lp_fixed = sum(1 for n in lp if lp[n] > 0 and base_lp[n] <= 0)
        lp_broke = sum(1 for n in lp if lp[n] <= 0 and base_lp[n] > 0)
        rows.append((label, len(patch.flips), patch.size_bytes,
                     sum(ok.values()), len(fixed), len(broke), lp_fixed, lp_broke))
        detail[label] = {"fixed": fixed, "broke": broke,
                         "per_category": {cat: sum(ok[p.name] for p in ps)
                                          for cat, ps in VAL_BY_CATEGORY}}

    n = len(ALL_VAL)
    print(f"\n{'='*84}")
    print(f"HEAD-TO-HEAD on {n} held-out probes — identical probes, identical metric")
    print(f"{'='*84}")
    print(f"{'patch':32s} {'flips':>6s} {'bytes':>6s} {'gen':>7s} {'fix':>4s} {'brk':>4s} "
          f"{'lp fix':>7s} {'lp brk':>7s}")
    print(f"{'(no patch)':32s} {'-':>6s} {'-':>6s} {sum(base_ok.values()):>3d}/{n:<3d} "
          f"{'-':>4s} {'-':>4s} {'-':>7s} {'-':>7s}")
    for label, f, b, acc, fx, bk, lfx, lbk in rows:
        print(f"{label:32s} {f:6d} {b:6d} {acc:>3d}/{n:<3d} {fx:4d} {bk:4d} {lfx:7d} {lbk:7d}")

    print("\nPer-category (greedy generation):")
    cats = [c for c, _ in VAL_BY_CATEGORY]
    print(f"{'patch':32s} " + " ".join(f"{c[:9]:>10s}" for c in cats))
    print(f"{'(no patch)':32s} " +
          " ".join(f"{sum(base_ok[p.name] for p in ps):>10d}" for _, ps in VAL_BY_CATEGORY))
    for label, _ in PATCHES:
        d = detail[label]["per_category"]
        print(f"{label:32s} " + " ".join(f"{d[c]:>10d}" for c in cats))

    for label, _ in PATCHES:
        print(f"\n{label}: fixed={detail[label]['fixed']} broke={detail[label]['broke']}")

    with open("results/experiment14_headtohead.json", "w") as f:
        json.dump({"baseline_correct": sum(base_ok.values()), "n": n,
                   "rows": rows, "detail": detail}, f, indent=2)
    print("\nResults written: results/experiment14_headtohead.json")


if __name__ == "__main__":
    main()
