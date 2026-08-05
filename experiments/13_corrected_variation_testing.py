"""Bankai Experiment 13: Experiment 5, rerun under a corrected probe metric.

Experiment 5 applied the 6-probe Experiment 4 patch to 90 novel variations and
concluded that patches trained on few probes memorize rather than generalize.
That conclusion is not contradicted by the audit, but its sign-flip counts were
computed on a probe set containing 7 dead probes (gap identically zero, scored as
wrong) — so the magnitudes were unreliable.

This is evaluation only; no search runs. The same Experiment 4 patch is applied
to the same 90 prompts, scored with full-answer log-probability and greedy
generation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankai.backends import get_backend
from bankai.eval import evaluate_generation, generation_delta
from bankai.patch import Patch, apply_patch, remove_patch
from bankai.probes import get_metric
from experiments_exp13_data import ALL_CATEGORIES, ALL_VARIATIONS

MODEL_PATH = "models/bonsai-8b-mlx"
PATCH_PATH = "patches/calculus_v1.json"


def measure(backend, probes):
    prepare, measure_fn = get_metric("seq_logprob")
    return measure_fn(backend, prepare(backend, probes), [p.name for p in probes])


def sign_flips(before: dict, after: dict) -> dict:
    out = {"fixed": [], "broke": [], "stayed_right": [], "stayed_wrong": []}
    for name in before:
        b, a = before[name] > 0, after[name] > 0
        key = ("stayed_right" if b else "fixed") if a else ("broke" if b else "stayed_wrong")
        out[key].append(name)
    return out


def show(title, delta, total):
    before = len(delta["stayed_right"]) + len(delta["broke"])
    after = len(delta["stayed_right"]) + len(delta["fixed"])
    print(f"\n{title}")
    print(f"  accuracy: {before}/{total} -> {after}/{total}")
    print(f"  fixed: {len(delta['fixed'])}  {delta['fixed']}")
    print(f"  broke: {len(delta['broke'])}  {delta['broke']}")


def main():
    print("=" * 70)
    print("Bankai Experiment 13: corrected variation testing (Experiment 5 redone)")
    print(f"  Patch: {PATCH_PATH} | {len(ALL_VARIATIONS)} variation probes")
    print("=" * 70)

    backend = get_backend("mlx")
    backend.load(MODEL_PATH)
    patch = Patch.load(PATCH_PATH)

    print("\nBaseline...", flush=True)
    logp_before = measure(backend, ALL_VARIATIONS)
    gen_before = evaluate_generation(backend, ALL_VARIATIONS)

    apply_patch(backend, patch)
    print("Patched...", flush=True)
    logp_after = measure(backend, ALL_VARIATIONS)
    gen_after = evaluate_generation(backend, ALL_VARIATIONS)
    remove_patch(backend, patch)

    lp = sign_flips(logp_before, logp_after)
    gd = generation_delta(gen_before, gen_after)
    show("seq_logprob", lp, len(ALL_VARIATIONS))
    show("greedy generation  [HEADLINE]", gd, len(ALL_VARIATIONS))

    print("\nPer-category (greedy generation):")
    for cat, probes in ALL_CATEGORIES:
        names = [p.name for p in probes]
        b = sum(gen_before[n]["correct"] for n in names)
        a = sum(gen_after[n]["correct"] for n in names)
        print(f"  {cat:14s} {b}/{len(names)} -> {a}/{len(names)}")

    out = {
        "patch": PATCH_PATH,
        "n_flips": len(patch.flips),
        "seq_logprob": lp,
        "generation": gd,
        "generation_detail": {"before": gen_before, "after": gen_after},
        "gaps": {"before": logp_before, "after": logp_after},
    }
    with open("results/experiment13_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults written: results/experiment13_results.json")


if __name__ == "__main__":
    main()
