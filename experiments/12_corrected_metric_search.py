"""Bankai Experiment 12: Experiment 6, rerun under a corrected probe metric.

Experiment 6 reported 4 held-out fixes with 0 breaks. An audit (see the paper's
errata section) found the measurement it optimized was unsound in three ways:
answers were reduced to a single token id by a last-subtoken rule that collapsed
" 20" and " 0" onto the same id, distractors were a constant " 0" that ranked 23rd
among the model's actual candidates, and the integral category repeated a single
token contrast fifteen times.

This experiment holds everything else fixed — same 90 prompts, same layers, same
projections, same 300 iterations, same seed, same control set — and changes only
the measurement:

  fitness   summed logprob of the full answer string vs. a plausible per-probe
            distractor, instead of a single-token logit gap
  headline  greedy generation accuracy, which depends on no distractor at all

Run with --layers extended to test the untested hypothesis in the Experiment 6
writeup: that layers 5, 6 and 10 are high-impact for calculus but were never in
the search set.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankai.backends import get_backend
from bankai.eval import evaluate_generation, generation_delta
from bankai.patch import apply_patch, remove_patch
from bankai.probes import KNOWLEDGE_PROBES, get_metric
from bankai.search import greedy_search
from experiments_exp12_data import ALL_TRAIN, ALL_VAL, VAL_BY_CATEGORY

MODEL_PATH = "models/bonsai-8b-mlx"
BASELINE_LAYERS = [1, 2, 3, 4, 34]
EXTENDED_LAYERS = [1, 2, 3, 4, 5, 6, 10, 34]


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


def report(title, delta, total):
    # Baseline correct = probes right before the patch = stayed_right + broke.
    before = len(delta["stayed_right"]) + len(delta["broke"])
    after = len(delta["stayed_right"]) + len(delta["fixed"])
    print(f"\n{title}")
    print(f"  fixed (wrong->right): {len(delta['fixed'])}  {delta['fixed']}")
    print(f"  broke (right->wrong): {len(delta['broke'])}  {delta['broke']}")
    print(f"  accuracy: {before}/{total} -> {after}/{total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", choices=["baseline", "extended"], default="baseline",
                    help="baseline reuses Experiment 6's layers; extended adds 5, 6, 10")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    layers = BASELINE_LAYERS if args.layers == "baseline" else EXTENDED_LAYERS
    tag = f"calculus_corrected_{args.layers}"

    print("=" * 70)
    print(f"Bankai Experiment 12: corrected metric, {args.layers} layers {layers}")
    print(f"  Training: {len(ALL_TRAIN)} probes | Validation: {len(ALL_VAL)} held out")
    print(f"  Fitness metric: seq_logprob | Headline metric: greedy generation")
    print("=" * 70)

    backend = get_backend("mlx")
    backend.load(MODEL_PATH)

    print("\nBaseline generation eval (pre-search)...", flush=True)
    gen_before_val = evaluate_generation(backend, ALL_VAL)
    gen_before_train = evaluate_generation(backend, ALL_TRAIN)
    logp_before_val = measure(backend, ALL_VAL)
    logp_before_train = measure(backend, ALL_TRAIN)

    patch = greedy_search(
        backend,
        target_probes=ALL_TRAIN,
        control_probes=KNOWLEDGE_PROBES,
        search_layers=layers,
        search_projs=["gate_proj", "up_proj"],
        max_iters=args.iters,
        control_penalty=2.0,
        fitness_mode="mean",
        metric="seq_logprob",
        seed=args.seed,
        patch_name=tag,
        patch_description=(
            f"Calculus patch, corrected seq_logprob metric, layers {layers}, "
            f"{args.iters} iters, mean fitness"
        ),
    )
    out_patch = f"patches/{tag}.json"
    patch.save(out_patch)
    print(f"\nPatch saved: {out_patch} — {len(patch.flips)} flips, {patch.size_bytes} bytes")

    # greedy_search leaves accepted flips applied; measure clean, then re-apply.
    remove_patch(backend, patch)
    print("\nPost-search baseline re-measure (sanity: must match pre-search)...", flush=True)
    check = measure(backend, ALL_VAL)
    drift = max(abs(check[k] - logp_before_val[k]) for k in check)
    print(f"  baseline drift after revert: {drift:.2e} (0 means XOR revert is exact)")

    apply_patch(backend, patch)
    logp_after_train = measure(backend, ALL_TRAIN)
    logp_after_val = measure(backend, ALL_VAL)
    print("\nPatched generation eval...", flush=True)
    gen_after_train = evaluate_generation(backend, ALL_TRAIN)
    gen_after_val = evaluate_generation(backend, ALL_VAL)

    report("TRAIN — seq_logprob", sign_flips(logp_before_train, logp_after_train), len(ALL_TRAIN))
    report("VAL   — seq_logprob", sign_flips(logp_before_val, logp_after_val), len(ALL_VAL))
    gd_train = generation_delta(gen_before_train, gen_after_train)
    gd_val = generation_delta(gen_before_val, gen_after_val)
    report("TRAIN — greedy generation", gd_train, len(ALL_TRAIN))
    report("VAL   — greedy generation  [HEADLINE]", gd_val, len(ALL_VAL))

    print("\nPer-category validation (greedy generation):")
    for cat, probes in VAL_BY_CATEGORY:
        names = [p.name for p in probes]
        b = sum(gen_before_val[n]["correct"] for n in names)
        a = sum(gen_after_val[n]["correct"] for n in names)
        print(f"  {cat:14s} {b}/{len(names)} -> {a}/{len(names)}")

    print("\nGeneration changes on validation:")
    for name in gd_val["fixed"] + gd_val["broke"]:
        tagm = "FIXED" if name in gd_val["fixed"] else "BROKE"
        print(f"  {tagm} {name:12s} {gen_before_val[name]['generated']!r}")
        print(f"        {'':12s} -> {gen_after_val[name]['generated']!r}")

    results = {
        "config": {"layers": layers, "iters": args.iters, "seed": args.seed,
                   "metric": "seq_logprob", "n_flips": len(patch.flips),
                   "size_bytes": patch.size_bytes},
        "baseline_drift_after_revert": drift,
        "seq_logprob": {
            "train": sign_flips(logp_before_train, logp_after_train),
            "val": sign_flips(logp_before_val, logp_after_val),
        },
        "generation": {"train": gd_train, "val": gd_val},
        "generation_detail": {
            "val_before": gen_before_val, "val_after": gen_after_val,
            "train_before": gen_before_train, "train_after": gen_after_train,
        },
        "gaps": {
            "val_before": logp_before_val, "val_after": logp_after_val,
            "train_before": logp_before_train, "train_after": logp_after_train,
        },
    }
    out_json = f"results/experiment12_{args.layers}_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written: {out_json}")


if __name__ == "__main__":
    main()
