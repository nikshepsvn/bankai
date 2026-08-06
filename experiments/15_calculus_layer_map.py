"""Bankai Experiment 15: calculus-specific layer impact map.

Experiment 6's writeup asserted that a calculus-specific layer impact map
"identified layers 5, 6, and 10 as high-impact for calculus" and used that to
explain why trigonometry and exponential derivatives saw no held-out fixes. No
artifact for that map was ever committed — experiments/02_logit_steering.py runs
on a generic 8-probe set — so the claim rested on an unpublished measurement.

Experiment 12's extended run already falsified the downstream hypothesis: adding
layers 5, 6 and 10 to the search left trigonometry and second derivatives
unmoved and reduced held-out fixes from 5 to 3. This experiment tests the premise
underneath it, and does so with the corrected metric rather than token_gap.

Method follows Experiment 2A: XOR the entire MLP of one layer (gate, up and
down projections), measure the mean absolute change in probe score, restore, and
repeat for all 36 layers. Large change means the layer is load-bearing for these
probes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankai.backends import get_backend
from bankai.probes import get_metric
from experiments_exp12_data import ALL_TRAIN, ALL_VAL

MODEL_PATH = "models/bonsai-8b-mlx"
PROJECTIONS = ["gate_proj", "up_proj", "down_proj"]
PROBES = ALL_TRAIN + ALL_VAL
CLAIMED = [5, 6, 10]  # the layers README asserted were high-impact for calculus
SEARCHED = [1, 2, 3, 4, 34]  # the layers Experiments 3-6 actually searched


def main():
    backend = get_backend("mlx")
    backend.load(MODEL_PATH)
    prepare, measure_fn = get_metric("seq_logprob")

    pre = prepare(backend, PROBES)
    names = [p.name for p in PROBES]
    baseline = measure_fn(backend, pre, names)

    print("=" * 72)
    print(f"Calculus layer impact map — {len(PROBES)} calculus probes, seq_logprob")
    print("=" * 72)
    print(f"{'layer':>5} {'mean |delta|':>13} {'max |delta|':>12} {'sign flips':>11}")

    rows = []
    for layer in range(backend.num_layers()):
        for proj in PROJECTIONS:
            backend.flip_projection(layer, proj)

        flipped = measure_fn(backend, pre, names)

        for proj in PROJECTIONS:  # XOR is self-inverse
            backend.flip_projection(layer, proj)

        deltas = [abs(flipped[n] - baseline[n]) for n in names]
        mean_d = sum(deltas) / len(deltas)
        max_d = max(deltas)
        flips = sum(1 for n in names if (flipped[n] > 0) != (baseline[n] > 0))
        rows.append({"layer": layer, "mean_abs_delta": mean_d,
                     "max_abs_delta": max_d, "sign_flips": flips})

        tag = ""
        if layer in CLAIMED:
            tag = "  <- claimed high-impact for calculus"
        elif layer in SEARCHED:
            tag = "  <- in the Exp 3-6 search set"
        print(f"{layer:>5} {mean_d:>13.4f} {max_d:>12.4f} {flips:>11}{tag}", flush=True)

    ranked = sorted(rows, key=lambda r: r["mean_abs_delta"], reverse=True)
    rank_of = {r["layer"]: i + 1 for i, r in enumerate(ranked)}

    print(f"\nTop 8 layers by mean |delta|: {[r['layer'] for r in ranked[:8]]}")
    print(f"\n{'layer':>5} {'rank':>6}  status")
    for layer in sorted(set(CLAIMED + SEARCHED)):
        status = "claimed high-impact" if layer in CLAIMED else "searched in Exp 3-6"
        print(f"{layer:>5} {rank_of[layer]:>6}  {status}")

    claimed_ranks = [rank_of[l] for l in CLAIMED]
    searched_ranks = [rank_of[l] for l in SEARCHED]
    print(f"\nclaimed layers {CLAIMED} rank {claimed_ranks} of {len(rows)}")
    print(f"searched layers {SEARCHED} rank {searched_ranks} of {len(rows)}")
    verdict = ("SUPPORTED" if max(claimed_ranks) <= 10 else "NOT SUPPORTED")
    print(f"\nPremise that 5/6/10 are high-impact for calculus: {verdict}")

    with open("results/experiment15_calculus_layer_map.json", "w") as f:
        json.dump({"probes": len(PROBES), "metric": "seq_logprob",
                   "layers": rows, "rank_of": rank_of,
                   "claimed": CLAIMED, "searched": SEARCHED}, f, indent=2)
    print("\nResults written: results/experiment15_calculus_layer_map.json")


if __name__ == "__main__":
    main()
