"""Re-score stored generation outputs under the current matcher.

Experiment 12 saves every generated string, so accepted-answer rules can be
revised without re-running the model. Both the before and after sets are always
re-scored with the identical rule, so a rule change cannot manufacture a fix.

    python tools/rescore_generation.py results/experiment12_baseline_results.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankai.eval import answer_matches
from experiments_exp12_data import ALL_TRAIN, ALL_VAL, VAL_BY_CATEGORY


def score(detail: dict, probes: list) -> dict[str, bool]:
    out = {}
    for p in probes:
        gen = detail[p.name]["generated"]
        accepted = (p.correct_token,) + tuple(p.alternates)
        out[p.name] = any(answer_matches(gen, form) for form in accepted)
    return out


def delta(before: dict[str, bool], after: dict[str, bool]):
    fixed = [n for n in before if not before[n] and after[n]]
    broke = [n for n in before if before[n] and not after[n]]
    return fixed, broke, sum(before.values()), sum(after.values())


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/experiment12_baseline_results.json"
    d = json.load(open(path))
    det = d["generation_detail"]

    for split, probes, kb, ka in (
        ("TRAIN", ALL_TRAIN, "train_before", "train_after"),
        ("VAL", ALL_VAL, "val_before", "val_after"),
    ):
        b, a = score(det[kb], probes), score(det[ka], probes)
        fixed, broke, nb, na = delta(b, a)
        print(f"\n{split} — greedy generation, re-scored")
        print(f"  accuracy: {nb}/{len(probes)} -> {na}/{len(probes)}")
        print(f"  fixed: {len(fixed)}  {fixed}")
        print(f"  broke: {len(broke)}  {broke}")

    b, a = score(det["val_before"], ALL_VAL), score(det["val_after"], ALL_VAL)
    print("\nPer-category validation:")
    for cat, probes in VAL_BY_CATEGORY:
        names = [p.name for p in probes]
        print(f"  {cat:14s} {sum(b[n] for n in names)}/{len(names)} -> "
              f"{sum(a[n] for n in names)}/{len(names)}")

    print(f"\nconfig: {d['config']}")


if __name__ == "__main__":
    main()
