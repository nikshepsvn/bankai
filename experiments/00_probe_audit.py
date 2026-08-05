"""Bankai Experiment 0: probe-set audit.

A precondition check for every experiment that reports logit gaps. Run it before
trusting a probe set; it is what turned up the measurement defects documented in
the errata (see README "Errata and Corrections").

Two classes of defect, both invisible in the reported numbers:

  dead probes     encode_token() keeps only the LAST subtoken of an answer. On a
                  tokenizer that splits digits into single characters, " 20" and
                  " 0" both reduce to "0" — correct_id == wrong_id, so the gap is
                  identically zero and no weight flip can ever move the probe. It
                  still scores as "wrong", inflating the count of fixable probes.

  offset probes   The answer is multi-token, so the id being measured is not the
                  token the model emits next. For " 4" after "d/dx [...] =", the
                  model's actual next token is a space; the digit is scored one
                  position early.

With --with-model, the audit additionally reports where the hardcoded distractor
sits among the model's real candidates. A distractor the model ranks 23rd measures
distance to a token it was never going to emit.

    python experiments/00_probe_audit.py
    python experiments/00_probe_audit.py --with-model --probe-set exp6
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankai.probes import Probe

MODEL_PATH = "models/bonsai-8b-mlx"


def load_probe_sets(which: str) -> dict[str, list[Probe]]:
    if which == "exp6":
        from experiments_exp6_data import ALL_TRAIN, ALL_VAL
        return {"exp6_train": ALL_TRAIN, "exp6_val": ALL_VAL}
    if which == "exp12":
        from experiments_exp12_data import ALL_TRAIN, ALL_VAL
        return {"exp12_train": ALL_TRAIN, "exp12_val": ALL_VAL}
    from bankai.probes import CODE_PROBES, KNOWLEDGE_PROBES, MATH_PROBES
    return {"math": MATH_PROBES, "code": CODE_PROBES, "knowledge": KNOWLEDGE_PROBES}


def audit_tokenization(tok, probes: list[Probe]) -> dict:
    def last(s):
        return int(tok.encode(s)[-1])

    dead = [p for p in probes if last(p.correct_token) == last(p.wrong_token)]
    offset = [p for p in probes if len(tok.encode(p.correct_token)) > 1]
    return {"n": len(probes), "dead": dead, "offset": offset}


def audit_distractors(backend, probes: list[Probe]) -> dict:
    """Where does the hardcoded wrong token sit among the model's candidates?"""
    import mlx.core as mx

    ranks, mismatched = [], 0
    for p in probes:
        logits = backend.model(mx.array(backend.encode(p.prompt))[None, :])
        last = logits[0, -1, :]
        mx.eval(last)
        order = [int(t) for t in mx.argsort(-last)[:200].tolist()]
        c_id, w_id = backend.encode_token(p.correct_token), backend.encode_token(p.wrong_token)
        top_wrong = order[0] if order[0] != c_id else order[1]
        if w_id != top_wrong:
            mismatched += 1
        if w_id in order:
            ranks.append(order.index(w_id))
    return {"mismatched": mismatched, "ranks": ranks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-set", choices=["exp6", "exp12", "builtin"], default="exp6")
    ap.add_argument("--with-model", action="store_true",
                    help="also rank hardcoded distractors against real model predictions")
    args = ap.parse_args()

    sets = load_probe_sets(args.probe_set)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("=" * 66)
    print(f"Probe audit — {args.probe_set}")
    print("=" * 66)
    print(f"{'set':16s} {'n':>4s} {'dead':>5s} {'offset':>7s}")
    total_dead = []
    for label, probes in sets.items():
        r = audit_tokenization(tok, probes)
        print(f"{label:16s} {r['n']:4d} {len(r['dead']):5d} {len(r['offset']):7d}")
        total_dead += [(label, p) for p in r["dead"]]

    if total_dead:
        print("\nDead probes — gap is identically 0.0 for any patch:")
        for label, p in total_dead:
            print(f"  [{label}] {p.name:14s} {p.correct_token!r} vs {p.wrong_token!r}")

    if not args.with_model:
        print("\n(pass --with-model for distractor-rank analysis)")
        return

    from bankai.backends import get_backend
    backend = get_backend("mlx")
    print(f"\nloading {MODEL_PATH} ...", flush=True)
    backend.load(MODEL_PATH)

    print("\nDistractor quality:")
    for label, probes in sets.items():
        r = audit_distractors(backend, probes)
        med = statistics.median(r["ranks"]) if r["ranks"] else float("nan")
        print(f"  {label:16s} hardcoded distractor is not the top competitor on "
              f"{r['mismatched']}/{len(probes)}; median rank {med:.0f}")


if __name__ == "__main__":
    main()
