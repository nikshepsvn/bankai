"""
Bankai Experiment 7: Patch Stacking
====================================
Apply the Experiment 3 math patch and Experiment 4 calculus patch
simultaneously via sequential XOR. Measures:

  1. Mechanical correctness — order independence, perfect reversibility
  2. Behavioral composition — do individual improvements add up, or interfere?

Finding: Stacking is mechanically sound (order-independent, fully reversible,
zero drift) but behaviorally lossy — individual improvements partially cancel.
Stacking is safer (fewer breakages) but weaker (fewer fixes) than either
patch alone.
"""

from bankai.backends import get_backend
from bankai.patch import Patch, apply_patch, remove_patch
from bankai.probes import Probe, MATH_PROBES, KNOWLEDGE_PROBES, measure_probes


CALC_PROBES = [
    Probe("d/dx [x^4 + 3x^2] =", " 4", " 0", "poly_deriv", "calc"),
    Probe("The second derivative of x^4 is 12x^", "2", "3", "second_deriv", "calc"),
    Probe("The integral of x^2 dx = ", " 1", " 2", "integral", "calc"),
]


def sign_flip(baseline, patched, probes):
    fixed = broke = 0
    for p in probes:
        b, a = baseline[p.name], patched[p.name]
        if b <= 0 and a > 0:
            fixed += 1
        elif b > 0 and a <= 0:
            broke += 1
    return fixed, broke


def main():
    print("=" * 60)
    print("Bankai Experiment 7: Patch Stacking")
    print("=" * 60)

    backend = get_backend("mlx")
    backend.load("models/bonsai-8b-mlx")

    math_patch = Patch.load("patches/patch_math_v1.json")
    calc_patch = Patch.load("patches/calculus_v1.json")

    math_flips = {(f.layer, f.proj, f.row) for f in math_patch.flips}
    calc_flips = {(f.layer, f.proj, f.row) for f in calc_patch.flips}
    overlap = math_flips & calc_flips

    print(f"\nMath patch: {len(math_flips)} flips")
    print(f"Calculus patch: {len(calc_flips)} flips")
    print(f"Overlap (would cancel out): {len(overlap)}")

    all_probes = MATH_PROBES + CALC_PROBES + KNOWLEDGE_PROBES

    # Baseline (no patch)
    baseline = measure_probes(backend, all_probes)

    # Math patch alone
    apply_patch(backend, math_patch)
    math_only = measure_probes(backend, all_probes)
    remove_patch(backend, math_patch)

    # Calculus patch alone
    apply_patch(backend, calc_patch)
    calc_only = measure_probes(backend, all_probes)
    remove_patch(backend, calc_patch)

    # Stacked (math + calc)
    apply_patch(backend, math_patch)
    apply_patch(backend, calc_patch)
    stacked = measure_probes(backend, all_probes)

    # Order-independence check: swap apply order, compare
    remove_patch(backend, calc_patch)
    remove_patch(backend, math_patch)
    apply_patch(backend, calc_patch)
    apply_patch(backend, math_patch)
    stacked_reverse = measure_probes(backend, all_probes)

    # Reversibility check: remove both, should return to baseline
    remove_patch(backend, math_patch)
    remove_patch(backend, calc_patch)
    restored = measure_probes(backend, all_probes)

    max_drift = max(abs(restored[p.name] - baseline[p.name]) for p in all_probes)
    order_diff = max(abs(stacked[p.name] - stacked_reverse[p.name]) for p in all_probes)

    print("\n── Sign-flip comparison ──")
    print(f"  Math only: fixed={sign_flip(baseline, math_only, all_probes)}")
    print(f"  Calc only: fixed={sign_flip(baseline, calc_only, all_probes)}")
    print(f"  Stacked:   fixed={sign_flip(baseline, stacked, all_probes)}")

    print("\n── Mechanical properties ──")
    print(f"  Order independence: max diff between math+calc and calc+math = {order_diff:.6f}")
    print(f"  Reversibility: max drift after removing both = {max_drift:.6f}")
    print(f"  Verdict: {'✓ PASS' if max_drift < 1e-5 and order_diff < 1e-5 else '✗ FAIL'}")


if __name__ == "__main__":
    main()
