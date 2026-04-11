"""
Probe-based evaluation for measuring behavioral shifts.

A probe is a (prompt, correct_token, wrong_token) triple.
We measure the logit gap between correct and wrong as a
fast, single-forward-pass behavioral signal.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bankai.backends.base import Backend


@dataclass
class Probe:
    prompt: str
    correct_token: str
    wrong_token: str
    name: str
    category: str = "general"


# ── Built-in probe sets ──

MATH_PROBES = [
    Probe("1 + 1 =", " 2", " 3", "add_1", "math"),
    Probe("2 + 2 =", " 4", " 5", "add_2", "math"),
    Probe("7 * 8 =", " 56", " 54", "mul_1", "math"),
    Probe("The square root of 144 is", " 12", " 14", "sqrt_1", "math"),
    Probe("If x = 3, then x^2 =", " 9", " 8", "algebra_1", "math"),
    Probe("100 / 4 =", " 25", " 20", "div_1", "math"),
]

CODE_PROBES = [
    Probe("def hello():\n    print(\"Hello", " World", " Goodbye", "hello_world", "code"),
    Probe("In Python, to open a file you use the", " open", " close", "python_open", "code"),
    Probe("for i in range(10):\n    print(", "i", "x", "for_loop", "code"),
    Probe("import json\ndata = json.", "loads", "dump", "json_loads", "code"),
]

KNOWLEDGE_PROBES = [
    Probe("The capital of France is", " Paris", " London", "france_capital", "knowledge"),
    Probe("The capital of Japan is", " Tokyo", " Beijing", "japan_capital", "knowledge"),
    Probe("The color of the sky is", " blue", " red", "sky_color", "knowledge"),
    Probe("Einstein is famous for the theory of", " relativity", " evolution", "einstein", "knowledge"),
    Probe("The chemical formula for water is H", "2", "3", "water_formula", "knowledge"),
]


def measure_probes(backend: "Backend", probes: list[Probe]) -> dict[str, float]:
    """Measure logit gap (correct - wrong) for each probe.

    Uses backend.batch_logit_gaps if available (big speedup for GGUF) and
    falls back to per-probe calls otherwise.
    """
    if not probes:
        return {}

    precomputed = []
    for probe in probes:
        tokens = backend.encode(probe.prompt)
        c_id = backend.encode_token(probe.correct_token)
        w_id = backend.encode_token(probe.wrong_token)
        precomputed.append((tokens, c_id, w_id))

    batch_fn = getattr(backend, "batch_logit_gaps", None)
    if batch_fn is not None:
        values = batch_fn(precomputed)
        return {p.name: v for p, v in zip(probes, values)}

    gaps = {}
    for probe, (tokens, c_id, w_id) in zip(probes, precomputed):
        gaps[probe.name] = backend.logit_gap(tokens, c_id, w_id)
    return gaps


def compute_fitness(
    target_gaps: dict[str, float],
    control_gaps: dict[str, float],
    target_baseline: dict[str, float],
    control_baseline: dict[str, float],
    control_penalty: float = 2.0,
) -> float:
    """Fitness = avg target improvement - penalty × avg control degradation."""
    target_improvement = sum(
        target_gaps[n] - target_baseline[n] for n in target_baseline
    ) / len(target_baseline)

    control_degradation = sum(
        max(0, control_baseline[n] - control_gaps[n]) for n in control_baseline
    ) / len(control_baseline)

    return target_improvement - control_penalty * control_degradation


def compute_fitness_min(
    target_gaps: dict[str, float],
    control_gaps: dict[str, float],
    target_baseline: dict[str, float],
    control_baseline: dict[str, float],
    control_penalty: float = 2.0,
) -> float:
    """Min-of-probes fitness: maximize the worst target improvement.

    Prevents the search from overfitting to the easiest probe — a flip
    must help the worst-performing probe to be accepted.
    """
    improvements = [
        target_gaps[n] - target_baseline[n] for n in target_baseline
    ]
    target_min = min(improvements)

    control_degradation = sum(
        max(0, control_baseline[n] - control_gaps[n]) for n in control_baseline
    ) / len(control_baseline)

    return target_min - control_penalty * control_degradation
