"""Generation-level probe evaluation.

Logit gaps are a search signal, not a result. A gap can move without the model's
actual output changing, and — as the Experiment 6 audit showed — a gap measured
against a mis-tokenized target can move without meaning anything at all.

This module evaluates what the model actually emits under greedy decoding, which
depends on no distractor token and cannot be satisfied by shifting probability
between two hardcoded ids. It is the metric Experiment 12 reports as its headline.
"""

from typing import TYPE_CHECKING

from bankai.probes import Probe

if TYPE_CHECKING:
    from bankai.backends.base import Backend


# Characters that would extend a mathematical expression. If the generated text
# continues with one of these right after the expected answer, the model said
# something longer and different — " 6x^2" must not count as " 6x".
_CONTINUING = set("0123456789abcdefghijklmnopqrstuvwxyz^/*+-(")


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase, ignoring presentation-only differences.

    Grouping and explicit multiplication carry no content: (x^7)/7 == x^7/7 and
    2*e^x == 2e^x. The radical sign is spelled out so sqrt(2) and √2 compare equal.
    """
    out = "".join(text.split()).lower()
    for ch in "(){}[]*":
        out = out.replace(ch, "")
    return out.replace("√", "sqrt")


def answer_matches(generated: str, expected: str) -> bool:
    """True if greedy output begins with `expected` as a complete expression.

    The answer must not run on into a different expression: " 6x^2" does not
    satisfy " 6x". A trailing constant of integration is the one permitted
    continuation, wherever it appears — "x^4/4 + C. So..." is a correct
    antiderivative, while "6x + 2" is not a correct "6x".
    """
    gen, exp = _normalize(generated), _normalize(expected)
    if not exp or not gen.startswith(exp):
        return False
    rest = gen[len(exp):]
    if not rest or rest.startswith("+c"):
        return True
    return rest[0] not in _CONTINUING


def greedy_answer(backend: "Backend", prompt: str, max_tokens: int = 14) -> str:
    """Greedy-decode a short continuation, stopping at a newline.

    Uses a KV cache: without one, each new token re-runs the whole sequence and a
    90-probe sweep costs ~15 minutes instead of ~1.
    """
    import mlx.core as mx  # local: only the MLX backend supports this path today
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(backend.model)
    ids = backend.encode(prompt)
    logits = backend.model(mx.array(ids)[None, :], cache=cache)

    out: list[int] = []
    for _ in range(max_tokens):
        nxt = int(mx.argmax(logits[0, -1, :]).item())
        out.append(nxt)
        if "\n" in backend.tokenizer.decode([nxt]):
            break
        logits = backend.model(mx.array([[nxt]]), cache=cache)
    return backend.tokenizer.decode(out)


def evaluate_generation(
    backend: "Backend",
    probes: list[Probe],
    max_tokens: int = 14,
) -> dict[str, dict]:
    """Greedy-decode each probe and check whether the answer is correct.

    Returns {probe_name: {"generated": str, "correct": bool, "said_distractor": bool}}.
    `said_distractor` flags the specific wrong answer the probe was built around,
    which separates "the patch fixed the intended error" from "the patch changed
    the output to some third thing".
    """
    results = {}
    for probe in probes:
        gen = greedy_answer(backend, probe.prompt, max_tokens)
        accepted = (probe.correct_token,) + tuple(probe.alternates)
        results[probe.name] = {
            "generated": gen,
            "correct": any(answer_matches(gen, form) for form in accepted),
            "said_distractor": answer_matches(gen, probe.wrong_token),
        }
    return results


def generation_delta(before: dict[str, dict], after: dict[str, dict]) -> dict[str, list[str]]:
    """Split probes into fixed / broke / stayed_right / stayed_wrong."""
    out = {"fixed": [], "broke": [], "stayed_right": [], "stayed_wrong": []}
    for name in before:
        b, a = before[name]["correct"], after[name]["correct"]
        if not b and a:
            out["fixed"].append(name)
        elif b and not a:
            out["broke"].append(name)
        elif b and a:
            out["stayed_right"].append(name)
        else:
            out["stayed_wrong"].append(name)
    return out
