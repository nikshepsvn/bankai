"""
Backend-agnostic search algorithms for finding optimal XOR patches.

Greedy hill climbing with screening optimization:
  - Pre-tokenize all probes once (via backend.encode)
  - Screen candidates on the 2 worst target probes before full eval
  - Skip redundant backend calls on reverts

Works with any Backend implementation (MLX, GGUF, ...).
"""

import time
from typing import Any

import numpy as np

from bankai.backends.base import Backend
from bankai.patch import Patch, PatchFlip
from bankai.probes import Probe, compute_fitness, compute_fitness_min, get_metric


def _pre_tokenize(backend: Backend, probes: list[Probe]) -> list[tuple[list[int], int, int]]:
    """Tokenize probes once. Returns list of (token_ids, correct_id, wrong_id)."""
    result = []
    for probe in probes:
        tokens = backend.encode(probe.prompt)
        c_id = backend.encode_token(probe.correct_token)
        w_id = backend.encode_token(probe.wrong_token)
        result.append((tokens, c_id, w_id))
    return result


def _measure_fast(
    backend: Backend,
    precomputed: list[tuple[list[int], int, int]],
    names: list[str],
) -> dict[str, float]:
    """Measure logit gaps using pre-tokenized probes.

    Uses backend.batch_logit_gaps if available (GGUF backend) for a big
    speedup by pipelining commands to the subprocess. Falls back to a
    per-probe loop otherwise (MLX backend).
    """
    batch_fn = getattr(backend, "batch_logit_gaps", None)
    if batch_fn is not None and precomputed:
        values = batch_fn(precomputed)
        return dict(zip(names, values))

    gaps = {}
    for (tokens, c_id, w_id), name in zip(precomputed, names):
        gaps[name] = backend.logit_gap(tokens, c_id, w_id)
    return gaps


def greedy_search(
    backend: Backend,
    target_probes: list[Probe],
    control_probes: list[Probe],
    search_layers: list[int] | None = None,
    search_projs: list[str] | None = None,
    max_iters: int = 200,
    control_penalty: float = 2.0,
    fitness_mode: str = "mean",
    metric: str = "token_gap",
    seed: int = 42,
    patch_name: str = "untitled",
    patch_description: str = "",
    base_model: str = "prism-ml/Bonsai-8B",
    verbose: bool = True,
    granularity: str = "row",
) -> Patch:
    """Greedy hill climbing with screening optimization.

    Screening: measures only the 2 worst target probes first. If neither
    improved, rejects immediately without evaluating the remaining probes.

    fitness_mode: "mean" (default) averages improvement across all target probes.
    "min" uses the worst probe's improvement, preventing overfitting to easy probes.

    granularity: "row" (default) flips all bits in an output row.
                 "group" flips only one 128-bit group inside a row — 32x more
                 precise but 32x larger search space. Requires backend.flip_group
                 to be implemented.

    Args:
        backend: A loaded Backend implementation (MLX, GGUF, etc.)
        target_probes: Probes to optimize (push logit gaps higher)
        control_probes: Probes to preserve (penalize degradation)
        ...
    """
    if search_layers is None:
        search_layers = [1, 2, 3, 4, 34]
    if search_projs is None:
        search_projs = ["gate_proj", "up_proj"]

    if granularity not in ("row", "group"):
        raise ValueError(f"granularity must be 'row' or 'group', got {granularity!r}")
    if granularity == "group" and not hasattr(backend, "flip_group"):
        raise RuntimeError(
            f"Backend {type(backend).__name__} does not implement flip_group; "
            "cannot use granularity='group'"
        )

    rng = np.random.default_rng(seed)

    # Pre-tokenize all probes once, under the selected metric
    prepare_fn, measure_fn = get_metric(metric)
    target_pre = prepare_fn(backend, target_probes)
    control_pre = prepare_fn(backend, control_probes)
    target_names = [p.name for p in target_probes]
    control_names = [p.name for p in control_probes]

    def _measure(pre, names):
        return measure_fn(backend, pre, names)

    # Measure baselines
    target_baseline = _measure(target_pre, target_names)
    control_baseline = _measure(control_pre, control_names)

    if verbose:
        print(f"Baseline target gaps: { {k: f'{v:+.3f}' for k, v in target_baseline.items()} }")
        print(f"Baseline control gaps: { {k: f'{v:+.3f}' for k, v in control_baseline.items()} }")

    # Identify the 2 worst target probes for screening
    sorted_targets = sorted(target_baseline.items(), key=lambda x: x[1])
    screen_names = [sorted_targets[i][0] for i in range(min(2, len(sorted_targets)))]
    screen_indices = [target_names.index(n) for n in screen_names]
    screen_pre = [target_pre[i] for i in screen_indices]

    if verbose:
        print(f"Screen probes: {screen_names} (worst baseline gaps)")

    # Build candidate pool weighted by scale magnitude
    # Row granularity: candidate = (layer, proj, row), weight = avg row scale
    # Group granularity: candidate = (layer, proj, row, group), weight = row scale
    candidates = []
    weights = []
    groups_per_row = 32  # Q1_0_g128 with cols=4096

    for layer_idx in search_layers:
        for proj in search_projs:
            row_scales = backend.get_row_scales(layer_idx, proj)
            n_rows = backend.num_rows(layer_idx, proj)
            if granularity == "row":
                for row in range(n_rows):
                    candidates.append((layer_idx, proj, row))
                    weights.append(row_scales[row])
            else:
                for row in range(n_rows):
                    for g in range(groups_per_row):
                        candidates.append((layer_idx, proj, row, g))
                        weights.append(row_scales[row])

    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    if verbose:
        units = "rows" if granularity == "row" else "groups"
        print(f"Search space: {len(candidates)} {units} across {len(search_layers)} layers")
        print(f"Running {max_iters} iterations ({granularity}-level)...\n")

    def apply_candidate(key):
        if granularity == "row":
            backend.flip_row(*key)
        else:
            backend.flip_group(*key)

    def revert_candidate(key):
        # XOR is self-inverse
        apply_candidate(key)

    # Greedy search with screening
    accepted = []
    current_fitness = 0.0
    tried = set()
    screened_out = 0
    t0 = time.time()

    for step in range(max_iters):
        # Sample an untried candidate
        attempts = 0
        while attempts < 100:
            idx = rng.choice(len(candidates), p=weights)
            key = candidates[idx]
            if key not in tried:
                tried.add(key)
                break
            attempts += 1
        else:
            if verbose:
                print("  Exhausted candidate pool.")
            break

        apply_candidate(key)

        # Phase 1: screen on worst 2 target probes
        screen_gaps = _measure(screen_pre, screen_names)
        screen_improved = any(screen_gaps[n] > target_baseline[n] for n in screen_names)

        if not screen_improved:
            revert_candidate(key)
            screened_out += 1
            if verbose and (step + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{step+1:>4}/{max_iters}] ...  "
                      f"screened={screened_out}  accepted={len(accepted)}  "
                      f"[{(step+1)/elapsed:.1f} it/s]")
            continue

        # Phase 2: full eval (skip probes already measured in screen)
        remaining_idx = [i for i in range(len(target_pre)) if i not in screen_indices]
        remaining_pre = [target_pre[i] for i in remaining_idx]
        remaining_names = [target_names[i] for i in remaining_idx]
        remaining_gaps = _measure(remaining_pre, remaining_names)
        target_gaps = {**screen_gaps, **remaining_gaps}
        control_gaps = _measure(control_pre, control_names)

        fitness_fn = compute_fitness_min if fitness_mode == "min" else compute_fitness
        fitness = fitness_fn(
            target_gaps, control_gaps, target_baseline, control_baseline, control_penalty
        )

        if fitness > current_fitness:
            if granularity == "row":
                layer_idx, proj, row_idx = key
                accepted.append(PatchFlip(layer_idx, proj, row_idx))
                label = f"L{layer_idx}.{proj}[{row_idx}]"
            else:
                layer_idx, proj, row_idx, group_idx = key
                accepted.append(PatchFlip(layer_idx, proj, row_idx, group=group_idx))
                label = f"L{layer_idx}.{proj}[{row_idx},g{group_idx}]"
            current_fitness = fitness
            if verbose:
                elapsed = time.time() - t0
                print(f"  [{step+1:>4}/{max_iters}] ACCEPT  "
                      f"fitness={fitness:>+.4f}  {label}  "
                      f"({len(accepted)} flips, {elapsed:.0f}s)")
        else:
            revert_candidate(key)

    elapsed = time.time() - t0
    if verbose:
        print(f"\nSearch complete: {len(accepted)} flips accepted, "
              f"{screened_out} screened out, in {elapsed:.1f}s "
              f"({max_iters/elapsed:.1f} it/s)")

    return Patch(
        name=patch_name,
        description=patch_description,
        base_model=base_model,
        flips=accepted,
        metadata={
            "search_algorithm": f"greedy_hill_climbing_screened_{fitness_mode}",
            "metric": metric,
            "backend": type(backend).__name__,
            "search_layers": search_layers,
            "search_projs": search_projs,
            "max_iters": max_iters,
            "control_penalty": control_penalty,
            "final_fitness": current_fitness,
            "search_time_seconds": elapsed,
            "screened_out": screened_out,
        },
    )
