"""
Bankai CLI — create, apply, and analyze XOR patches for 1-bit LLMs.

Usage:
    bankai search --backend mlx --model <path> --target math --output patch.json
    bankai search --backend gguf --model <path> --target math --output patch.json
    bankai search --backend gguf --runner modal --target math --output patch.json
    bankai apply  --backend mlx --model <path> --patch patch.json --prompt "..."
    bankai info   --patch patch.json
    bankai eval   --backend mlx --model <path> --patch patch.json
"""

import argparse
import json
import sys
from pathlib import Path as FilePath

from bankai.backends import get_backend
from bankai.patch import Patch, apply_patch, remove_patch
from bankai.probes import (
    Probe, MATH_PROBES, CODE_PROBES, KNOWLEDGE_PROBES,
    measure_probes,
)
from bankai.search import greedy_search


def _supports_seq_logprob(backend) -> bool:
    """Whether this backend implements continuation scoring rather than inheriting
    the base method that raises."""
    from bankai.backends.base import Backend
    return type(backend).seq_logprobs is not Backend.seq_logprobs


PROBE_SETS = {
    "math": MATH_PROBES,
    "code": CODE_PROBES,
    "knowledge": KNOWLEDGE_PROBES,
}


def load_probes(target: str) -> list[Probe]:
    """Load probes from a built-in set name or a JSON file path."""
    if target in PROBE_SETS:
        return PROBE_SETS[target]

    path = FilePath(target)
    if path.exists() and path.suffix == ".json":
        data = json.loads(path.read_text())
        return [
            Probe(
                prompt=p["prompt"],
                correct_token=p["correct"],
                wrong_token=p["wrong"],
                name=p.get("name", f"probe_{i}"),
                category=p.get("category", "custom"),
            )
            for i, p in enumerate(data)
        ]

    print(f"Unknown target: '{target}'. Use a built-in set ({', '.join(PROBE_SETS)}) or a path to a JSON probe file.")
    sys.exit(1)


def cmd_search(args):
    """Search for an optimal patch."""
    target_probes = load_probes(args.target)

    # Use all other built-in probe sets as control
    control_probes = []
    for name, probes in PROBE_SETS.items():
        if name != args.target:
            control_probes.extend(probes)

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    # Dispatch to runner
    if args.runner == "modal":
        from bankai.runners.modal_runner import run_modal_search
        patch = run_modal_search(
            backend=args.backend,
            model_path=args.model,
            target_probes=target_probes,
            control_probes=control_probes,
            search_layers=layers,
            max_iters=args.iters,
            control_penalty=args.penalty,
            seed=args.seed,
            patch_name=f"{args.target}_patch",
            patch_description=f"Optimized for {args.target} probes",
        )
    else:
        backend = get_backend(args.backend)
        backend.load(args.model)
        if args.metric == "seq_logprob" and not _supports_seq_logprob(backend):
            raise SystemExit(
                f"{type(backend).__name__} does not implement seq_logprobs(). "
                "Re-run with --metric token_gap, noting that it measures a single "
                "token id and has the defects documented in the README errata."
            )
        patch = greedy_search(
            backend,
            target_probes=target_probes,
            control_probes=control_probes,
            search_layers=layers,
            max_iters=args.iters,
            control_penalty=args.penalty,
            metric=args.metric,
            seed=args.seed,
            patch_name=f"{args.target}_patch",
            patch_description=f"Optimized for {args.target} probes ({args.metric})",
        )

    patch.save(args.output)
    print(f"\nPatch saved to {args.output}")
    print(f"  Flips: {len(patch.flips)}")
    print(f"  Bits: {patch.n_bits_flipped:,}")
    print(f"  Size: {patch.size_bytes} bytes")


def cmd_apply(args):
    """Apply a patch and run a prompt."""
    backend = get_backend(args.backend)
    backend.load(args.model)
    patch = Patch.load(args.patch)

    if args.prompt:
        print(f"[without patch]")
        response = backend.generate(args.prompt, max_tokens=args.max_tokens)
        print(f"  {response}\n")

        apply_patch(backend, patch)
        print(f"[with patch: {patch.name}]")
        response = backend.generate(args.prompt, max_tokens=args.max_tokens)
        print(f"  {response}")
    else:
        apply_patch(backend, patch)
        print(f"Patch '{patch.name}' applied ({len(patch.flips)} flips)")


def cmd_info(args):
    """Show patch metadata."""
    patch = Patch.load(args.patch)
    print(f"Patch: {patch.name}")
    print(f"Description: {patch.description}")
    print(f"Base model: {patch.base_model}")
    print(f"Flips: {len(patch.flips)}")
    print(f"Bits flipped: {patch.n_bits_flipped:,}")
    print(f"Size: {patch.size_bytes} bytes")
    if patch.metadata:
        print(f"Metadata:")
        for k, v in patch.metadata.items():
            print(f"  {k}: {v}")


def cmd_eval(args):
    """Evaluate a patch against probe sets."""
    backend = get_backend(args.backend)
    backend.load(args.model)
    patch = Patch.load(args.patch)

    probe_names = args.probes.split(",") if args.probes else list(PROBE_SETS.keys())
    all_probes = []
    for name in probe_names:
        if name in PROBE_SETS:
            all_probes.extend(PROBE_SETS[name])

    # Baseline
    baseline = measure_probes(backend, all_probes)

    # With patch
    apply_patch(backend, patch)
    patched = measure_probes(backend, all_probes)
    remove_patch(backend, patch)

    print(f"{'Probe':<20} {'Baseline':>10} {'Patched':>10} {'Delta':>10}")
    print("-" * 52)
    for probe in all_probes:
        b = baseline[probe.name]
        p = patched[probe.name]
        d = p - b
        marker = " *" if abs(d) > 0.5 else ""
        print(f"  {probe.name:<18} {b:>+10.3f} {p:>+10.3f} {d:>+10.3f}{marker}")


def _add_backend_args(p):
    p.add_argument("--backend", default="mlx", choices=["mlx", "gguf"],
                   help="Model backend (default: mlx)")
    p.add_argument("--runner", default="local", choices=["local", "modal"],
                   help="Where to run the search (default: local)")


def main():
    parser = argparse.ArgumentParser(prog="bankai", description="XOR patches for 1-bit LLMs")
    sub = parser.add_subparsers(dest="command")

    # search
    p = sub.add_parser("search", help="Search for an optimal patch")
    _add_backend_args(p)
    p.add_argument("--model", required=True, help="Path to model (MLX dir or GGUF file or HF repo)")
    p.add_argument("--target", required=True, help="Probe set (math, code, knowledge) or JSON file")
    p.add_argument("--output", default="patch.json", help="Output patch file")
    p.add_argument("--iters", type=int, default=200, help="Search iterations")
    p.add_argument("--layers", default=None, help="Comma-separated layer indices")
    p.add_argument("--penalty", type=float, default=2.0, help="Control degradation penalty")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--metric", default="seq_logprob", choices=["seq_logprob", "token_gap"],
                   help="Probe metric. seq_logprob scores whole answer strings and is the "
                        "sound default; token_gap is the original single-token metric kept "
                        "for reproducing Experiments 1-11 and has known defects (see the "
                        "README errata). token_gap is currently the only option on GGUF.")

    # apply
    p = sub.add_parser("apply", help="Apply a patch and optionally run a prompt")
    _add_backend_args(p)
    p.add_argument("--model", required=True, help="Path to model")
    p.add_argument("--patch", required=True, help="Patch file")
    p.add_argument("--prompt", default=None, help="Prompt to run")
    p.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")

    # info
    p = sub.add_parser("info", help="Show patch metadata")
    p.add_argument("--patch", required=True, help="Patch file")

    # eval
    p = sub.add_parser("eval", help="Evaluate a patch against probes")
    _add_backend_args(p)
    p.add_argument("--model", required=True, help="Path to model")
    p.add_argument("--patch", required=True, help="Patch file")
    p.add_argument("--probes", default=None, help="Comma-separated probe sets")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"search": cmd_search, "apply": cmd_apply, "info": cmd_info, "eval": cmd_eval}[args.command](args)


if __name__ == "__main__":
    main()
