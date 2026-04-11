"""
Experiment 6 + attention projections.

First run ever to search XOR flips in attention Q/K/V/O projections.
Same 60 training / 30 validation probes, 8 layers, 800 iterations,
but search_projs now covers MLP (gate_proj, up_proj) AND attention
(q_proj, k_proj, v_proj, o_proj).

The attention projections have 4096 (q, o) or 1024 (k, v, GQA) rows
each, so the search space is ~3x larger than the MLP-only version.
"""

import modal

app = modal.App("bankai-exp6-attn")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "cmake", "build-essential", "ninja-build", "wget")
    .run_commands(
        "git clone --depth 1 https://github.com/PrismML-Eng/llama.cpp.git /root/llama.cpp",
    )
    .run_commands(
        "cd /root/llama.cpp/gguf-py && pip install -e .",
    )
    .pip_install("huggingface_hub", "numpy")
    # Stable llama.cpp build (cached)
    .run_commands(
        "cd /root/llama.cpp && cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=OFF "
        "-DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda "
        "-DCMAKE_CUDA_ARCHITECTURES=89",
        "cd /root/llama.cpp && cmake --build build -j --config Release --target llama",
        "cd /root/llama.cpp && cmake --build build -j --config Release --target common",
        gpu="L4",
    )
    # Dynamic: bankai_eval source
    .add_local_dir("tools", "/root/bankai_tools", copy=True)
    .run_commands(
        "mkdir -p /root/llama.cpp/examples/bankai_eval",
        "cp /root/bankai_tools/bankai_eval.cpp /root/llama.cpp/examples/bankai_eval/",
        "cp /root/bankai_tools/CMakeLists.txt /root/llama.cpp/examples/bankai_eval/",
        "echo 'add_subdirectory(bankai_eval)' >> /root/llama.cpp/examples/CMakeLists.txt",
        "cd /root/llama.cpp && cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=OFF "
        "-DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda "
        "-DCMAKE_CUDA_ARCHITECTURES=89",
        "cd /root/llama.cpp && cmake --build build -j --config Release --target bankai_eval",
        gpu="L4",
    )
    .add_local_python_source("bankai")
    .add_local_python_source("experiments_exp6_data")
)


@app.function(image=image, gpu="L40S", timeout=3600, memory=24576)
def run_attn_exp6():
    import time
    from huggingface_hub import hf_hub_download
    from experiments_exp6_data import ALL_TRAIN, ALL_VAL, VAL_BY_CATEGORY
    from bankai.backends.gguf_backend import GGUFBackend
    from bankai.probes import KNOWLEDGE_PROBES, measure_probes
    from bankai.search import greedy_search
    from bankai.patch import apply_patch, remove_patch

    print(f"[attn] Training: {len(ALL_TRAIN)} probes, Validation: {len(ALL_VAL)} probes")

    model_path = hf_hub_download(
        repo_id="prism-ml/Bonsai-8B-gguf",
        filename="Bonsai-8B.gguf",
        cache_dir="/root/models",
    )

    backend = GGUFBackend(bankai_eval_path="/root/llama.cpp/build/bin/bankai_eval")
    t0 = time.time()
    backend.load(model_path)
    print(f"[attn] Loaded in {time.time()-t0:.1f}s")

    # Sanity check: can we query the attention tensors?
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        n = backend.num_rows(0, proj)
        print(f"[attn] layer 0 {proj}: {n} rows")

    # Expanded search: 8 layers × 6 projections (2 MLP + 4 attention)
    search_layers = [1, 2, 3, 4, 5, 6, 10, 34]
    search_projs = ["gate_proj", "up_proj", "q_proj", "k_proj", "v_proj", "o_proj"]

    print(f"\n[attn] Searching {len(search_layers)} layers × {len(search_projs)} projs")

    t0 = time.time()
    patch = greedy_search(
        backend,
        target_probes=ALL_TRAIN,
        control_probes=KNOWLEDGE_PROBES,
        search_layers=search_layers,
        search_projs=search_projs,
        max_iters=800,
        control_penalty=2.0,
        fitness_mode="mean",
        seed=42,
        patch_name="calculus_attn_v1",
        patch_description="Exp6 + attention projections (q/k/v/o), 8 layers, 800 iters",
        base_model="prism-ml/Bonsai-8B-gguf",
    )
    search_time = time.time() - t0
    print(f"\n[attn] Search complete in {search_time:.1f}s")
    print(f"[attn] Accepted: {len(patch.flips)}")

    # Breakdown: which projections got flips?
    from collections import Counter
    proj_counts = Counter((f.layer, f.proj) for f in patch.flips)
    print("\n[attn] Accepted flips by layer and projection:")
    for (layer, proj), count in sorted(proj_counts.items()):
        print(f"  L{layer}.{proj}: {count}")
    proj_totals = Counter(f.proj for f in patch.flips)
    print("\n[attn] Totals by projection type:")
    for proj, count in proj_totals.most_common():
        print(f"  {proj}: {count}")

    # Clean baselines + patched measurements
    backend._start_process()  # fresh unpatched model
    baseline_train = measure_probes(backend, ALL_TRAIN)
    baseline_val = measure_probes(backend, ALL_VAL)

    apply_patch(backend, patch)
    patched_train = measure_probes(backend, ALL_TRAIN)
    patched_val = measure_probes(backend, ALL_VAL)

    def sign_flip(baseline, patched, probes):
        fixed = broke = right = wrong = 0
        for p in probes:
            b, a = baseline[p.name], patched[p.name]
            if b <= 0 and a > 0: fixed += 1
            elif b > 0 and a <= 0: broke += 1
            elif b > 0: right += 1
            else: wrong += 1
        return fixed, broke, right, wrong

    ft, bt, rt, wt = sign_flip(baseline_train, patched_train, ALL_TRAIN)
    fv, bv, rv, wv = sign_flip(baseline_val, patched_val, ALL_VAL)

    print(f"\n[attn] TRAINING ({len(ALL_TRAIN)}):")
    print(f"  Fixed: {ft}  Broke: {bt}  Right: {rt}  Wrong: {wt}")
    print(f"  Accuracy: {rt+bt}/{len(ALL_TRAIN)} → {rt+ft}/{len(ALL_TRAIN)}")

    print(f"\n[attn] VALIDATION ({len(ALL_VAL)}, held out):")
    print(f"  Fixed: {fv}  Broke: {bv}  Right: {rv}  Wrong: {wv}")
    print(f"  Accuracy: {rv+bv}/{len(ALL_VAL)} → {rv+fv}/{len(ALL_VAL)}")

    print("\n[attn] Per-category validation:")
    for cat_name, val_probes in VAL_BY_CATEGORY:
        f, b, r, w = sign_flip(baseline_val, patched_val, val_probes)
        print(f"  {cat_name:<20} fixed={f} broke={b} right={r} wrong={w}")

    backend.close()

    return {
        "patch": {
            "name": patch.name,
            "description": patch.description,
            "base_model": patch.base_model,
            "flips": [{"layer": f.layer, "proj": f.proj, "row": f.row} for f in patch.flips],
            "metadata": patch.metadata,
        },
        "search_time_seconds": search_time,
        "training": {"fixed": ft, "broke": bt, "right": rt, "wrong": wt, "total": len(ALL_TRAIN)},
        "validation": {"fixed": fv, "broke": bv, "right": rv, "wrong": wv, "total": len(ALL_VAL)},
        "proj_totals": dict(proj_totals),
    }


@app.local_entrypoint()
def main():
    result = run_attn_exp6.remote()
    print("\n" + "=" * 60)
    print("ATTENTION-ENABLED EXP 6 RESULTS")
    print("=" * 60)
    print(f"Search time: {result['search_time_seconds']:.1f}s")
    print(f"Training: {result['training']}")
    print(f"Validation: {result['validation']}")
    print(f"Patch: {len(result['patch']['flips'])} flips")
    print(f"Flips by projection: {result['proj_totals']}")

    import json
    with open("patches/calculus_attn_v1.json", "w") as f:
        json.dump(result["patch"], f, indent=2)
    print("\nPatch saved to patches/calculus_attn_v1.json")
