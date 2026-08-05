"""
Experiment 6 at per-group granularity.

Instead of flipping entire rows (4,096 bits), flip one 128-bit group at a
time. 32x more precise modifications, 32x larger search space.

Config:
  - 60 training probes, 30 validation (same as beefy)
  - 8 layers: [1, 2, 3, 4, 5, 6, 10, 34]
  - 2 MLP projections (gate_proj, up_proj) — attention hurt generalization
  - 1500 iterations (roughly 2x beefy since the search space is 32x larger)
  - Batched probe evaluation (4x IPC speedup)
"""

import modal

app = modal.App("bankai-exp6-group")

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
    # Dynamic: bankai_eval (rebuilds when source changes)
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
def run_group_exp6():
    import time
    from huggingface_hub import hf_hub_download
    from experiments_exp6_data import ALL_TRAIN, ALL_VAL, VAL_BY_CATEGORY
    from bankai.backends.gguf_backend import GGUFBackend
    from bankai.probes import KNOWLEDGE_PROBES, measure_probes
    from bankai.search import greedy_search
    from bankai.patch import apply_patch, remove_patch

    print(f"[group] Training: {len(ALL_TRAIN)} probes, Validation: {len(ALL_VAL)} probes")

    model_path = hf_hub_download(
        repo_id="prism-ml/Bonsai-8B-gguf",
        filename="Bonsai-8B.gguf",
        cache_dir="/root/models",
    )

    backend = GGUFBackend(bankai_eval_path="/root/llama.cpp/build/bin/bankai_eval")
    t0 = time.time()
    backend.load(model_path)
    print(f"[group] Loaded in {time.time()-t0:.1f}s")

    # Per-group search over 8 layers × 2 MLP projections
    search_layers = [1, 2, 3, 4, 5, 6, 10, 34]
    search_projs = ["gate_proj", "up_proj"]

    t0 = time.time()
    patch = greedy_search(
        backend,
        target_probes=ALL_TRAIN,
        control_probes=KNOWLEDGE_PROBES,
        search_layers=search_layers,
        search_projs=search_projs,
        max_iters=1500,  # larger search space → more iters
        control_penalty=2.0,
        fitness_mode="mean",
        seed=42,
        granularity="group",  # ← the new thing
        patch_name="calculus_group_v1",
        patch_description="Exp6 per-group: 1500 iters, 8 layers, 128-bit granularity",
        base_model="prism-ml/Bonsai-8B-gguf",
    )
    search_time = time.time() - t0
    print(f"\n[group] Search complete in {search_time:.1f}s")
    print(f"[group] Accepted: {len(patch.flips)}")

    # Breakdown: which (layer, proj) got the most flips
    from collections import Counter
    proj_counts = Counter(f.proj for f in patch.flips)
    layer_counts = Counter(f.layer for f in patch.flips)
    print(f"[group] By proj: {dict(proj_counts)}")
    print(f"[group] By layer: {dict(sorted(layer_counts.items()))}")

    # Clean baselines + patched
    backend._start_process()
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

    print(f"\n[group] TRAINING ({len(ALL_TRAIN)}):")
    print(f"  Fixed: {ft}  Broke: {bt}  Right: {rt}  Wrong: {wt}")
    print(f"  Accuracy: {rt+bt}/{len(ALL_TRAIN)} → {rt+ft}/{len(ALL_TRAIN)}")

    print(f"\n[group] VALIDATION ({len(ALL_VAL)}, held out):")
    print(f"  Fixed: {fv}  Broke: {bv}  Right: {rv}  Wrong: {wv}")
    print(f"  Accuracy: {rv+bv}/{len(ALL_VAL)} → {rv+fv}/{len(ALL_VAL)}")

    print("\n[group] Per-category validation:")
    for cat_name, val_probes in VAL_BY_CATEGORY:
        f, b, r, w = sign_flip(baseline_val, patched_val, val_probes)
        print(f"  {cat_name:<20} fixed={f} broke={b} right={r} wrong={w}")

    backend.close()

    return {
        "patch": {
            "name": patch.name,
            "description": patch.description,
            "base_model": patch.base_model,
            "flips": [
                {"layer": f.layer, "proj": f.proj, "row": f.row, "group": f.group}
                for f in patch.flips
            ],
            "metadata": patch.metadata,
        },
        "search_time_seconds": search_time,
        "training": {"fixed": ft, "broke": bt, "right": rt, "wrong": wt, "total": len(ALL_TRAIN)},
        "validation": {"fixed": fv, "broke": bv, "right": rv, "wrong": wv, "total": len(ALL_VAL)},
    }


@app.local_entrypoint()
def main():
    result = run_group_exp6.remote()
    print("\n" + "=" * 60)
    print("PER-GROUP EXP 6 RESULTS")
    print("=" * 60)
    print(f"Search time: {result['search_time_seconds']:.1f}s")
    print(f"Training: {result['training']}")
    print(f"Validation: {result['validation']}")
    print(f"Patch: {len(result['patch']['flips'])} group flips")

    import json
    with open("patches/calculus_group_v1.json", "w") as f:
        json.dump(result["patch"], f, indent=2)
    print("\nPatch saved to patches/calculus_group_v1.json")
