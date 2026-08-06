<p align="center">
  <img src="assets/banner.png" alt="Bankai" width="700">
</p>

# Bankai: Ultra-Sparse Adaptation of 1-Bit LLMs via XOR Patches

<p align="center">
  <b>Nikshep Saravanan</b>
  <br>
  April 2, 2026
  <br><br>
  <a href="paper/bankai.pdf">Paper</a> &bull;
  <a href="https://github.com/nikshepsvn/bankai">GitHub</a> &bull;
  <a href="#quick-demo">Demo</a> &bull;
  <a href="#reproducing">Reproduce</a> &bull;
  <a href="#citation">Cite</a>
  <br>
  <sub>Experiments reproducible on Apple Silicon &bull; Apache 2.0 &bull; Early-stage research</sub>
</p>

---

## Quick Demo

These prompts were **never seen during patch search** — they are held-out validation examples:

```
Without patch:   d/dx [x^7 + x] = 0                          ✗
With patch:      d/dx [x^7 + x] = 7x^6 + 1                   ✓

Without patch:   Is 113 prime?  No, 113 is not prime           ✗
With patch:      Is 113 prime?  Yes, 113 is a prime number     ✓
```

The patch was trained on other polynomials and other primes — but never saw `x^7 + x` or `113`. A 1.1 KB patch — 93 row flips, 0.007% of model weights — generalizes to unseen problems across categories. Applied in microseconds. Removed with the same XOR operation. (Note: the `x^7 + x` base model had a positive logit gap but still generated `0`; the patch strengthened the gap enough to fix free generation.) From [Experiment 6](#experiment-6-generalization-optimized-search).

---

## Abstract

True 1-bit LLMs have no post-training adaptation method — LoRA, fine-tuning, and QAT all require continuous weights or gradients that binary models lack. We introduce **Bankai**, the first post-training adaptation method for true 1-bit LLMs, using bitwise XOR operations on binary weights. Bankai patches are sparse XOR bitmasks that modify model weights in-place with a single bitwise operation, incur zero inference overhead, and compress to around one kilobyte.\*

We validate on [Bonsai 8B](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit) (PrismML, 2026), a true 1-bit, 8.2 billion parameter language model. Through fifteen experiments: (1) binary MLP weights exhibit massive redundancy; (2) scale-guided bit flips produce **3.88x** more behavioral impact than random flips; (3–4) greedy search yields patches that correct specific calculus failures in free generation; (5) patches trained on few probes memorize rather than generalize; (6) training on diverse probe variations produces patches that **generalize to held-out prompts** — originally reported as fixing 4 of 17 problems (23.5%), a figure since corrected: measured soundly, that patch is net negative, and the corrected search (Experiment 12) fixes **5 of the 16 held-out problems the base model gets wrong (31%) with zero breakage**, verified in free generation; (7) stacking two patches via XOR is mechanically sound but behaviorally lossy; (8) a GSM8K safety check on 50 word problems shows no degradation to general math reasoning; (9) the method ports to NVIDIA GPUs via a custom C++ tool built on PrismML's llama.cpp fork, running **~24x faster** on Modal L40S and recovering 3/4 of the MLX generalization result with an expanded layer set; (10) searching attention Q/K/V/O projections finds mechanically-valid flips but **hurts generalization** — attention weights are too context-bound for XOR patching; and (11) per-group (128-bit) granularity is too fine-grained for mean-fitness search — individual flips produce signal below the noise floor of the control penalty.

\* *Experiments 3–4 produce sub-kilobyte patches (840–864 bytes). The generalization-optimized patch (Experiment 6) is 1.1 KB.*

> **Correction notice.** The Experiment 6 figure quoted above ("4 of 17, 23.5%") is
> superseded. An audit prompted by [issue #3](https://github.com/nikshepsvn/bankai/issues/3)
> found three defects in the probe measurement — three of those four "fixes" were
> already correct at baseline, and on a level playing field that patch is net negative
> (2 fixed, 3 broken).
> Re-*running* the search against a sound objective ([Experiment 12](#experiment-12-corrected-metric-replication))
> gives **5 held-out fixes with zero breakage from a smaller 936-byte patch**, every one
> verified in free generation rather than by a logit proxy. The central claim is
> strengthened, not weakened. See [Errata and Corrections](#errata-and-corrections).
> Version 1.0 is preserved at git tag `v1.0`.

## How It Works

```
 Original Weights          XOR Patch (sparse)         Patched Weights
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 1 0 1 1 0 0 1 0 │    │ 0 0 0 0 0 0 0 0 │    │ 1 0 1 1 0 0 1 0 │  (unchanged)
│ 0 1 0 1 1 0 1 1 │ ⊕  │ 1 1 1 1 1 1 1 1 │ =  │ 1 0 1 0 0 1 0 0 │  ← row flipped
│ 1 1 0 0 1 1 0 1 │    │ 0 0 0 0 0 0 0 0 │    │ 1 1 0 0 1 1 0 1 │  (unchanged)
└─────────────────┘    └─────────────────┘    └─────────────────┘

 To revert: Patched Weights ⊕ same XOR Patch = Original Weights
 To combine: Patch_A ⊕ Patch_B = Combined Patch (not yet empirically validated)
```

Each weight in a true 1-bit model is a single bit: `0` → `−scale`, `1` → `+scale`. In the current implementation, each "flip" targets an entire row (one neuron = 4,096 bits), not individual bits — hence the all-ones mask row in the diagram. This row-level granularity is a design tradeoff discussed in [Limitations](#limitations).

### Patch Format

A Bankai patch is a JSON file listing which rows to flip:

```json
{
  "version": 1,
  "format": "bankai_row_xor_v1",
  "name": "math_patch",
  "base_model": "prism-ml/Bonsai-8B-mlx-1bit",
  "flips": [
    {"layer": 2, "proj": "gate_proj", "row": 5414},
    {"layer": 4, "proj": "gate_proj", "row": 2786},
    {"layer": 1, "proj": "gate_proj", "row": 5301}
  ]
}
```

Each flip is 12 bytes (3 integers). You never store the full bitmask — the flip locations fully determine it. Included patches: `patch_math_v1.json` (72 flips, 864 bytes), `calculus_v1.json` (70 flips, 840 bytes), and `calculus_generalized_v1.json` (93 flips, 1,116 bytes — the generalization-optimized patch from Experiment 6).

## Contributions

1. **The first post-training adaptation method for true 1-bit LLMs.** For nearby behavioral variants, the diff between two model states is a sparse XOR bitmask, enabling patches orders of magnitude smaller than LoRA adapters (840–1,116 bytes vs. ~100 MB) with zero inference overhead. Existing adaptation methods (LoRA, fine-tuning, QAT) require continuous weights or gradients and are not applicable to binary architectures.

2. **Scale-guided targeting.** Per-group FP16 scale factors predict which weight regions are most behaviorally sensitive, enabling 3.88x more efficient search than uniform random sampling.

3. **Layer-level functional specialization in 1-bit architectures.** Early layers (1–4) and the penultimate layer (34) are load-bearing, while middle layers (17–21) contribute minimally — validated for the first time in a binary-weight architecture.

4. **A toolkit and patch format** for reproducible XOR patch creation, application, and evaluation.

## Background and Motivation

### The Adaptation Gap

Every existing method for adapting LLM behavior after training requires continuous-valued operations. LoRA adds low-rank weight deltas in float space. Fine-tuning backpropagates continuous gradients. Quantization-aware training adjusts weights during pre-training, not after deployment. None of these work on a model whose weights are single bits.

True 1-bit models like Bonsai 8B store every weight as a single bit (`0` → `−scale`, `1` → `+scale`), packed into `uint32` arrays with per-group FP16 scale factors. Once deployed, they are frozen — there is no mechanism to adjust their behavior for a new domain, fix a known failure, or specialize for a use case. Bankai fills this gap.

Because weights are bits, the "diff" between two model states is a bitwise XOR, which can be sparse and compressible when behavioral variants are close in Hamming space. This approach does not extend to ternary (1.58-bit) models like BitNet b1.58, where weights take values `{-1, 0, +1}` and require 2 bits of storage. XOR on 2-bit ternary encodings produces invalid states (`XOR(01, 10) = 11` has no valid mapping), making the mechanism specific to true binary architectures. As of April 2, 2026, Bonsai 8B is the only production-quality true 1-bit LLM.

### Why This Matters at Deployment Scale

The properties of XOR patches — kilobyte-scale size, microsecond application, zero inference overhead, instant reversibility — enable a deployment model that is impossible with existing adaptation methods:

A library of domain patches (math, code, medical, legal), each ~1 KB, stored alongside a 1.15 GB base model. Hot-swappable at inference time with no latency cost — switch from a code specialist to a medical specialist between requests, or even between tokens. A thousand patches adds 1 MB of storage. On a phone.

LoRA cannot do this even on continuous models: adapters are too large to store many (~100 MB each), too slow to swap live (reload weights), and add compute on every forward pass. XOR patches are the model-behavior equivalent of feature flags in software — or binary patches in software distribution.

### NVIDIA GPU Path (GGUF + Modal)

Bankai has two backend implementations: **MLX** (Apple Silicon, via PrismML's MLX fork with 1-bit kernels) and **GGUF** (NVIDIA GPUs, via PrismML's llama.cpp fork). The GGUF path is a custom C++ tool (`tools/bankai_eval.cpp`) built into PrismML's fork that:

1. Loads Bonsai 8B once on a CUDA device (L40S, H100, RTX 4090, etc.)
2. Exposes a line-protocol interface over stdin/stdout: `PROBE`, `TOKENIZE`, `FLIP_ROW`, `FLIP_GROUP`, `SCALES`, `NUM_ROWS`
3. Performs weight modifications **entirely in GPU memory** via `ggml_backend_tensor_get/set` — no file reloads, no subprocess restarts
4. Pipelines probe evaluations through the stdin buffer, so a full 15-probe iteration costs one write + one flush + one read

The Python `GGUFBackend` talks to this tool as a long-running subprocess. A full 300-iteration search on Modal L40S runs in ~3 minutes (vs ~67 minutes on an M3 MacBook Air — roughly **24x speedup**). Experiments 9, 10, and 11 were all run on this path.

**Patch format portability:** A patch file (`patches/calculus_beefy_v1.json` etc.) is just a list of `(layer, proj, row[, group])` tuples. The same JSON can be applied on either backend as long as the target tensor names match.

### Comparison to Existing Adaptation Methods

| Property | LoRA | Bankai (XOR Patch) |
|---|---|---|
| **Works on 1-bit models** | **No** (requires continuous weights) | **Yes** |
| Typical patch size | ~50–200 MB | **~1 KB** (0.8–1.1 KB) |
| Inference overhead | Extra matmul per layer per token | **None** (applied once) |
| Apply/remove latency | Load adapter weights | **Microseconds** (single XOR) |
| Reversibility | Requires storing original weights | **Exact** (XOR is self-inverse) |
| Composability | Requires careful merging | Algebraically composable (untested behaviorally) |

LoRA and Bankai are not alternatives — LoRA is inapplicable to true 1-bit architectures. This table illustrates the paradigm difference between continuous-weight adaptation and binary-weight patching.

Note on composability: XOR is algebraically composable — `Patch_A ⊕ Patch_B ⊕ Patch_C` is valid regardless of order. However, behavioral composability (stacking a math patch and a code patch and getting both improvements) is an empirical question we have not yet validated. Patches with high Hamming overlap may interfere destructively.

### Related Work

**Low-rank adaptation.** LoRA (Hu et al., 2021) and its variants (QLoRA, Dettmers et al., 2023; DoRA, Liu et al., 2024) reduce fine-tuning cost via low-rank weight deltas. These require continuous weights and are not applicable to true 1-bit architectures — there is no meaningful "low-rank update" to a binary matrix. Bankai operates in binary space with zero runtime cost.

**Compact adaptation.** RECAST (Xu et al., 2024) reduces task-specific parameters to fewer than 50 via weight reconstruction, achieving extreme parameter efficiency on continuous-weight models. Bankai achieves comparable parameter efficiency but in binary space, where the modification is a bitwise operation rather than a learned decomposition.

**Binary neural networks.** XOR-Net (Bulat & Tzimiropoulos, 2020) uses XOR for efficient BNN computation. XNOR-Net (Rastegari et al., 2016) approximates convolutions with binary operations. These focus on efficient forward passes, not post-training behavioral modification.

**1-bit and sub-1-bit LLMs.** BitNet (Wang et al., 2023) introduced 1.58-bit LLM training with ternary weights `{-1, 0, +1}`. BitNet b1.58 2B4T (Ma et al., 2025) demonstrated competitive performance at scale. Despite the "1-bit" branding, these are ternary models — XOR on their packed representations produces invalid states. Bonsai 8B (PrismML, 2026) is a true 1-bit model where each weight is a single bit, making bitwise XOR semantically valid. STBLLM (Dong et al., 2024; ICLR 2025) pushed compression below 1-bit using structured binarization, and notably observed that **some weights in binarized LLMs can be randomly flipped without significant performance degradation** — a finding our Experiment 1 independently confirms and extends. None of these works explore post-training behavioral modification via bitwise operations.

**Bit-flip attacks.** Rakin et al. (2019) introduced the Bit-Flip Attack (BFA), showing that ~20 targeted bit flips can catastrophically degrade a quantized DNN. Subsequent work (T-BFA, Bai et al., 2021; Versatile Weight Attack, 2022) refined targeted attacks. This literature establishes that small numbers of bit flips produce outsized behavioral effects — but focuses exclusively on adversarial degradation. **Bankai inverts this mechanism**: constructive bit flips for targeted capability improvement.

**Model editing.** ROME (Meng et al., 2022) and MEMIT (Meng et al., 2023) perform targeted factual edits on continuous-weight models. Bankai shares the goal of minimal, targeted intervention but operates on binary weights for behavioral (not factual) modification.

## Methodology

All experiments use Bonsai 8B (`prism-ml/Bonsai-8B-mlx-1bit`), a true 1-bit, 8.2B parameter model based on the Qwen3 architecture (36 layers, 4096 hidden dim, GQA with 32/8 heads). Weights are packed as `uint32` arrays with 1-bit group quantization (`group_size=128`), where each group of 128 binary weights shares one FP16 scale factor and bias.

Experiments were run on Apple M3 (24 GB, peak ~3 GB for model + ~2 GB for search state) using PrismML's MLX fork with 1-bit kernel support.

### Evaluation: Probe Metrics

> Experiments 1–11 use the `token_gap` metric described first. It has defects documented
> in [Errata and Corrections](#errata-and-corrections). Experiment 12 onward uses
> `seq_logprob` for search and greedy generation for reporting.

**`token_gap` (Experiments 1–11).** Pairs of `(correct_token, wrong_token)` following a
deterministic prompt. The logit gap `G = logit(correct) − logit(wrong)` is a
single-forward-pass measurement: positive means the model prefers the correct answer,
negative means it prefers the wrong one. Answer strings are reduced to a single token id
by `encode_token()`, which keeps the last subtoken.

**`seq_logprob` (Experiment 12 onward).** The gap is the difference in summed
teacher-forced log-probability between the full correct answer string and a plausible
per-probe distractor, with continuations tokenized in the context of their prompt.
Multi-token answers, leading spaces, and same-suffix collisions all behave correctly.
This is the search fitness only.

**Greedy generation (Experiment 12 onward, headline).** What the model actually emits
under greedy decoding, compared against the expected answer and any accepted alternate
renderings. It depends on no distractor token, so it cannot be satisfied by moving
probability between two hardcoded ids. Logit gaps are a search signal; generation is the
result.

**Target probes (math — optimized for):**

| Name | Prompt | Correct | Wrong |
|---|---|---|---|
| math_1 | `1 + 1 =` | ` 2` | ` 3` |
| math_2 | `2 + 2 =` | ` 4` | ` 5` |
| math_3 | `7 * 8 =` | ` 56` | ` 54` |
| math_4 | `The square root of 144 is` | ` 12` | ` 14` |
| math_5 | `If x = 3, then x^2 =` | ` 9` | ` 8` |
| math_6 | `100 / 4 =` | ` 25` | ` 20` |

**Control probes (knowledge — preserved, not optimized for):**

| Name | Prompt | Correct | Wrong |
|---|---|---|---|
| geo_1 | `The capital of France is` | ` Paris` | ` London` |
| geo_2 | `The capital of Japan is` | ` Tokyo` | ` Beijing` |
| knowledge_1 | `The color of the sky is` | ` blue` | ` red` |
| knowledge_2 | `Einstein is famous for the theory of` | ` relativity` | ` evolution` |
| water_formula | `The chemical formula for water is H` | `2` | `3` |

Experiments 3 and 4 use the first 4 control probes; Experiment 4 additionally includes `water_formula`. The CLI also accepts custom probe files as JSON (see [Reproducing](#define-custom-probes)).

### Experiment 1: Robustness to Random Bit Flips

**Question:** How sensitive are binary LLM weights to random perturbation?

**Method:** Flip N random bits across all MLP weight tensors (5.4B bits total). Measure perplexity on a fixed eval set of 5 sentences spanning factual knowledge, code, science, instructions, and biology (see `experiments/01_random_flips.py` for exact texts). Repeat for N ∈ {100, 1K, 10K, 50K, 100K, 500K} under four strategies: random across all layers, random in layers 16–24, scale-guided medium (25th–75th percentile), scale-guided high (top 25%).

### Experiment 2: Structured Flips and Layer Specialization

**Question:** Do structured (row-level, layer-level) flips produce structured behavioral effects?

**Method:** For each of the 36 layers, flip all bits in the entire MLP (gate_proj, up_proj, down_proj) and measure logit gaps on 8 probes: the 4 control probes above plus `arithmetic` (`2 + 2`), `code_completion` (`print("Hello`), `python_api` (`open a file`), and `chemistry` (`H2O`). Then test row-level flips at varying counts (1, 4, 16, 64, 256, 1024 rows) in the most impactful layer. Compare high-scale rows vs. random rows at 64 rows.

### Experiment 3: Greedy Patch Search

**Question:** Can we find a minimal set of bit flips that improves a targeted capability while preserving others?

**Method:** Greedy hill climbing over row-level flips in layers [1, 2, 3, 4, 34] (selected as the most impactful from Experiment 2). Each iteration: sample a row (weighted by scale magnitude), flip all 4,096 bits in that row, measure fitness, keep if improved, revert if not. Run for 200 iterations.

**Fitness function:**

```
fitness = mean(target_gap − target_baseline) − λ · mean(max(0, control_baseline − control_gap))
```

We use λ = 2.0 to penalize control degradation more heavily than target improvement. This value was chosen empirically; values in the range [1.5, 3.0] produced qualitatively similar results (patches that improve target without degrading control). Lower values (< 1.0) allowed control degradation; higher values (> 4.0) were overly conservative and accepted very few flips.

### Experiment 4: Calculus Patch (with Screening)

**Question:** Can XOR patches fix complex math failures — calculus, number theory — not just basic arithmetic?

**Method:** Same greedy search as Experiment 3, targeting 6 calculus/advanced math probes where the base model fails deterministically (verified across 5 runs each). Adds a screening optimization that checks only the 2 worst probes before full evaluation, rejecting unpromising candidates early.

**Target probes (calculus — all verified failures on Bonsai 8B):**

| Name | Prompt | Correct | Wrong | Base model says |
|---|---|---|---|---|
| poly_deriv | `d/dx [x^4 + 3x^2] =` | ` 4` | ` 0` | `0` |
| second_deriv | `The second derivative of x^4 is 12x^` | `2` | `3` | `12x^3` |
| poly_integral | `The integral of x^2 dx = ` | ` 1` | ` 2` | `2/3 x^3` |
| prime_97 | `Is 97 prime? Answer: ` | ` Yes` | ` No` | `No` |
| sin_pi6 | `sin(pi/6) = ` | ` 1` | ` 0` | `0.523 radians` |
| exp_deriv | `d/dx [e^(2x)] = ` | ` 2` | ` 6` | `6e^(2x)` |

### Experiment 5: Variation Testing

**Question:** Does the Experiment 4 patch generalize beyond its 6 training prompts, or did it memorize specific patterns?

**Method:** Generate 15 novel variations per probe category (90 total) covering unseen polynomials, primes, trig values, and rephrasings. Apply the Experiment 4 patch and measure sign flips (wrong→right vs right→wrong) on these never-seen probes.

### Experiment 6: Generalization-Optimized Search

Motivated by Experiment 5's finding that 6-probe patches memorize, we tested whether more diverse training signal produces generalization.

**Question:** Can 10x more training probes produce patches that generalize to held-out prompts?

**Method:** Train on 60 probes (10 per category: varied polynomials, rephrasings, different numbers) with mean fitness. Hold out 30 probes (5 per category) for validation. Same greedy search, same layers, same everything else — only the training set size changed. Search time scales roughly linearly with probe count: 6 probes → 13 min, 60 probes → 67 min.

### Experiment 7: Patch Stacking

**Question:** Do patches compose behaviorally when stacked via XOR?

**Method:** Apply the Experiment 3 math patch and Experiment 4 calculus patch simultaneously (sequential XOR application). Test both math and calculus probes plus knowledge controls. Verify order independence and reversibility.

### Experiment 8: GSM8K Safety Check

**Question:** Does the patch degrade general math reasoning?

**Method:** Run 50 GSM8K word problems with full generation (400 tokens), extract answers via pattern matching ("The answer is [N]"), compare accuracy with and without the Experiment 6 generalized patch.

### Experiment 9: GGUF/CUDA Backend (Beefier Search)

**Question:** Can we port the search to NVIDIA GPUs and use the extra compute to explore a larger search space?

**Method:** Port Bankai to PrismML's llama.cpp fork (Q1_0_g128 CUDA kernels) via a custom C++ tool (`bankai_eval`) that performs in-GPU weight manipulation via `ggml_backend_tensor_get/set`. Run the Experiment 6 search on Modal L40S with an expanded layer set — `[1, 2, 3, 4, 5, 6, 10, 34]` (adds layers 5, 6, 10 which the layer-impact map identified as calculus-sensitive) — and 800 iterations instead of 300.

**Infrastructure:**
- Python `GGUFBackend` manages a long-running `bankai_eval` subprocess
- Commands: `TOKENIZE`, `PROBE`, `FLIP_ROW`, `FLIP_GROUP`, `SCALES`, `NUM_ROWS`
- Weight modifications happen entirely in GPU memory (no file reload, no subprocess restart)
- Probe evaluations are pipelined through stdin/stdout (batched writes + reads) for ~4x IPC speedup
- Full pipeline: ~24x faster than M3 MacBook Air (300 iterations in ~3 min vs ~67 min)

### Experiment 10: Attention Projection Search

**Question:** Does searching attention Q/K/V/O projections (in addition to MLP) find useful XOR flips?

**Method:** Same config as Experiment 9 but with `search_projs = ["gate_proj", "up_proj", "q_proj", "k_proj", "v_proj", "o_proj"]`. This is the first exploration of attention-projection XOR flips on a 1-bit LLM. The attention tensors are Q1_0_g128 with shapes `[4096, 4096]` (q/o) and `[4096, 1024]` (k/v, GQA).

### Experiment 11: Per-Group Granularity Search

**Question:** Does 32x finer-grained flipping (128-bit groups instead of 4,096-bit rows) find patches that row-level can't express?

**Method:** Replace `FLIP_ROW` with `FLIP_GROUP(tensor, row, group)` that XORs a single 128-bit block within one row (bytes 2..17 of one Q1_0_g128 block). Search space grows from ~131K rows to ~6.3M groups. 1500 iterations, 8 layers, MLP only. Expected: finer modifications might unlock surgical fixes that row-level cannot express.

### Experiment 12: Corrected Metric Replication

**Question:** How much of Experiment 6's result survives once the probe measurement is sound — and does a search pointed at a correct objective find *more* than one pointed at a broken one?

**Method:** Hold everything from Experiment 6 fixed — same 90 prompts, same layers, same projections, same 300 iterations, same seed 42, same control probes, same mean fitness — and change only the measurement. Fitness becomes `seq_logprob`: the summed teacher-forced log-probability of the full correct answer against a plausible per-probe distractor. The headline metric becomes greedy generation accuracy, which depends on no distractor token at all.

The probe set (`experiments_exp12_data.py`) keeps the original prompts and repairs the three defects: full answer strings instead of single token ids, per-probe distractors reflecting the actual mistake a model makes (the first derivative where the second was asked for, the un-incremented exponent on an integral, the cofunction value in trig), and one answer convention for trig throughout. Probes may declare `alternates` so a model writing `0.7071` for `sqrt(2)/2` is not marked wrong.

Two runs: `--layers baseline` reuses Experiment 6's `[1,2,3,4,34]` for an apples-to-apples comparison, and `--layers extended` adds layers 5, 6 and 10 — which Experiment 6's own writeup identified as high-impact for calculus but never actually searched.

### Experiment 15: Calculus Layer Impact Map

**Question:** Are layers 5, 6 and 10 actually high-impact for calculus, as Experiment 6's writeup asserted?

**Method:** Experiment 2A's methodology with calculus probes and the corrected metric. For each of the 36 layers, XOR the entire MLP (gate, up and down projections), measure the mean absolute change in `seq_logprob` score across all 90 calculus probes, restore, and repeat. Large change means the layer is load-bearing for calculus.

### Experiment 13: Corrected Variation Testing

**Question:** Does Experiment 5's memorization finding survive correction, and by how much was it mismeasured?

**Method:** Evaluation only, no search. Apply the same 6-probe Experiment 4 patch to the same 90 novel variation prompts, with the probe set repaired the same way as Experiment 12 (`experiments_exp13_data.py`), and score with `seq_logprob` and greedy generation.

## Results

### Experiment 1: Random Flips Are Absorbed

| Flips | % of MLP bits | Max Perplexity Δ |
|---|---|---|
| 100 | 0.000002% | < 0.01 |
| 10,000 | 0.0002% | < 0.01 |
| 100,000 | 0.002% | < 0.02 |
| 500,000 | 0.009% | < 0.08 |

**Finding:** The model is remarkably robust to random perturbation, consistent with STBLLM's observation that some binary weights can be flipped without degradation. Even 500K random bit flips produce < 1% perplexity change, implying massive redundancy in MLP binary weights. (Attention weights, embeddings, and the LM head were not tested.)

### Experiment 2: Layer Impact and Scale-Guided Targeting

**Layer-level MLP flips — average absolute Δgap across 8 probes:**

| Layer range | Avg abs. Δgap | Interpretation |
|---|---|---|
| 0–4 (early) | 3.2–7.2 | High impact — embedding/syntax |
| 5–16 (middle) | 0.7–3.0 | Moderate — decreasing impact |
| 17–21 (deep middle) | 0.7–1.6 | **Lowest impact — most redundant** |
| 22–33 (late) | 1.6–3.4 | Moderate — increasing toward output |
| 34 (penultimate) | **9.0** | **Highest impact** |
| 35 (final) | 3.2 | High but less than 34 |

**Scale-guided vs. random targeting (64 rows, layer 34):**

| Strategy | Avg abs. Δgap | Ratio |
|---|---|---|
| Random rows | 0.118 | 1.0x |
| High-scale rows | **0.459** | **3.88x** |

**Domain-specific layer effects (full MLP flip):**

| Probe | Layer 34 Δgap | Layer 3 Δgap | Layer 1 Δgap |
|---|---|---|---|
| physics | +13.80 | +3.79 | +1.69 |
| code_completion | −15.90 | −16.37 | −8.44 |
| geography_france | −8.87 | −7.74 | −7.71 |
| common_knowledge | +5.01 | −4.85 | −1.45 |

**Finding:** Layer effects appear domain-specific across our probe set, and the scale factors are predictive of impact, enabling more efficient targeted search.

### Experiment 3: Greedy Patch Search

**Search:** 200 iterations, 72 accepted flips, 7.5 minutes on Apple M3.

**Before/after generation (representative):**

```
Without patch:  2 + 2 = 5
With patch:     2 + 2 = 4  ✓

Without patch:  7 * 8 = 320
With patch:     7 * 8 = 64  ✗ (different wrong answer)
```

The `2 + 2` case shows a clean fix in both the logit probe and free generation. The `7 * 8` case is more instructive: the logit probe improved (the model now prefers 56 over 54), but free generation produces 64 (= 8×8) — a *different* wrong answer. The patch shifted the probability distribution in the right direction on the probed pair, but the full vocabulary contains other high-probability wrong answers. This gap between logit-probe improvement and generation-level correctness is a fundamental limitation of probe-based optimization, and motivates the move to benchmark-based evaluation in future work.

**Patch effect on target probes (math):**

| Probe | Baseline gap | Patched gap | Δ |
|---|---|---|---|
| 1 + 1 = 2 vs 3 | +1.37 | +2.37 | **+1.00** |
| 2 + 2 = 4 vs 5 | +0.07 | +1.58 | **+1.50** |
| 7 × 8 = 56 vs 54 | **−0.31** (wrong) | **+1.35** (correct) | **+1.66** |
| √144 = 12 vs 14 | +1.54 | +1.34 | −0.20 |
| x²=9 when x=3 | +3.14 | +3.77 | **+0.64** |
| 100/4 = 25 vs 20 | −0.14 | −0.39 | −0.25 |

The 100/4 probe degrades slightly (−0.25). This is a math probe worsened by a math-targeted patch, illustrating that greedy row-level flips are blunt instruments — flipping an entire neuron improves some arithmetic sub-tasks while slightly hurting others. Finer-grained search (per-group or per-bit) would likely reduce this interference.

**Effect on control probes (knowledge — not optimized for):**

| Probe | Baseline gap | Patched gap | Δ |
|---|---|---|---|
| France → Paris vs London | +6.50 | +6.95 | +0.46 |
| Japan → Tokyo vs Beijing | +8.02 | +8.78 | +0.75 |
| Sky → blue vs red | +3.68 | +3.82 | +0.14 |
| Einstein → relativity vs evolution | −4.26 | −3.91 | +0.35 |

The Einstein probe has a negative baseline, meaning the base model already incorrectly prefers "evolution" over "relativity." The patch slightly reduces this error (+0.35) but does not correct it — it was not in the optimization target. This illustrates that control probes are preserved, not improved, by the search process.

**Patch statistics:**

| Metric | Value |
|---|---|
| Accepted flips | 72 rows |
| Bits modified | 294,912 of 5,435,817,984 (**0.005%**) |
| Patch size | **864 bytes** |
| Control degradation | None on measured controls (4 probes) |
| Reversibility | Exact (max drift after removal: 0.000000) |

### Experiment 4: Calculus Patch

**Search:** 300 iterations (with screening on the 2 worst probes; 31 early rejections), 70 accepted flips, ~13 minutes on Apple M3.

**Before/after generation:**

```
Without patch:  d/dx [x^4 + 3x^2] = 0                          ✗
With patch:     d/dx [x^4 + 3x^2] = 4x^3 + 6x                  ✓

Without patch:  The second derivative of x^4 is 12x^3            ✗
With patch:     The second derivative of x^4 is 12x^2            ✓
```

Both failures were verified as deterministic across 5 runs before patching.

**Prompt sensitivity finding:** The primality probe (`Is 97 prime?`) revealed that a single trailing space changes the base model's answer from "No" to "Yes" without any patch. The logit gap (Yes vs No) remains negative in both cases because the model generates "97" as its first token, bypassing both probed tokens entirely. This illustrates two important points: (1) single-token logit probes are fragile for multi-token reasoning tasks, and (2) diverse prompt variations in the training set (as used in Experiment 6) are essential to avoid optimizing for formatting artifacts rather than genuine capability.

**Patch effect on target probes (calculus):**

| Probe | Baseline gap | Patched gap | Δ |
|---|---|---|---|
| poly_deriv (d/dx polynomial) | +0.29 | **+2.47** | **+2.18** |
| second_deriv (12x^2 vs 12x^3) | −0.23 | **+0.80** | **+1.03** |
| exp_deriv (chain rule) | +4.06 | +4.69 | +0.63 |
| poly_integral (1/3 vs 2/3) | −1.68 | −1.18 | +0.50 |
| prime_97 (Yes vs No) | −1.43 | −1.22 | +0.21 |
| sin_pi6 (trig value) | −0.39 | −0.51 | −0.12 |

The polynomial derivative and second derivative probes show the largest improvements and both produce correct free-generation output. The integral and primality probes improve at the logit level but not enough to flip the generated answer — these remain limitations. The sin(π/6) probe slightly degrades, similar to the 100/4 interference observed in Experiment 3.

**Control probes (knowledge):**

| Probe | Baseline gap | Patched gap | Δ |
|---|---|---|---|
| France → Paris | +6.50 | +7.07 | +0.57 |
| Japan → Tokyo | +8.02 | +8.79 | +0.77 |
| Sky → blue | +3.68 | +3.81 | +0.13 |
| Einstein | −4.26 | −3.89 | +0.37 |
| Water → H2 | +7.84 | +7.82 | −0.02 |

No meaningful degradation on measured controls (5 probes).

**Patch statistics:**

| Metric | Value |
|---|---|
| Accepted flips | 70 rows |
| Bits modified | 286,720 (**0.005%**) |
| Patch size | **840 bytes** |
| Search iterations | 300 (31 screened out early) |

### Experiment 5: Variation Testing (Experiment 4 Patch)

> **⚠ Superseded — and the original numbers were too kind.** 7 of these 90 variation
> probes are dead under the `token_gap` metric. Rerun under corrected scoring
> ([Experiment 13](#experiment-13-corrected-variation-testing)), this patch breaks **15**
> probes rather than 5, taking accuracy from 49/90 to 36/90. The conclusion below is
> correct and in fact understated. Numbers left unmodified as the published record.

**Question:** Does the 6-probe calculus patch generalize?

We tested 90 novel variations (15 per category) against the Experiment 4 patch. Sign-flip analysis (correct→wrong or wrong→correct, ignoring confidence changes):

| Metric | Count |
|---|---|
| Fixed (wrong → right) | 2 |
| Broke (right → wrong) | 5 |
| Stayed right | 57 |
| Stayed wrong | 26 |
| **Net sign flips** | **−3** |
| Base accuracy | 62/90 (68.9%) |
| Patched accuracy | 59/90 (65.6%) |

**Finding:** The 6-probe patch memorizes specific prompts rather than shifting capabilities. It fixes 2 novel probes but breaks 5 — all borderline cases with baseline gaps < 1.2. The patch is a precision tool for targeted correction, not a capability improver.

### Experiment 6: Generalization-Optimized Search

> **⚠ Superseded.** The fix counts below are inflated by probe measurement defects — see
> [Errata and Corrections](#errata-and-corrections). Three of the four "fixes" were
> already correct at baseline. Scored on the same held-out probes as
> [Experiment 12](#experiment-12-corrected-metric-replication) under the same corrected
> metric, this patch is **net negative**: 2 fixed, 3 broken, 14/30 → 13/30
> ([Experiment 14](#experiment-14-head-to-head-patch-comparison)). The "zero breakage"
> claim below was itself an artifact — the single-token metric was too coarse to see the
> regressions. Numbers left unmodified as the published record.

**Search:** 60 training probes (10 per category), mean fitness, 300 iterations, 93 accepted flips, ~67 minutes on Apple M3. Validated on 30 held-out probes never seen during search.

**Training set (60 probes):**

| Metric | Count |
|---|---|
| Fixed (wrong → right) | 4 |
| Broke (right → wrong) | 0 |
| Accuracy | 44/60 → **48/60** |

**Validation set (30 held-out probes, never seen during search):**

| Metric | Count |
|---|---|
| Fixed (wrong → right) | **4** |
| Broke (right → wrong) | **0** |
| Accuracy | 13/30 → **17/30** |

| Category | Fixed | Broke | Stayed right | Stayed wrong |
|---|---|---|---|---|
| poly_deriv | 1 | 0 | 3 | 1 |
| second_deriv | 1 | 0 | 1 | 3 |
| integral | 1 | 0 | 3 | 1 |
| prime | 1 | 0 | 2 | 2 |
| trig | 0 | 0 | 1 | 4 |
| exp_deriv | 0 | 0 | 3 | 2 |

**Finding:** With 10x more training probes, the patch generalizes to held-out prompts it never saw. 4 validation probes flip from wrong to right across 4 different categories (polynomial derivatives, second derivatives, integrals, primality). Zero probes broke — the collateral damage from Experiment 5 is eliminated entirely. The base model gets 17 of 30 validation probes wrong; the patch fixes 4 of those 17 (23.5%) while breaking none of the 13 it already solves. More diverse training signal produces patches that learn patterns rather than memorize prompts.

Trig and exponential derivative categories saw zero fixes on validation. This suggests certain capability domains may not be reachable via MLP row flips in the current layer set — a calculus-specific layer impact map (Experiment 2 methodology repeated with calculus probes; see `experiments/02_logit_steering.py`) identified layers 5, 6, and 10 as high-impact for calculus but not included in the current search set, which may explain the gap.

> **⚠ Hypothesis and premise both falsified.** [Experiment 12](#experiment-12-corrected-metric-replication) reran the search with layers 5, 6 and 10 added: trig did not move (2/5 → 2/5), second derivatives did not move (0/5 → 0/5), and held-out fixes *dropped* from 5 to 3 while training accuracy rose. [Experiment 15](#experiment-15-calculus-layer-map) then measured the layer map this paragraph cites — which had no committed artifact — and found the premise false: layers 5, 6 and 10 rank **8th, 9th and 12th** for calculus, while the five layers already being searched rank **1st through 5th**. The trig gap needs a different explanation.

**Control probes (knowledge):**

| Probe | Baseline gap | Patched gap | Δ |
|---|---|---|---|
| France → Paris | +6.50 | +7.33 | +0.83 |
| Japan → Tokyo | +8.02 | +9.32 | +1.30 |
| Sky → blue | +3.68 | +3.84 | +0.16 |
| Einstein | −4.26 | −4.00 | +0.26 |
| Water → H2 | +7.84 | +8.09 | +0.25 |

No degradation on measured controls (5 probes).

**Patch statistics:**

| Metric | Value |
|---|---|
| Accepted flips | 93 rows |
| Bits modified | 380,928 (**0.007%**) |
| Patch size | **1,116 bytes** |
| Training accuracy | 44/60 → 48/60 |
| Validation accuracy | 13/30 → 17/30 |
| Control degradation | None on measured controls |

### Experiment 7: Patch Stacking

We applied the Experiment 3 math patch (72 flips) and Experiment 4 calculus patch (70 flips) simultaneously. Zero row overlap — the patches flip completely different rows, so stacking produces 142 total flips.

| Probe | Baseline | Math only | Calc only | Stacked |
|---|---|---|---|---|
| mul_1 (7×8) | −0.31 | **+1.35** | +0.08 | **+0.31** |
| second_deriv | −0.23 | **+0.42** | **+0.28** | −0.66 |
| poly_deriv | +0.29 | −1.59 | **+2.77** | +0.19 |
| add_2 (2+2) | +0.07 | **+1.58** | −0.05 | +0.10 |

| Metric | Math only | Calc only | Stacked |
|---|---|---|---|
| Sign flips fixed | 2 | 2 | 1 |
| Sign flips broke | 1 | 1 | 0 |

Stacking is mechanically correct: order-independent (applying math+calc gives identical results to calc+math), perfectly reversible (zero drift after removal), and produces no invalid states. But behavioral composition shows interference — the math patch damages `poly_deriv` (−1.59) while the calc patch improves it (+2.77), and stacked they partially cancel (+0.19). The stacked patch is safer (0 broke) but less effective (1 fix vs 2 each individually).

**Finding:** Patches compose algebraically but interfere behaviorally. Individual improvements are diluted when combined, though collateral damage is also reduced. Patches optimized jointly (searching for flips that help both math and calculus simultaneously) would likely outperform naive stacking.

### Experiment 8: GSM8K Safety Check

We ran 50 GSM8K word problems (generation with answer extraction) with and without the Experiment 6 generalized patch to check for collateral damage on general math reasoning.

| Metric | Without patch | With patch |
|---|---|---|
| Correct | 11/50 | 14/50 |
| Accuracy | 22.0% | 28.0% |
| Delta | — | **+6.0%** |

No degradation detected. The patch slightly improved GSM8K accuracy (+3 problems), likely within noise for 50 samples but directionally positive. Note: our GSM8K accuracy (22%) is below Bonsai's reported benchmark (88%) due to differences in evaluation harness (prompt format, answer extraction, generation length). The relative comparison between base and patched is the meaningful signal, not the absolute number.

### Experiment 9: GGUF/CUDA Backend (Beefier Search)

**Search:** 800 iterations on Modal L40S, 8 layers × 2 MLP projections, 60 training / 30 validation probes, seed 42. **424 seconds total** — ~9.5x faster than the original M3 run (4018s).

**Patch statistics:**
- 73 flips accepted (vs 93 on MLX Exp 6 with a smaller search space)
- 54 screened out (much higher screen-out rate than the original MLX run, implying the expanded search was more rigorous)
- Patch size: 876 bytes
- Layers 5, 6, 10 all contributed accepted flips (they were not in the original search set)

**Training set (60 probes):** 43/60 → 42/60 (2 fixed, 3 broke)

**Validation set (30 held-out probes):** **13/30 → 16/30 (3 fixed, 0 broke)**

| Category | Fixed | Broke |
|---|---|---|
| poly_deriv | 1 | 0 |
| second_deriv | 1 | 0 |
| integral | 1 | 0 |
| prime | 0 | 0 |
| trig | 0 | 0 |
| exp_deriv | 0 | 0 |

**Finding:** The GGUF format (scale-only Q1_0_g128, 1.125 bpw) is strictly less expressive than MLX (scale+bias, 1.25 bpw), and baseline logit gaps differ between the two (e.g., `Paris vs London` is +6.50 on MLX, +3.54 on GGUF for the same tokens). The GGUF search with expanded layers **recovers 3/4 of the MLX validation result** (3 fixed vs 4) while running at a fraction of the time. Layers 5, 6, 10 contributing to accepted flips confirms the layer-impact map's prediction.

### Experiment 10: Attention Projection Search

**Search:** Same as Experiment 9 but with `search_projs = [gate, up, q, k, v, o]`. 800 iterations, 433 seconds.

**Patch statistics:**
- 85 flips total — 69 MLP (gate: 37, up: 32), **16 attention** (q: 7, o: 4, k: 3, v: 2)
- All four attention projection types received accepted flips, confirming the mechanism works on attention weights

**Training set:** 43/60 → 44/60 (2 fixed, 1 broke)

**Validation set:** **13/30 → 13/30 (0 fixed, 0 broke)**

**Finding (negative result):** Attention-projection XOR flips are mechanically functional but **hurt generalization**. Adding the attention search space to the beefier run (Experiment 9) eliminated all 3 held-out validation fixes. Per-category, every validation category regressed from 3 total fixes to 0.

**Interpretation:** Attention weights encode context-specific routing patterns. Flipping rows in attention projections finds modifications that improve training-set fitness (+0.23 mean fitness gain) but those modifications depend on exact prompt structure and don't transfer to novel phrasings. MLP rows, by contrast, encode more abstract representations whose modifications generalize better. **Conclusion: For behavioral patching of 1-bit LLMs, MLP is the right target; attention is too context-bound.** This is the first empirical characterization of attention XOR flips on 1-bit models.

### Experiment 11: Per-Group Granularity Search

**Search:** 1500 iterations on Modal L40S, 8 layers × 2 MLP projections, `granularity="group"`. 755 seconds. Candidate space: **6.3 million groups** (vs ~131K rows in Experiment 9).

**Patch statistics:**
- **Only 6 flips accepted** (out of 1500 iterations)
- 331 screened out (most iterations rejected before full eval)
- 5 of 6 accepts concentrated in layer 34
- Max fitness reached: **+0.0067** (vs +0.32 for row-level beefy run)

**Training set:** 43/60 → 43/60 (0 fixed, 0 broke)

**Validation set:** 13/30 → 13/30 (0 fixed, 0 broke)

**Finding (negative result):** Per-group flips (128 bits each) are too fine-grained for the current mean-fitness search. Each group-flip produces probe gap changes on the order of 0.001–0.01 — about 10x smaller than row-level flips — and the control degradation penalty (λ=2.0) dominates this tiny signal. The search correctly rejected ~99.6% of candidates and the accepted flips produced no measurable behavioral change.

**Interpretation:** Finer granularity is not free — it needs either (a) a much more sensitive fitness function (e.g., log-probability differences instead of logit gaps), (b) a lower control penalty to let small improvements through, (c) cumulative flipping (accept pairs/triples of groups together to reach a meaningful effect size), or (d) a completely different search algorithm (evolutionary with population, or gradient-free optimization with variance reduction). Naive greedy hill climbing at group granularity is not viable.

### Experiment 12: Corrected Metric Replication

**Search:** 60 training probes, `seq_logprob` fitness, 300 iterations, seed 42, layers `[1,2,3,4,34]` — identical to Experiment 6 in every respect except the measurement. 78 accepted flips, 936 bytes. XOR revert verified bit-exact (baseline drift after revert: `0.00e+00`).

**Validation (30 held-out probes, greedy generation — the headline):**

| Metric | Count |
|---|---|
| Fixed (wrong → right) | **5** |
| Broke (right → wrong) | **0** |
| Accuracy | 14/30 → **19/30** |

**Training set (60 probes, greedy generation):** 30/60 → 33/60, 4 fixed, 1 broke (`int_train_7`).

**Per-category validation:**

| Category | Before | After |
|---|---|---|
| poly_deriv | 3/5 | 4/5 |
| second_deriv | 0/5 | 1/5 |
| integral | 3/5 | 3/5 |
| prime | 2/5 | **4/5** |
| trig | 2/5 | 2/5 |
| exp_deriv | 4/5 | **5/5** |

**All five held-out fixes, as actual model output:**

```
pd_val_0  d/dx [x^7 + x] =        ' 0'                    -> ' 7x^6 + 1'
sd_val_3  2nd derivative of 3x^3  ' 18x^2'                -> ' 18x'
pr_val_1  Is 113 prime?           ' No, 113 is not prime' -> ' Yes, 113 is a prime number'
pr_val_2  Is 53 prime?            ' No, 53 is not a...'   -> ' Yes, 53 is a prime number'
ed_val_2  d/dx [e^(-2x)] =        ' -2e^(-2x) + 0 = -2'   -> ' -2e^(-2x)'
```

**Finding:** Correcting the metric did not merely deflate the original result — it produced a better one. The corrected search finds **5 held-out fixes with zero breakage** using a **smaller** patch (78 flips / 936 bytes vs 93 flips / 1,116 bytes), and every fix is verified at the generation level rather than by a logit proxy. This is a strictly harder bar than Experiment 6's: the model must emit the complete correct expression, not merely rank one token above another. Under that bar the base model answers 14 of 30 held-out probes correctly, and the patch takes it to 19.

The comparison to Experiment 6 is therefore not 4 → 2 but rather *4 claimed under a broken metric* → *5 demonstrated under a sound one*. Three of Experiment 6's four fixes were measurement artifacts; the patch found here fixes five problems that were genuinely wrong and stay fixed in free generation.

Primality improves most (2/5 → 4/5), consistent with it being the one original category whose contrast alternated direction and therefore could not be won by a token bias. The single training-set regression (`int_train_7`: "The antiderivative of x^3 is" goes from `x^4/4` to `3/4 x^4`) is a real one and is reported rather than screened out.

**Extended layer set (negative result).** Experiment 6 speculated that trig and exponential derivatives saw zero fixes because layers 5, 6 and 10 — identified as high-impact for calculus — were not in the search set. Rerunning with `[1,2,3,4,5,6,10,34]`, changing nothing else:

| | Baseline `[1,2,3,4,34]` | Extended `+[5,6,10]` |
|---|---|---|
| Train (generation) | 30/60 → 33/60 | 30/60 → **35/60** |
| **Held-out (generation)** | 14/30 → **19/30** (5 fixed) | 14/30 → 17/30 (**3** fixed) |
| Broke (held-out) | 0 | 0 |
| Patch | 78 flips, 936 B | 95 flips, 1,140 B |
| Final search fitness | lower | **+0.4763** |

The extended search fits the training set **better** (35/60 vs 33/60) and generalizes **worse** (17/30 vs 19/30). Since training accuracy improved, this is overfitting rather than under-sampling of a larger candidate pool. The hypothesis is falsified on its own terms: adding those layers did not move trig (2/5 → 2/5) or second derivatives (0/5 → 0/5), the two categories it was supposed to explain.

This replicates the Experiment 10 pattern from an independent direction. There, adding attention projections to the search space eliminated all held-out fixes; here, adding MLP layers halves them. Two unrelated expansions of the search space at fixed iteration budget both traded generalization for training fit, which suggests the constraint is a property of greedy hill climbing on this objective rather than of any particular layer or projection type. **The narrower search set is the better one**, and Experiment 6's original layer choice was sound even though its stated reason for the trig gap was not.

Caveat: both runs used 300 iterations while the candidate pool grew from ~131K to ~210K rows, so the extended run sampled a smaller fraction of its space. A budget-matched-by-coverage rerun would separate "more layers hurt" from "more layers need more iterations" — but it would not rescue the specific claim about trig, which is unmoved.

**Evaluation notes.** Generation is scored by prefix match after normalizing whitespace, grouping, and explicit multiplication, with `√` spelled out; a trailing constant of integration is accepted, and probes may declare alternate renderings so `0.7071` counts for `sqrt(2)/2`. Two matcher defects were found and fixed while analysing this run — `+ C` followed by further text was rejected, and `1/√2` was not accepted as `sqrt(2)/2`. Both were corrected and **both the before and after sets were re-scored under the identical rule** via `tools/rescore_generation.py`, which re-scores stored outputs without re-running the model. Answers are capped at 14 generated tokens, so a small number of long expressions (e.g. `pd_val_2`) are truncated and scored wrong in both conditions; this understates absolute accuracy but not the delta.

### Experiment 14: Head-to-Head Patch Comparison

Experiment 6 reported 4 held-out fixes and Experiment 12 reported 5, but under different probe sets and different metrics — so those numbers cannot be compared directly. This scores every patch on the **same** 30 held-out probes with the **same** corrected metric. Evaluation only, no search.

| Patch | Flips | Bytes | Held-out (generation) | Fixed | Broke |
|---|---|---|---|---|---|
| *(no patch)* | — | — | 14/30 | — | — |
| Experiment 6 (`token_gap` search) | 93 | 1,116 | **13/30** | 2 | **3** |
| **Experiment 12 (corrected, base layers)** | 78 | 936 | **19/30** | **5** | **0** |
| Experiment 12 (corrected, extended layers) | 95 | 1,140 | 17/30 | 3 | 0 |

**Finding:** On a level playing field the Experiment 6 patch is **net negative** — it fixes 2 held-out probes and breaks 3 (`int_val_4`, `trig_val_0`, `ed_val_4`), leaving the model worse than unpatched. Experiment 6's headline claim of **zero breakage was itself a measurement artifact**: the single-token metric was too coarse to detect the regressions, which only become visible when the model is required to produce the complete expression.

Per-category, the damage is specific: trig drops 2 → 1 and exp_deriv 4 → 3 under the Experiment 6 patch, while the Experiment 12 patch holds or improves every category (second_deriv 0 → 1, prime 2 → 4, exp_deriv 4 → 5).

In fairness to the original, the Experiment 6 patch was optimized against a different objective, so underperforming on this one is expected. But the claim attached to it — "fixes 4 of 17 problems the base model gets wrong" — was a claim about capability rather than about logit gaps, and on those terms it does not survive sound measurement.

This is the comparison that answers whether the correction produced a better patch, and by how much: from **−1 net** to **+5 net** on the same yardstick, using 15 fewer flips and 180 fewer bytes.

### Experiment 13: Corrected Variation Testing

**Setup:** Evaluation only, no search. The same 6-probe Experiment 4 patch (`calculus_v1.json`) applied to the same 90 novel variations as Experiment 5, scored with `seq_logprob` and greedy generation.

| Metric | Experiment 5 (published) | Experiment 13 (corrected) |
|---|---|---|
| Fixed (wrong → right) | 2 | 2 |
| Broke (right → wrong) | 5 | **15** |
| Accuracy | 62/90 → 59/90 | 49/90 → **36/90** |

Under `seq_logprob` the picture is the same: 65/90 → 57/90, **0 fixed, 8 broke**.

**Per-category (greedy generation):**

| Category | Before | After |
|---|---|---|
| poly_deriv | 11/15 | 6/15 |
| second_deriv | 1/15 | 0/15 |
| integral | 8/15 | 6/15 |
| prime | 7/15 | 7/15 |
| trig | 10/15 | 6/15 |
| exp_deriv | 12/15 | 11/15 |

**Finding:** Experiment 5's conclusion holds and was **understated**. A patch trained on 6 probes does not merely fail to generalize — it actively damages capability across five of six categories, breaking three times as many probes as originally measured. The original writeup rationalized the 5 breaks as "all borderline cases with baseline gaps < 1.2"; that explanation does not survive correction, because the damage is broad rather than marginal.

This matters for reading the correction as a whole. The same measurement fix that **reduced** Experiment 6's claimed benefit **increased** Experiment 5's measured harm. The defects were not biased toward flattering results, and correcting them was not an exercise in preserving them.

Taken together, Experiments 12 and 13 sharpen the paper's central empirical claim considerably. Training on 6 probes: 2 fixed, 15 broken. Training on 60 diverse probes: 5 fixed, 0 broken on held-out prompts. The contrast between memorization and generalization is far starker under sound measurement than it was under the original metric.

### Experiment 15: Calculus Layer Impact Map

Experiment 6 explained its zero trig and exponential-derivative fixes by asserting that a calculus-specific layer impact map "identified layers 5, 6, and 10 as high-impact for calculus but not included in the current search set." No artifact for that map was ever committed. This measures it.

**Top 12 of 36 layers, by mean |Δ| across 90 calculus probes:**

| Rank | Layer | Mean \|Δ\| | Status |
|---|---|---|---|
| 1 | 34 | 25.96 | searched in Exp 3–6 |
| 2 | 2 | 11.77 | searched in Exp 3–6 |
| 3 | 1 | 9.30 | searched in Exp 3–6 |
| 4 | 3 | 8.28 | searched in Exp 3–6 |
| 5 | 4 | 8.23 | searched in Exp 3–6 |
| 6 | 0 | 7.31 | — |
| 7 | 9 | 5.74 | — |
| 8 | **5** | 5.15 | **claimed high-impact** |
| 9 | **10** | 4.74 | **claimed high-impact** |
| 10 | 8 | 4.58 | — |
| 11 | 11 | 4.55 | — |
| 12 | **6** | 4.46 | **claimed high-impact** |

**Finding:** The premise was false. The five layers Experiments 3–6 actually searched — `[1, 2, 3, 4, 34]` — are **ranks 1 through 5** for calculus. The three layers claimed to be high-impact rank **8th, 9th and 12th**, below even layers 0 and 9, which nobody proposed adding. Layer 34 alone carries more than twice the impact of the next layer.

This explains the [Experiment 12 extended-layer result](#experiment-12-corrected-metric-replication) mechanically rather than statistically. Adding layers 5, 6 and 10 did not enlarge the search into more promising territory; it diluted a candidate pool that was already optimally chosen, spending a fixed iteration budget on rows with roughly half the behavioral leverage. Better training fit with worse generalization is what that predicts.

So Experiment 6's account of the trig gap was wrong twice over: the hypothesis was falsified by Experiment 12, and the premise it rested on is contradicted here. The original layer selection was not a limitation to be corrected — it was the best available choice, and the trig gap needs a different explanation.

**If the search set is ever widened,** this map says the candidates are layers 0 and 9, not 5, 6 and 10.

## Errata and Corrections

**Status.** Experiments 5 and 6 report inflated fix counts. The underlying finding — that ultra-sparse XOR patches produce targeted behavioral change with no measured collateral damage — replicates under corrected measurement. Affected numbers are corrected here and superseded by [Experiment 12](#experiment-12-corrected-metric-replication). Version 1 results are annotated rather than rewritten, and the artifact as published is preserved at git tag `v1.0`, so anything already cited stays retrievable.

### Origin

[@sbenjam1n](https://github.com/sbenjam1n) independently reproduced Experiment 6 in a standalone PyTorch Q1_0 engine and reported two problems in [issue #3](https://github.com/nikshepsvn/bankai/issues/3): the hardcoded wrong token was not the model's actual top competitor on 74 of 90 probes, and the integral category was a single token contrast repeated fifteen times rather than fifteen independent measurements. Both replicate here — 76/90 and confirmed respectively. Auditing them surfaced a third, larger defect not previously reported. Their reproduction also confirmed the baseline of 17/30 validation probes wrong, matching across MLX and GGUF/PyTorch runtimes.

### Defect 1 — dead probes

`encode_token()` keeps only the last subtoken of an answer string. Bonsai's tokenizer splits digits into single characters:

```
" 20" -> [220, 17, 15]  [' ', '2', '0']   last subtoken = 15
" 0"  -> [220, 15]      [' ', '0']        last subtoken = 15
```

Both reduce to the same id, so `correct_id == wrong_id` and the logit gap is **identically 0.0 regardless of any weight flip**. Such a probe cannot be fixed, cannot be broken, and scores as "wrong" because the gap is not positive.

| Experiment | Probes | Dead | Effect |
|---|---|---|---|
| 5 (variation testing) | 90 | 7 | inflates the wrong-answer denominator |
| 6 (generalization) | 90 | 8 | 4 of them inside the 30-probe validation set |

For Experiment 6 the reported baseline of "17 of 30 wrong" contains 4 probes structurally incapable of being either right or wrong. The fixable pool was 13, not 17, so the reported 4/17 (23.5%) used an inflated denominator.

Reproduce with `python experiments/00_probe_audit.py --probe-set exp6`.

### Defect 2 — the measured token is not the emitted token

68 of 90 Experiment 6 answers are multi-token, so the scored id is not the token the model emits next. On 52 of 90 probes the model's actual top-1 next token is a bare space — the digit was scored one position early. Under `token_gap` the model appears to answer 13/30 validation probes correctly; scoring the full answer string puts it at 22/30, and greedy decoding at 17/30.

### Defect 3 — distractor quality and category degeneracy

The hardcoded distractor was not the model's top competitor on 76/90 probes, median rank 23. Contrast diversity per category was badly uneven:

| Category | Distinct (correct, wrong) pairs across 15 probes |
|---|---|
| poly_deriv | 10 |
| exp_deriv | 10 |
| second_deriv | 8 |
| trig | 3 |
| prime | 2 |
| **integral** | **1** |

The integral category is one measurement reported as fifteen, as issue #3 states. Primality is a two-token contrast but alternates direction across probes — some want ` Yes` over ` No`, others the reverse — so it cannot be won by a token bias and is not degenerate in the same way. Trig additionally mixed answer conventions: `sin(pi/6)` was scored against the fraction form while `sin(pi/4)` was scored against the decimal form, so a flip helping one necessarily hurt the other.

### What is corrected

Re-scoring the **same 93-flip Experiment 6 patch** under full-answer logprob, changing nothing else:

| Metric | base OK | patched | fixed | broke | dead probes |
|---|---|---|---|---|---|
| `token_gap` (as published) | 13 | 17 | **4** | 0 | 4 |
| first-token only | 6 | 7 | 1 | 0 | 21 |
| `seq_logprob` | 22 | 24 | **2** | 0 | 0 |
| greedy generation | 17 | 19 | **2** | 0 | — |

Three of the four published fixes were **already correct at baseline** and were counted as failures only because of Defect 2:

| Probe | Category | `token_gap` | `seq_logprob` | verdict |
|---|---|---|---|---|
| pd_val_1 | poly_deriv | −0.086 → +0.566 | **+1.500** → +3.062 | already correct |
| sd_val_3 | second_deriv | −0.088 → +0.242 | **+0.391** → +0.836 | already correct |
| int_val_4 | integral | −0.107 → +0.043 | **+0.344** → +0.578 | already correct |
| pr_val_1 | prime | −0.250 → +0.289 | −0.250 → +0.289 | **real fix** |

The integral probe is one of the three that evaporate — exactly the degenerate category issue #3 identified. Note also that `int_val_4`'s post-patch margin of +0.043 sits only 3.5× above the ~0.012 logit noise floor of batched 1-bit kernels.

The corrected metric also finds a fix the original **missed** (`pd_val_0`), and greedy decoding confirms both survivors as real changes in output:

```
d/dx [x^7 + x] =    ' 0\n\nWait, let'  ->  ' 7x^6 +'
Is 113 prime?       ' No, 113'        ->  ' Yes, 113'
```

The layer-set explanation Experiment 6 offered for its trig gap has since been tested end to end: the hypothesis is falsified in [Experiment 12](#experiment-12-corrected-metric-replication) and the premise it rested on is contradicted in [Experiment 15](#experiment-15-calculus-layer-map), which supplies the layer map that was cited but never committed.

Experiment 5's sign-flip counts were affected by 7 dead probes and understated the harm: rerun as [Experiment 13](#experiment-13-corrected-variation-testing), the 6-probe patch breaks 15 probes rather than 5. The defects were not biased toward flattering results — the same fix that reduced Experiment 6's claimed benefit increased Experiment 5's measured damage.

### What is unchanged — and what improved

A kilobyte-scale patch of row flips (0.007% of weights) changes targeted behavior with no measured collateral damage on held-out probes. That claim is unaffected, and the evidence behind it is now stronger, because generation-level verification depends on no distractor token at all.

The correction also improved the result. Re-scoring the old patch gives 2 fixes; re-running the search against a sound objective ([Experiment 12](#experiment-12-corrected-metric-replication)) gives **5 held-out fixes and 0 breaks from a 936-byte patch** — fewer flips than Experiment 6 used, under a strictly harder bar. The original search spent 300 iterations partly optimizing a mis-tokenized target scored against a rank-23 distractor; pointing it at the right objective recovered more than the defect had cost.

### Methodological change

Logit gaps are retained as a **search signal** — cheap enough to hill-climb on — but are no longer reported as a **result**. This implements the fix Experiment 11's own interpretation identified ("a much more sensitive fitness function, e.g. log-probability differences instead of logit gaps"). The original metric remains available as `metric="token_gap"` so Experiments 1–11 stay reproducible exactly as published.

## Limitations

**The corrected metric is MLX-only.** `seq_logprob` requires `Backend.seq_logprobs()`, which is implemented for MLX but not for GGUF — the `bankai_eval` subprocess exposes a `PROBE` command returning a single logit gap and would need a continuation-scoring command alongside it. Until that lands, Experiments 9–11 (the GGUF/CUDA results) rest on `token_gap` and inherit its defects; their numbers should be read with the same caution as Experiment 6's. The GGUF backend raises a clear `NotImplementedError` rather than silently falling back, so no experiment can quietly run on the wrong metric.

**Evaluation harness limitations.** Our GSM8K accuracy (22%) is well below reported benchmarks, indicating our evaluation setup doesn't match standard methodology. Logit gap probes are fast but don't always predict generation-level outcomes (visible in the 7×8 example). Proper benchmark evaluation with standard harnesses is a next step.

**Greedy search finds local optima.** Population-based evolutionary search with crossover (XOR of XOR patches is a valid patch) could find better solutions in the same search budget.

**Row-level granularity is right for mean-fitness search.** Per-group search (Experiment 11) confirmed that naive finer granularity doesn't work: individual 128-bit flips produce signal too small to overcome the control penalty. Finer granularity needs a more sensitive fitness function (log-probability differences) or cumulative flipping (groups of groups).

**Attention projections don't help generalization** (Experiment 10). XOR flips on attention Q/K/V/O produce mechanically-valid modifications that improve training fitness but regress validation. Attention weights appear to encode context-specific routing that overfits to training prompts.

**GGUF format is less expressive than MLX.** Q1_0_g128 stores scale only (1.125 bpw); MLX g128 stores scale + bias (1.25 bpw). The same probe produces different logit gaps on the two formats, and patches found on one don't necessarily transfer. Experiments 3–8 use MLX; 9–11 use GGUF.

**Patch stacking shows interference.** Experiment 7 confirms that stacking is mechanically sound but behaviorally lossy — individual patches partially cancel each other's improvements. Joint optimization would likely outperform naive stacking.

**Single model, single architecture.** All experiments use Bonsai 8B, currently the only production-quality true 1-bit LLM. The approach does not extend to ternary/1.58-bit models. Generalization depends on the emergence of additional true 1-bit models.

**Scale factors are not patched.** Current patches modify only binary weights, not the FP16 scale/bias values. Including scale deltas could enable finer-grained control at the cost of larger patches.

**Small evaluation set.** Our probes cover limited domains. A comprehensive evaluation across diverse tasks is needed to characterize the full potential and failure modes of XOR patching.

## Responsible Use

Bankai modifies model behavior with kilobyte-scale patches that are invisible at inference time. The same mechanism that enables constructive behavioral steering could, in principle, be used to insert subtle malicious behavioral changes — this is the dual-use nature of inverting adversarial bit-flip research.

However, XOR patches are **transparent by design**:
- Every patch is a readable JSON file listing exactly which rows were flipped
- Patch verification is trivial: compute the Hamming distance between patched and unpatched weights and confirm it matches the patch manifest
- The patch format is structured for auditing, diffing, and revocation
- The XOR operation is deterministic — there are no hidden states or opaque transformations

We recommend that any deployment of XOR-patched models include patch provenance metadata (who created it, what fitness function was used, what probes were optimized) and that patches be verified against their manifest before use. If patch libraries become a real deployment pattern (as described in [Why This Matters at Deployment Scale](#why-this-matters-at-deployment-scale)), provenance and verification become critical infrastructure — not just good practice, but a requirement for trust in the patch ecosystem.

## Future Work

- **Sensitive fitness for finer granularity** — per-group (Experiment 11) and per-bit search need fitness signals beyond logit gaps. Log-probability differences, or full-generation KL against a teacher, would preserve the signal that a single group flip produces.
- **Cumulative group voting** — accept combinations of 2–4 group flips together so the effect size crosses the noise threshold. Equivalent to a two-phase greedy search with a larger basic "step".
- **Larger training sets (120–240 probes)** — Experiment 9 plateaued on 60 probes. More probe variations per category should push generalization further, and the 24x-faster GGUF pipeline makes this tractable.
- **Proper benchmark evaluation** — MMLU subcategories, GSM8K, HumanEval with a standard harness. The bankai_eval subprocess can be extended with a `GENERATE` command to produce full answers.
- **Evolutionary search with crossover** — population-based, XOR-of-XOR-patches as crossover, Hamming distance as diversity pressure. The fast GGUF backend makes ~50-individual populations tractable.
- **Cross-model extraction** — If PrismML releases a Bonsai variant, the XOR between the two models is itself a patch. Sparse by construction.
- **Joint-domain search** — Experiment 7 showed naive stacking of math + calculus patches loses improvements. A single search optimizing both domains simultaneously (not as post-hoc union) should outperform stacking.
- **Attention revisited with per-group** — Experiment 10 showed row-level attention flips overfit. Per-group might find surgical attention modifications that don't, but needs Experiment 11's fitness problems fixed first.
- **Theoretical analysis** — connect patch sparsity to information-theoretic bounds on binary weight redundancy.

## Reproducing

### Requirements

**MLX path (experiments 1–8):** Apple Silicon Mac (M-series), Python 3.11+, PrismML's MLX fork.
**GGUF/CUDA path (experiments 9–11):** NVIDIA GPU (T4, L40S, H100, etc.) or a Modal account (~$1–2 per search on L40S). Built automatically inside a Modal image.

### Setup (MLX path)

```bash
git clone https://github.com/nikshepsvn/bankai.git
cd bankai

python -m venv .venv && source .venv/bin/activate
pip install mlx-lm
pip install "mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism"
pip install -e ".[dev]"

# Download model (~1.3 GB)
huggingface-cli download prism-ml/Bonsai-8B-mlx-1bit --local-dir models/bonsai-8b-mlx
```

### Setup (GGUF/Modal path)

```bash
pip install modal
python -m modal setup  # one-time browser auth
```

The Modal experiment scripts (09–11) define their own image — cloning PrismML's llama.cpp fork, building the custom `bankai_eval` tool, and running on a Modal GPU. The first build is ~8 minutes (cached thereafter).

### Run experiments

```bash
# Audit a probe set before trusting any logit-gap result (~10 s, no GPU)
python experiments/00_probe_audit.py --probe-set exp6
python experiments/00_probe_audit.py --probe-set exp6 --with-model  # + distractor ranks

# Corrected-metric experiments (MLX)
python experiments/12_corrected_metric_search.py                  # Exp 6 redone (~2 h)
python experiments/12_corrected_metric_search.py --layers extended # + layers 5,6,10 (~2.5 h)
python experiments/13_corrected_variation_testing.py              # Exp 5 redone (~8 min)
python experiments/14_patch_headtohead.py                         # All patches, one metric (~6 min)
python experiments/15_calculus_layer_map.py                       # Calculus layer impact (~15 min)

# Re-score stored generations after changing accepted-answer rules (no GPU)
python tools/rescore_generation.py results/experiment12_baseline_results.json

# MLX path (run on Apple Silicon)
python experiments/01_random_flips.py           # Robustness to random flips (~8 min)
python experiments/02_logit_steering.py         # Layer impact map (~2 min)
python experiments/03_patch_search.py           # Arithmetic patch search (~8 min)
python experiments/04_calculus_patch.py         # Calculus patch with screening (~13 min)
python experiments/05_variation_testing.py     # Does the patch generalize? (~3 min)
python experiments/06_generalization_search.py  # 60-probe generalization search (~67 min)
python experiments/07_patch_stacking.py         # Math + calculus stacking (~3 min)
python experiments/08_gsm8k_safety.py           # GSM8K safety check (~20 min)

# GGUF/Modal path (runs on L40S)
modal run experiments/09_gguf_beefy_search.py   # 8 layers × 2 projs × 800 iters (~7 min)
modal run experiments/10_attention_search.py   # + attention projections (~7 min)
modal run experiments/11_per_group_search.py   # Per-group granularity (~13 min)
```

### Use the toolkit

```bash
# Search for a patch using built-in probes
bankai search --model models/bonsai-8b-mlx --target math --output patches/my_patch.json

# Search using a custom probe file
bankai search --model models/bonsai-8b-mlx --target my_probes.json --output patches/custom.json

# Reproduce a pre-correction experiment with the original single-token metric.
# Defaults to --metric seq_logprob; token_gap has the defects in the errata and
# is the only option on the GGUF backend.
bankai search --model models/bonsai-8b-mlx --target math --metric token_gap --output patches/legacy.json

# Evaluate a patch
bankai eval --model models/bonsai-8b-mlx --patch patches/patch_math_v1.json --probes math,knowledge

# Compare generation with/without patch
bankai apply --model models/bonsai-8b-mlx --patch patches/patch_math_v1.json --prompt "2 + 2 ="
```

### Define custom probes

Create a JSON file with your target behavior:

```json
[
  {"prompt": "SELECT * FROM", "correct": " users", "wrong": " tables", "name": "sql_1", "category": "sql"},
  {"prompt": "git checkout -b", "correct": " feature", "wrong": " master", "name": "git_1", "category": "git"}
]
```

Then: `bankai search --model models/bonsai-8b-mlx --target my_probes.json`

## Citation

```bibtex
@misc{saravanan2026bankai,
  title   = {Bankai: Ultra-Sparse Adaptation of 1-Bit LLMs via XOR Patches},
  author  = {Saravanan, Nikshep},
  year    = {2026},
  url     = {https://github.com/nikshepsvn/bankai}
}
```

## References

- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Rakin, A. S., et al. (2019). Bit-Flip Attack: Crushing Neural Network with Progressive Bit Search. [ICCV 2019](https://arxiv.org/abs/1903.12269)
- Dong, P., et al. (2024). STBLLM: Breaking the 1-Bit Barrier with Structured Binary LLMs. [ICLR 2025](https://arxiv.org/abs/2408.01803)
- Wang, H., et al. (2023). BitNet: Scaling 1-bit Transformers for Large Language Models. [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)
- Ma, S., et al. (2025). BitNet b1.58 2B4T Technical Report. [arXiv:2504.12285](https://arxiv.org/abs/2504.12285)
- PrismML. (2026). Bonsai 8B. [prismml.com/news/bonsai-8b](https://prismml.com/news/bonsai-8b)
- Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. [NeurIPS 2022](https://arxiv.org/abs/2202.05262)
- Meng, K., et al. (2023). Mass-Editing Memory in a Transformer. [ICLR 2023](https://arxiv.org/abs/2210.07229)
- Bai, Y., et al. (2021). Targeted Attack against Deep Neural Networks via Flipping Limited Weight Bits. [ICLR 2021](https://arxiv.org/abs/2102.10496)
- Xu, Y., et al. (2024). RECAST: Reparameterized, Compact weight Adaptation for Sequential Tasks. [arXiv:2411.16870](https://arxiv.org/abs/2411.16870)
- Bulat, A. & Tzimiropoulos, G. (2020). XOR-Net: An Efficient Computation Pipeline for Binary Neural Network Inference on Edge Devices.
- Rastegari, M., et al. (2016). XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks. [ECCV 2016](https://arxiv.org/abs/1603.05279)

## License

Apache 2.0
