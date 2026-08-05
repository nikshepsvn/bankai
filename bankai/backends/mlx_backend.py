"""
MLX backend for Bankai — runs on Apple Silicon using PrismML's MLX fork
with 1-bit kernel support.
"""

import numpy as np
import mlx.core as mx

from bankai.backends.base import Backend


def _get_module(model, path: str):
    """Navigate dotted path like 'model.layers.0.mlp.gate_proj'."""
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


class MLXBackend(Backend):
    """MLX + PrismML 1-bit kernels. Apple Silicon only."""

    def __init__(self, batch_size: int = 8):
        self.model = None
        self.tokenizer = None
        self.batch_size = batch_size

    def load(self, model_path: str) -> None:
        from mlx_lm import load as mlx_load
        self.model, self.tokenizer = mlx_load(model_path)
        mx.eval(self.model.parameters())

    def num_layers(self) -> int:
        return len(self.model.model.layers)

    def _projection(self, layer: int, proj: str):
        path = f"model.layers.{layer}.mlp.{proj}"
        return _get_module(self.model, path)

    def num_rows(self, layer: int, proj: str) -> int:
        return int(self._projection(layer, proj).weight.shape[0])

    def get_row_scales(self, layer: int, proj: str) -> np.ndarray:
        mod = self._projection(layer, proj)
        return np.array(mx.mean(mx.abs(mod.scales), axis=1))

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def encode_token(self, text: str) -> int:
        return int(self.tokenizer.encode(text)[-1])

    def logit_gap(self, prompt_tokens: list[int], correct_id: int, wrong_id: int) -> float:
        tokens = mx.array(prompt_tokens)
        logits = self.model(tokens[None, :])
        last = logits[0, -1, :]
        mx.eval(last)
        return float(last[correct_id].item() - last[wrong_id].item())

    def seq_logprobs(self, items: list[tuple[list[int], list[int]]]) -> list[float]:
        """Batched teacher-forced continuation scoring.

        Sequences are bucketed by length and padded to a fixed batch shape so the
        matmul reduction order is identical on every call — without that, batched
        1-bit kernels jitter by ~0.01 logits between calls and the search would
        read that noise as fitness.
        """
        if not items:
            return []

        scores = [0.0] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
        pad_id = 0

        for start in range(0, len(order), self.batch_size):
            idxs = order[start:start + self.batch_size]
            seqs = [items[i][0] + items[i][1] for i in idxs]
            max_len = max(len(s) for s in seqs)

            padded, targets, masks = [], [], []
            for i, seq in zip(idxs, seqs):
                n_prompt, n_cont = len(items[i][0]), len(items[i][1])
                padded.append(seq + [pad_id] * (max_len - len(seq)))
                # Position p predicts token p+1, so score positions
                # [n_prompt-1, n_prompt+n_cont-2].
                tgt = [(seq + [pad_id] * (max_len - len(seq)))[p + 1] for p in range(max_len - 1)]
                msk = [1.0 if (n_prompt - 1) <= p <= (n_prompt + n_cont - 2) else 0.0
                       for p in range(max_len - 1)]
                targets.append(tgt)
                masks.append(msk)

            logits = self.model(mx.array(padded))[:, :-1, :]
            tgt_arr = mx.array(targets)[:, :, None]
            chosen = mx.take_along_axis(logits, tgt_arr, axis=-1)[:, :, 0]
            logprobs = chosen - mx.logsumexp(logits, axis=-1)
            totals = mx.sum(logprobs * mx.array(masks), axis=1)
            mx.eval(totals)

            for slot, i in enumerate(idxs):
                scores[i] = float(totals[slot].item())

        return scores

    def flip_row(self, layer: int, proj: str, row: int) -> None:
        path = f"model.layers.{layer}.mlp.{proj}"
        mod = _get_module(self.model, path)
        w = mod.weight
        mask = mx.zeros_like(w)
        ones = mx.full((w.shape[1],), 0xFFFFFFFF, dtype=mx.uint32)
        mask = mask.at[row].add(ones)
        new_w = w ^ mask
        self.model.load_weights([(f"{path}.weight", new_w)], strict=False)
        mx.eval(self.model.parameters())

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        from mlx_lm import generate as mlx_generate
        return mlx_generate(self.model, self.tokenizer, prompt=prompt,
                            max_tokens=max_tokens, verbose=False)
