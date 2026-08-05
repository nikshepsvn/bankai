"""
Backend interface for Bankai.

A Backend implements the minimal set of operations required for XOR patch
search: loading a 1-bit model, measuring logit gaps on probes, flipping rows
of binary weights in-place, and retrieving scale magnitudes for scale-guided
sampling.

Search and patch logic is backend-agnostic — the same search loop runs on
MLX (Apple Silicon) and GGUF+CUDA (NVIDIA) by swapping the backend.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Backend(ABC):
    """Abstract interface for 1-bit LLM runtimes that support XOR patching."""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load the model into memory. Must be called before any other method."""

    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer layers in the loaded model."""

    @abstractmethod
    def num_rows(self, layer: int, proj: str) -> int:
        """Number of output rows (neurons) in a given MLP projection."""

    @abstractmethod
    def get_row_scales(self, layer: int, proj: str) -> np.ndarray:
        """Per-row scale magnitudes for scale-guided candidate sampling.

        Returns a 1D numpy array of length num_rows(layer, proj).
        Larger values indicate rows with higher expected behavioral impact.
        """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Tokenize text into a list of token IDs."""

    @abstractmethod
    def encode_token(self, text: str) -> int:
        """Encode a short string and return its final token ID.

        Used for probe correct/wrong tokens where we only care about the last
        subtoken of a multi-token string.
        """

    @abstractmethod
    def logit_gap(self, prompt_tokens: list[int], correct_id: int, wrong_id: int) -> float:
        """Measure logit(correct) - logit(wrong) at the position after prompt_tokens.

        This is the core probe measurement. Implementations should use a single
        forward pass per call.
        """

    @abstractmethod
    def flip_row(self, layer: int, proj: str, row: int) -> None:
        """XOR all bits in a single row of a binary weight tensor.

        Call again with the same (layer, proj, row) to revert — XOR is its own
        inverse. Implementations should do this in-place without copying weights.
        """

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from a prompt. Optional — only needed for benchmark evals."""
        raise NotImplementedError(f"{type(self).__name__} does not implement generate()")

    def seq_logprobs(self, items: list[tuple[list[int], list[int]]]) -> list[float]:
        """Teacher-forced total logprob of each continuation given its prompt.

        Each item is (prompt_ids, continuation_ids). Returns one summed logprob
        per item. Required by the "seq_logprob" probe metric, which scores whole
        answer strings instead of a single token id — see bankai.probes.

        Implementations should keep the batching layout deterministic so that
        repeated calls with identical inputs return identical values; the search
        compares measurements across weight flips, so reduction-order jitter
        between calls would masquerade as a fitness change.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement seq_logprobs()")

    # ── Convenience: batch probe measurement ──

    def measure_probes(self, probes: list[tuple[str, str, str]]) -> list[float]:
        """Measure logit gaps for a list of (prompt, correct_token, wrong_token) triples.

        Default implementation loops over logit_gap. Backends may override with a
        batched implementation for better throughput.
        """
        results = []
        for prompt, correct, wrong in probes:
            tokens = self.encode(prompt)
            c_id = self.encode_token(correct)
            w_id = self.encode_token(wrong)
            results.append(self.logit_gap(tokens, c_id, w_id))
        return results
