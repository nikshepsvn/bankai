"""
GGUF backend for Bankai (in-GPU weight manipulation).

Drives a long-running `bankai_eval` subprocess (built against PrismML's
llama.cpp fork with Q1_0_g128 kernels) via stdin/stdout. All weight
modifications happen in GPU memory via ggml_backend_tensor_get/set —
no GGUF file rewriting, no model reloading.

Protocol:
  TOKENIZE <text>          → "<n> <t0> <t1> ..."
  PROBE <c> <w> <n> <t...> → logit gap "c-w"
  FLIP_ROW <name> <row>    → XOR sign bits in row (self-inverse)
  NUM_ROWS <name>          → row count
  NUM_COLS <name>          → col count
  SCALES <name>            → avg |fp16 scale| per row
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from bankai.backends.base import Backend


PROJ_TO_GGUF = {
    # MLP projections
    "gate_proj": "ffn_gate",
    "up_proj":   "ffn_up",
    "down_proj": "ffn_down",
    # Attention projections (Bonsai uses GQA: k/v have fewer rows than q/o)
    "q_proj":    "attn_q",
    "k_proj":    "attn_k",
    "v_proj":    "attn_v",
    "o_proj":    "attn_output",
}


def _tensor_name(layer: int, proj: str) -> str:
    return f"blk.{layer}.{PROJ_TO_GGUF[proj]}.weight"


class GGUFBackend(Backend):
    """Q1_0_g128 GGUF backend with in-GPU weight manipulation via bankai_eval."""

    def __init__(self, bankai_eval_path: Optional[str] = None):
        self.bankai_eval_path = bankai_eval_path or "/root/llama.cpp/build/bin/bankai_eval"
        self.model_path: Optional[Path] = None
        self._proc: Optional[subprocess.Popen] = None
        self._n_layers = 36  # Bonsai 8B default (overridden in load if possible)
        self._row_count_cache: dict[tuple[int, str], int] = {}

    # ── Loading ──

    def load(self, model_path: str) -> None:
        """Load a Bonsai 8B GGUF file (local path or HuggingFace repo id)."""
        if not os.path.exists(model_path):
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id=model_path,
                filename="Bonsai-8B.gguf",
            )
        self.model_path = Path(model_path)
        self._start_process()

    def _start_process(self) -> None:
        self._stop_process()
        self._proc = subprocess.Popen(
            [self.bankai_eval_path, str(self.model_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Wait for READY with a timeout
        start = time.time()
        while time.time() - start < 180:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"bankai_eval died before READY (exit={self._proc.poll()})"
                )
            if line.strip() == "READY":
                return
        raise RuntimeError("bankai_eval did not say READY within 180s")

    def _stop_process(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def _send(self, cmd: str) -> str:
        if self._proc is None:
            raise RuntimeError("bankai_eval process not running")
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(
                f"bankai_eval died during '{cmd[:60]}' (exit={self._proc.poll()})"
            )
        return line.strip()

    # ── Model structure ──

    def num_layers(self) -> int:
        return self._n_layers

    def num_rows(self, layer: int, proj: str) -> int:
        key = (layer, proj)
        if key not in self._row_count_cache:
            resp = self._send(f"NUM_ROWS {_tensor_name(layer, proj)}")
            if resp == "ERROR":
                raise RuntimeError(f"NUM_ROWS failed for {_tensor_name(layer, proj)}")
            self._row_count_cache[key] = int(resp)
        return self._row_count_cache[key]

    def get_row_scales(self, layer: int, proj: str) -> np.ndarray:
        resp = self._send(f"SCALES {_tensor_name(layer, proj)}")
        if resp.startswith("ERROR"):
            raise RuntimeError(f"SCALES failed for {_tensor_name(layer, proj)}")
        parts = resp.split()
        n = int(parts[0])
        return np.array([float(x) for x in parts[1:1 + n]], dtype=np.float32)

    # ── Weight manipulation (in-GPU, no reload) ──

    def flip_row(self, layer: int, proj: str, row: int) -> None:
        resp = self._send(f"FLIP_ROW {_tensor_name(layer, proj)} {row}")
        if resp != "OK":
            raise RuntimeError(f"FLIP_ROW failed: {resp}")

    def flip_group(self, layer: int, proj: str, row: int, group: int) -> None:
        """Flip all 128 sign bits in a single (row, group) cell.

        32x more precise than flip_row, which XORs all 32 groups in a row.
        """
        resp = self._send(f"FLIP_GROUP {_tensor_name(layer, proj)} {row} {group}")
        if resp != "OK":
            raise RuntimeError(f"FLIP_GROUP failed: {resp}")

    def num_groups_per_row(self, layer: int, proj: str) -> int:
        """Number of 128-bit groups per row — always cols/128 for Q1_0_g128."""
        # Bonsai MLP gate/up projections have cols=4096 → 32 groups; down is
        # the same. Attention q/o have cols=4096; k/v have cols=4096 too.
        return 4096 // 128  # 32

    # ── Inference ──

    def encode(self, text: str) -> list[int]:
        if "\n" in text:
            raise ValueError("encode: newlines not supported in single-line protocol")
        resp = self._send(f"TOKENIZE {text}")
        parts = resp.split()
        n = int(parts[0])
        return [int(x) for x in parts[1:1 + n]]

    def encode_token(self, text: str) -> int:
        toks = self.encode(text)
        if not toks:
            raise ValueError(f"encode_token: empty result for {text!r}")
        return int(toks[-1])

    def logit_gap(self, prompt_tokens: list[int], correct_id: int, wrong_id: int) -> float:
        cmd = (
            f"PROBE {correct_id} {wrong_id} {len(prompt_tokens)} "
            + " ".join(str(t) for t in prompt_tokens)
        )
        resp = self._send(cmd)
        try:
            return float(resp)
        except ValueError:
            raise RuntimeError(f"bad PROBE response: {resp!r}")

    def batch_logit_gaps(
        self,
        probes: list[tuple[list[int], int, int]],
    ) -> list[float]:
        """Measure many logit gaps with a single pipelined stdin/stdout exchange.

        Each probe is (prompt_tokens, correct_id, wrong_id). We write all
        PROBE commands before reading any responses, which eliminates most
        of the per-command Python↔subprocess roundtrip latency.
        """
        if not probes:
            return []
        if self._proc is None:
            raise RuntimeError("bankai_eval process not running")

        # Write all commands up front
        lines = []
        for tokens, c_id, w_id in probes:
            lines.append(
                f"PROBE {c_id} {w_id} {len(tokens)} " + " ".join(str(t) for t in tokens)
            )
        self._proc.stdin.write("\n".join(lines) + "\n")
        self._proc.stdin.flush()

        # Read responses in order
        gaps: list[float] = []
        for _ in probes:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"bankai_eval died during batch probe "
                    f"(exit={self._proc.poll()}, got {len(gaps)} of {len(probes)})"
                )
            try:
                gaps.append(float(line.strip()))
            except ValueError:
                raise RuntimeError(f"bad PROBE response in batch: {line!r}")
        return gaps

    def measure_probes(self, probes):
        """Override base class to use batched measurement."""
        if not probes:
            return []
        encoded = [
            (self.encode(p[0]), self.encode_token(p[1]), self.encode_token(p[2]))
            for p in probes
        ]
        return self.batch_logit_gaps(encoded)

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        raise NotImplementedError("GGUF backend: generate() not implemented")

    # ── Cleanup ──

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
            except Exception:
                pass
            self._stop_process()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
