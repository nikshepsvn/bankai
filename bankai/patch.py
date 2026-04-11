"""
Patch format, application, and removal for 1-bit XOR patches.

A Bankai patch is a sparse XOR mask over binary weight tensors.
Applying a patch is a single bitwise XOR. Removing it is the same operation.
The patch format is backend-agnostic — the same JSON file works with MLX,
GGUF+CUDA, or any other backend that implements the Backend interface.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

# Backend is only imported for type hints to avoid circular imports at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bankai.backends.base import Backend


@dataclass
class PatchFlip:
    """A single bit flip.

    If `group` is None, the flip covers an entire row (all 32 groups = 4096
    sign bits). If `group` is set, the flip covers only one 128-bit group
    within the row.
    """
    layer: int
    proj: str
    row: int
    group: int | None = None


@dataclass
class Patch:
    """A collection of row-level flips that form a behavioral modification."""
    name: str
    description: str
    base_model: str
    flips: list[PatchFlip] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n_bits_flipped(self) -> int:
        """Each row flip = 4096 bits for gate/up, 12288 bits for down (MLP)."""
        # Approximation assuming gate/up rows (most common case)
        return len(self.flips) * 4096

    @property
    def size_bytes(self) -> int:
        """Approximate binary-format serialized size: 3 ints (12 bytes) per flip."""
        return len(self.flips) * 12

    def save(self, path: str | Path):
        data = {
            "version": 1,
            "format": "bankai_row_xor_v1",
            "name": self.name,
            "description": self.description,
            "base_model": self.base_model,
            "flips": [
                (
                    {"layer": f.layer, "proj": f.proj, "row": f.row, "group": f.group}
                    if f.group is not None
                    else {"layer": f.layer, "proj": f.proj, "row": f.row}
                )
                for f in self.flips
            ],
            "stats": {
                "n_flips": len(self.flips),
                "bits_flipped": self.n_bits_flipped,
                "size_bytes": self.size_bytes,
            },
            "metadata": self.metadata,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Patch":
        data = json.loads(Path(path).read_text())
        flips = [PatchFlip(**f) for f in data["flips"]]
        return cls(
            name=data.get("name", Path(path).stem),
            description=data.get("description", ""),
            base_model=data.get("base_model", "unknown"),
            flips=flips,
            metadata=data.get("metadata", {}),
        )


def apply_patch(backend: "Backend", patch: Patch):
    """Apply a patch to a loaded model. Call again to remove (XOR is self-inverse)."""
    for flip in patch.flips:
        if flip.group is None:
            backend.flip_row(flip.layer, flip.proj, flip.row)
        else:
            backend.flip_group(flip.layer, flip.proj, flip.row, flip.group)


def remove_patch(backend: "Backend", patch: Patch):
    """Remove a patch. Identical to apply (XOR is its own inverse)."""
    apply_patch(backend, patch)
