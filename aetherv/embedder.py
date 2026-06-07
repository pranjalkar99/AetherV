from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from fastembed import TextEmbedding


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-8)


class FastEmbedder:
    """FastEmbed-backed text embedder with L2 normalization."""

    def __init__(self, model: TextEmbedding | None = None) -> None:
        self.model = model or TextEmbedding()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(list(self.model.embed(texts)), dtype=np.float32)
        return normalize_vectors(vectors)
