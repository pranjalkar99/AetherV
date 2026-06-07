from __future__ import annotations

import numpy as np
import jax


@jax.jit
def score(vectors, query):
    return vectors @ query


class SegmentSearcher:
    """GPU-accelerated brute-force search within a single segment."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors_cpu = vectors
        self.vectors_gpu = jax.device_put(vectors)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = jax.device_put(query)
        scores = np.asarray(score(self.vectors_gpu, q))
        k = min(k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        return scores[idx], idx
