from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aetherv.config import Config
from aetherv.db import VectorDB
from aetherv.embedder import normalize_vectors


class DeterministicEmbedder:
    """Hash-based embedder for tests — no model download required."""

    dim = 8

    def embed(self, texts):
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = sum(ord(char) for char in text) % 997
            vectors[i] = np.arange(self.dim, dtype=np.float32) + seed
        return normalize_vectors(vectors)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "vectordb"


def test_insert_and_query(db_path: Path) -> None:
    config = Config(root=db_path, segment_size=2)
    db = VectorDB(config=config, embedder=DeterministicEmbedder())

    db.insert(
        ids=[1, 2, 3, 4],
        texts=[
            "JAX is a machine learning framework",
            "Cats are cute pets",
            "Vector databases store embeddings",
            "Polars is a dataframe library",
        ],
    )

    results = db.query("What is polars", k=3)
    assert len(results) == 3
    assert all(result.score > 0 for result in results)
    assert {result.id for result in results}.issubset({1, 2, 3, 4})


def test_metadata_lookup_is_o1(db_path: Path) -> None:
    config = Config(root=db_path, segment_size=10)
    db = VectorDB(config=config, embedder=DeterministicEmbedder())
    db.insert(ids=[42], texts=["hello world"])

    row = db.metadata.get(0, 0)
    assert row is not None
    assert row.id == 42
    assert row.text == "hello world"


def test_legacy_manifest_loading(db_path: Path) -> None:
    segments_dir = db_path / "segments"
    segments_dir.mkdir(parents=True)

    manifest = db_path / "manifest.json"
    manifest.write_text(
        """{
  "segments": [
    {"name": "segment_000000.arrow", "count": 2}
  ]
}"""
    )

    vectors = np.ones((2, DeterministicEmbedder.dim), dtype=np.float32)
    from aetherv.storage.arrow import ArrowSegment

    ArrowSegment.write(segments_dir / "segment_000000.arrow", vectors)

    db = VectorDB(config=Config(root=db_path), embedder=DeterministicEmbedder())
    assert 0 in db.searchers
    assert len(db.manifest.segments) == 1
    assert db.manifest.segments[0].vector_count == 2


def test_empty_insert_is_noop(db_path: Path) -> None:
    db = VectorDB(config=Config(root=db_path), embedder=DeterministicEmbedder())
    db.insert(ids=[], texts=[])
    assert db.query("anything") == []
