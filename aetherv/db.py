from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from aetherv.config import Config
from aetherv.embedder import Embedder, FastEmbedder
from aetherv.search.gpu import SegmentSearcher
from aetherv.segments import SegmentManager
from aetherv.storage.arrow import ArrowSegment
from aetherv.storage.manifest import Manifest
from aetherv.storage.metadata import MetadataStore
from aetherv.types import SearchResult, SegmentRecord


class VectorDB:
    """Segmented vector database with GPU-accelerated search."""

    def __init__(
        self,
        root: str | Path = "vectordb",
        *,
        config: Config | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        if config is None:
            self.config = Config(root=Path(root))
        else:
            self.config = Config(
                root=Path(root),
                segment_size=config.segment_size,
                metadata_filename=config.metadata_filename,
                manifest_filename=config.manifest_filename,
                segments_dirname=config.segments_dirname,
            )

        self.config.root.mkdir(exist_ok=True, parents=True)
        self.embedder = embedder or FastEmbedder()
        self.metadata = MetadataStore(self.config.metadata_path)
        self.manifest = Manifest(self.config.manifest_path)
        self.segment_manager = SegmentManager(self.config.segments_dir)
        self.searchers: dict[int, SegmentSearcher] = {}
        self._load_segments()

    def _load_segments(self) -> None:
        for segment in self.manifest.segments:
            path = self.segment_manager.path_for(segment.name)
            vectors = ArrowSegment.read(path)
            self.searchers[segment.segment_id] = SegmentSearcher(vectors)

    def insert(self, ids: Sequence[int], texts: Sequence[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        if not ids:
            return

        vectors = self.embedder.embed(texts)
        for start in range(0, len(vectors), self.config.segment_size):
            stop = start + self.config.segment_size
            batch_ids = ids[start:stop]
            batch_texts = texts[start:stop]
            batch_vectors = vectors[start:stop]
            self._insert_batch(batch_ids, batch_texts, batch_vectors)

        self.metadata.save()

    def _insert_batch(
        self,
        batch_ids: Sequence[int],
        batch_texts: Sequence[str],
        batch_vectors: np.ndarray,
    ) -> None:
        segment_id = self.manifest.next_segment_id()
        record = SegmentRecord.create(segment_id, len(batch_vectors))
        path = self.segment_manager.path_for(record.name)

        ArrowSegment.write(path, batch_vectors)
        self.manifest.add_segment(record)
        self.metadata.append(batch_ids, batch_texts, segment_id)
        self.searchers[segment_id] = SegmentSearcher(batch_vectors)

    def query(self, text: str, k: int = 5) -> list[SearchResult]:
        if not self.searchers:
            return []

        query = self.embedder.embed([text])[0]
        candidates: list[tuple[float, int, int]] = []

        with ThreadPoolExecutor() as pool:
            futures = [
                (segment_id, pool.submit(searcher.search, query, k))
                for segment_id, searcher in self.searchers.items()
            ]

        for segment_id, future in futures:
            scores, rows = future.result()
            for score_value, row in zip(scores, rows):
                candidates.append((float(score_value), segment_id, int(row)))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return self._resolve_candidates(candidates[:k])

    def _resolve_candidates(
        self,
        candidates: Sequence[tuple[float, int, int]],
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for score_value, segment_id, row_id in candidates:
            row = self.metadata.get(segment_id, row_id)
            if row is None:
                continue
            results.append(
                SearchResult(score=score_value, id=row.id, text=row.text)
            )
        return results
