from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class MetadataRow:
    id: int
    text: str
    segment: int
    row: int


class MetadataStore:
    """Parquet-backed metadata with O(1) lookup by segment and row."""

    SCHEMA = {
        "id": pl.Int64,
        "text": pl.String,
        "segment": pl.Int64,
        "row": pl.Int64,
    }

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            self.df = pl.read_parquet(path)
        else:
            self.df = pl.DataFrame(schema=self.SCHEMA)
        self._lookup: dict[tuple[int, int], MetadataRow] = {}
        self._rebuild_lookup()

    def _rebuild_lookup(self) -> None:
        self._lookup.clear()
        if self.df.is_empty():
            return

        for record in self.df.iter_rows(named=True):
            key = (int(record["segment"]), int(record["row"]))
            self._lookup[key] = MetadataRow(
                id=int(record["id"]),
                text=str(record["text"]),
                segment=int(record["segment"]),
                row=int(record["row"]),
            )

    def append(
        self,
        ids: Sequence[int],
        texts: Sequence[str],
        segment_id: int,
    ) -> None:
        rows = np.arange(len(ids), dtype=np.int64)
        new_df = pl.DataFrame(
            {
                "id": ids,
                "text": texts,
                "segment": [segment_id] * len(ids),
                "row": rows,
            },
            schema=self.SCHEMA,
        )
        self.df = pl.concat([self.df, new_df])

        for chunk_id, text, row in zip(ids, texts, rows):
            key = (segment_id, int(row))
            self._lookup[key] = MetadataRow(
                id=int(chunk_id),
                text=str(text),
                segment=segment_id,
                row=int(row),
            )

    def get(self, segment_id: int, row: int) -> MetadataRow | None:
        return self._lookup.get((segment_id, row))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df.write_parquet(self.path)
