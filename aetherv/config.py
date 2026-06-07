from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Runtime configuration for a vector database instance."""

    root: Path = Path("vectordb")
    segment_size: int = 10_000
    metadata_filename: str = "metadata.parquet"
    manifest_filename: str = "manifest.json"
    segments_dirname: str = "segments"

    @property
    def metadata_path(self) -> Path:
        return self.root / self.metadata_filename

    @property
    def manifest_path(self) -> Path:
        return self.root / self.manifest_filename

    @property
    def segments_dir(self) -> Path:
        return self.root / self.segments_dirname
