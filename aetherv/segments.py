from __future__ import annotations

from pathlib import Path


class SegmentManager:
    """Resolves segment file paths on disk."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(exist_ok=True, parents=True)

    def path_for(self, segment_name: str) -> Path:
        return self.root / segment_name
