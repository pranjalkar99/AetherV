from __future__ import annotations

import json
from pathlib import Path

from aetherv.types import SegmentRecord


class Manifest:
    """Tracks on-disk vector segments."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            raw = json.loads(path.read_text())
            self.segments = [
                SegmentRecord.from_dict(entry) for entry in raw["segments"]
            ]
        else:
            self.segments = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"segments": [segment.to_dict() for segment in self.segments]}
        self.path.write_text(json.dumps(payload, indent=2))

    def add_segment(self, record: SegmentRecord) -> None:
        self.segments.append(record)
        self.save()

    def next_segment_id(self) -> int:
        if not self.segments:
            return 0
        return max(segment.segment_id for segment in self.segments) + 1
