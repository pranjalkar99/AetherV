from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class SearchResult:
    score: float
    id: int
    text: str


@dataclass(frozen=True)
class SegmentRecord:
    segment_id: int
    name: str
    vector_count: int
    created_at: str
    centroid_id: int | None = None

    @classmethod
    def create(
        cls,
        segment_id: int,
        vector_count: int,
        *,
        centroid_id: int | None = None,
    ) -> SegmentRecord:
        return cls(
            segment_id=segment_id,
            name=f"segment_{segment_id:06d}.arrow",
            vector_count=vector_count,
            created_at=datetime.now(timezone.utc).isoformat(),
            centroid_id=centroid_id,
        )

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "name": self.name,
            "vector_count": self.vector_count,
            "created_at": self.created_at,
            "centroid_id": self.centroid_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SegmentRecord:
        if "segment_id" in data:
            return cls(
                segment_id=int(data["segment_id"]),
                name=str(data["name"]),
                vector_count=int(data["vector_count"]),
                created_at=str(data.get("created_at", "")),
                centroid_id=data.get("centroid_id"),
            )

        # Legacy manifest entries used "count" without segment_id.
        name = str(data["name"])
        segment_id = int(name.split("_")[1].split(".")[0])
        return cls(
            segment_id=segment_id,
            name=name,
            vector_count=int(data["count"]),
            created_at=str(data.get("created_at", "")),
            centroid_id=data.get("centroid_id"),
        )
