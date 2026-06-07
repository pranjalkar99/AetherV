from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


class ArrowSegment:
    """Read and write fixed-size embedding vectors as Arrow IPC files."""

    @staticmethod
    def write(path: Path, vectors: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows, dim = vectors.shape

        flat = pa.array(vectors.ravel(), type=pa.float32())
        embeddings = pa.FixedSizeListArray.from_arrays(flat, dim)
        table = pa.Table.from_arrays([embeddings], names=["embedding"])

        with pa.OSFile(str(path), "wb") as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

    @staticmethod
    def read(path: Path) -> np.ndarray:
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_file(source)
            table = reader.read_all()

        column = table.column("embedding")
        chunk = column.chunk(0)
        dim = chunk.type.list_size
        values = chunk.values.to_numpy(zero_copy_only=True)
        return values.reshape(len(chunk), dim)
