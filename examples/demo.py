from __future__ import annotations

from aetherv import VectorDB


def main() -> None:
    db = VectorDB()
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
    print()
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
