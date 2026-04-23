"""
compute_embedding_coords.py
---------------------------
Project test question embeddings to 2D with UMAP for each model.
Output: data/embedding_coords.json

Run from project root:
    python -m src.compute_embedding_coords
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dotenv import load_dotenv

try:
    import umap
except ImportError:
    raise SystemExit("UMAP saknas. Installera med:  pip install umap-learn")

from src.embeddings import get_embedding_model
from src.evaluate import TEST_QUESTIONS

# Läs .env så OPENAI_API_KEY blir tillgänglig
load_dotenv()


MODELS = [
    "text-embedding-3-small",
    "all-MiniLM-L6-v2",
    "multilingual-e5-base",
]

OUTPUT_PATH = Path("data/embedding_coords.json")
EVAL_PATH = Path("data/evaluation.json")


def _question_text(item: dict[str, Any]) -> str:
    return str(item.get("question", "")).strip()


def _question_difficulty(item: dict[str, Any]) -> str:
    return str(item.get("difficulty", "unknown")).strip().lower()


def load_evaluation_hits() -> dict[str, dict[str, dict[str, Any]]]:
    """Load {model_name: {question: {source_hit, top_score}}} from evaluation.json."""
    if not EVAL_PATH.exists():
        print(
            "Warning: data/evaluation.json not found, skipping source_hit/top_score enrichment"
        )
        return {}

    with EVAL_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for model in data.get("models", []):
        model_name = model.get("model_name")
        if not model_name:
            continue
        lookup: dict[str, dict[str, Any]] = {}
        for row in model.get("results", []):
            question = row.get("question")
            if not question:
                continue
            lookup[question] = {
                "source_hit": bool(row.get("source_hit", False)),
                "top_score": row.get("top_score"),
            }
        by_model[model_name] = lookup
    return by_model


def embed_questions(model_name: str, questions: list[str]) -> np.ndarray:
    """Return embeddings as a (N, dim) float32 array."""
    print(f"  Embedding {len(questions)} questions with {model_name}")
    embedder = get_embedding_model(model_name)

    if hasattr(embedder, "embed_documents"):
        vectors = embedder.embed_documents(questions)
    elif hasattr(embedder, "embed_query"):
        vectors = [embedder.embed_query(q) for q in questions]
    elif callable(embedder):
        vectors = embedder(questions)
    else:
        raise RuntimeError(f"Unsupported embedding interface for model: {model_name}")

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(
            f"Expected 2D embeddings for {model_name}, got shape {arr.shape}"
        )
    return arr


def project_2d(vectors: np.ndarray, seed: int = 42) -> np.ndarray:
    """Project vectors with UMAP using settings suitable for small datasets."""
    n_samples = int(vectors.shape[0])
    n_neighbors = min(10, max(2, n_samples - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.25,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(vectors)


def main() -> None:
    eval_hits = load_evaluation_hits()
    output: dict[str, list[dict[str, Any]]] = {}

    questions_meta = [
        {
            "question": _question_text(item),
            "difficulty": _question_difficulty(item),
        }
        for item in TEST_QUESTIONS
        if _question_text(item)
    ]

    questions_only = [item["question"] for item in questions_meta]

    for model_name in MODELS:
        print(f"\nProcessing {model_name}")
        try:
            vectors = embed_questions(model_name, questions_only)
            coords = project_2d(vectors)
        except Exception as exc:
            print(f"  Failed for {model_name}: {exc}")
            continue

        model_hits = eval_hits.get(model_name, {})
        points: list[dict[str, Any]] = []

        for idx, q in enumerate(questions_meta):
            hit_info = model_hits.get(q["question"], {})
            points.append(
                {
                    "question": q["question"],
                    "difficulty": q["difficulty"],
                    "x": float(coords[idx][0]),
                    "y": float(coords[idx][1]),
                    "source_hit": bool(hit_info.get("source_hit", False)),
                    "top_score": hit_info.get("top_score"),
                }
            )

        output[model_name] = points

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {OUTPUT_PATH}")
    for model_name, points in output.items():
        print(f"  {model_name}: {len(points)} points")


if __name__ == "__main__":
    main()
