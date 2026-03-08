# Build DB from .txt files
# Updated to support multiple embedding models for thesis comparison

from __future__ import annotations

import argparse
import os
import sys
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import get_db, get_collection_name_for_model
from src.embeddings import SUPPORTED_MODELS, list_supported_models


# --- Loading --------------------------------------------------------


def _read_text_best_effort(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def load_txt_documents(raw_dir: Path) -> list[Document]:
    docs: list[Document] = []

    for path in sorted(raw_dir.rglob("*.txt")):
        text = _read_text_best_effort(path).strip()
        if not text:
            continue

        try:
            source = path.relative_to(PROJECT_ROOT).as_posix()
        except Exception:
            source = path.as_posix()

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "title": path.stem.replace("_", " ").strip(),
                },
            )
        )

    return docs


# --- Chunk IDs ------------------------------------------------------


def assign_chunk_ids(chunks: list[Document]) -> list[str]:
    per_source_counter: dict[str, int] = {}
    ids: list[str] = []

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown_source"))
        idx = per_source_counter.get(source, 0)
        per_source_counter[source] = idx + 1

        chunk.metadata["chunk_index"] = idx
        ids.append(f"{source}::chunk_{idx}")

    return ids


# --- CLI ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build/update Chroma index from .txt files (supports multiple embedding models)"
    )
    p.add_argument(
        "--rebuild", action="store_true", help="Clean rebuild the index from scratch"
    )
    p.add_argument("--chunk-size", type=int, default=900, help="Chunk-size (digits)")
    p.add_argument("--chunk-overlap", type=int, default=150, help="Overlap (digits)")
    p.add_argument(
        "--collection",
        type=str,
        default=os.getenv(
            "CHROMA_COLLECTION", os.getenv("RAG_COLLECTION", "tolkien_lore")
        ),
        help="Base Chroma collection name",
    )
    p.add_argument(
        "--persist-dir",
        type=str,
        default=os.getenv(
            "CHROMA_PERSIST_DIR", os.getenv("RAG_PERSIST_DIR", "data/chroma")
        ),
        help="Directory to Chroma persist-mapp",
    )
    p.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Embedding model to use (see --list-models)",
    )
    p.add_argument(
        "--all-models",
        action="store_true",
        help="Build indexes for all supported models (for comparison)",
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Build indexes for specific models (space-separated)",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List supported embedding models and exit",
    )
    return p.parse_args()


# --- Pipeline -------------------------------------------------------


def build_index(
    *,
    raw_dir: Path,
    persist_dir: Path,
    collection: str,
    embedding_model: str,
    rebuild: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[int, int, float]:
    """Build/update the Chroma index.

    Returns: (num_docs, num_chunks, build_time_seconds)
    """

    # Disable Chroma telemetry noise early
    os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Cant find data/raw/ directory: {raw_dir}")

    # Use model-specific collection name
    full_collection = get_collection_name_for_model(collection, embedding_model)
    model_persist_dir = persist_dir / embedding_model.replace("/", "_")

    if rebuild and model_persist_dir.exists():
        shutil.rmtree(model_persist_dir)

    docs = load_txt_documents(raw_dir)
    if not docs:
        raise ValueError(f"No .txt documents found in: {raw_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    ids = assign_chunk_ids(chunks)

    print(f"  Building embeddings with {embedding_model}...")
    start_time = time.perf_counter()
    
    db = get_db(
        persist_dir=str(model_persist_dir),
        collection_name=full_collection,
        embedding_model=embedding_model,
    )
    db.add_documents(chunks, ids=ids)
    
    build_time = time.perf_counter() - start_time

    return len(docs), len(chunks), build_time


def main() -> None:
    load_dotenv()
    args = parse_args()

    # List models and exit
    if args.list_models:
        print("Supported embedding models:")
        print("-" * 60)
        for name, info in list_supported_models().items():
            print(f"  {name}")
            print(f"    Type: {info['type']}")
            print(f"    Dimension: {info['dimension']}")
            print(f"    Description: {info['description']}")
            print()
        return

    raw_dir = PROJECT_ROOT / "data" / "raw"
    persist_dir = Path(args.persist_dir)
    if not persist_dir.is_absolute():
        persist_dir = PROJECT_ROOT / persist_dir

    # Determine which models to build
    if args.all_models:
        models_to_build = list(SUPPORTED_MODELS.keys())
    elif args.models:
        models_to_build = args.models
    else:
        models_to_build = [args.embedding_model]

    # Validate models
    for model in models_to_build:
        if model not in SUPPORTED_MODELS:
            print(f"Error: Unknown model '{model}'")
            print(f"Use --list-models to see supported models")
            return

    # Build indexes
    results = []
    for model in models_to_build:
        print(f"\n{'='*60}")
        print(f"Building index for: {model}")
        print(f"{'='*60}")
        
        try:
            num_docs, num_chunks, build_time = build_index(
                raw_dir=raw_dir,
                persist_dir=persist_dir,
                collection=args.collection,
                embedding_model=model,
                rebuild=bool(args.rebuild),
                chunk_size=int(args.chunk_size),
                chunk_overlap=int(args.chunk_overlap),
            )
            results.append({
                "model": model,
                "docs": num_docs,
                "chunks": num_chunks,
                "time": build_time,
                "success": True,
            })
            print(f"  ✓ Done! {num_docs} docs, {num_chunks} chunks in {build_time:.1f}s")
        except Exception as e:
            results.append({
                "model": model,
                "error": str(e),
                "success": False,
            })
            print(f"  ✗ Error: {e}")

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"\n{'Model':<35} {'Docs':<8} {'Chunks':<10} {'Time':<10}")
        print("-" * 65)
        for r in results:
            if r["success"]:
                print(f"{r['model']:<35} {r['docs']:<8} {r['chunks']:<10} {r['time']:.1f}s")
            else:
                print(f"{r['model']:<35} FAILED: {r['error'][:30]}")
        
        print(f"\nIndexes saved in: {persist_dir}")


if __name__ == "__main__":
    main()