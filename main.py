"""
FastAPI backend for Tolkien RAG Embedding Comparison
Wraps the existing RAG/embeddings logic for the React frontend.

Usage:
    uvicorn main:app --reload --port 8000

Install extras:
    pip install fastapi uvicorn python-multipart
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup – same as the existing src/ modules expect
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Tolkien RAG Embedding API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", PROJECT_ROOT / "data" / "chroma"))
COLLECTION_BASE = os.getenv("CHROMA_COLLECTION", "tolkien_lore")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EVAL_JSON = PROJECT_ROOT / "data" / "evaluation.json"

SUPPORTED_MODELS = {
    "text-embedding-3-small": {
        "type": "openai",
        "dimension": 1536,
        "description": "OpenAI's small model — good cost/performance balance",
        "color": "#22d3ee",   # cyan
    },
    "all-MiniLM-L6-v2": {
        "type": "huggingface",
        "dimension": 384,
        "description": "Fast, lightweight English model. Popular for RAG.",
        "color": "#a78bfa",   # violet
    },
    "multilingual-e5-base": {
        "type": "huggingface",
        "dimension": 768,
        "description": "Multilingual model — great for Swedish/English mix.",
        "color": "#34d399",   # emerald
    },
}


def _get_indexed_models() -> list[str]:
    """Return which models actually have a built Chroma index."""
    indexed = []
    for model in SUPPORTED_MODELS:
        model_dir = PERSIST_DIR / model.replace("/", "_")
        if model_dir.exists() and (model_dir / "chroma.sqlite3").exists():
            indexed.append(model)
    return indexed


# Cache for open DB connections
_db_cache: dict[str, Any] = {}


def _get_db(embedding_model: str):
    if embedding_model in _db_cache:
        return _db_cache[embedding_model]

    try:
        from src.rag import get_db, get_collection_name_for_model
    except ModuleNotFoundError:
        from rag import get_db, get_collection_name_for_model

    model_dir = PERSIST_DIR / embedding_model.replace("/", "_")
    collection = get_collection_name_for_model(COLLECTION_BASE, embedding_model)
    db = get_db(
        persist_dir=str(model_dir),
        collection_name=collection,
        embedding_model=embedding_model,
    )
    _db_cache[embedding_model] = db
    return db


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    embedding_model: str = "text-embedding-3-small"
    k: int = 4
    threshold: float = 0.35
    last_topic: str | None = None
    last_language: str | None = None


class CompareRequest(BaseModel):
    question: str
    models: list[str] | None = None   # None = all indexed models
    k: int = 4
    threshold: float = 0.35


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "Tolkien RAG Embedding API"}


@app.get("/models")
def list_models():
    """Return all supported models with metadata and index availability."""
    indexed = _get_indexed_models()
    result = {}
    for name, info in SUPPORTED_MODELS.items():
        result[name] = {**info, "indexed": name in indexed}
    return result


@app.post("/chat")
def chat(req: ChatRequest):
    """Single-model RAG query."""
    try:
        from src.rag import answer_question
    except ModuleNotFoundError:
        from rag import answer_question

    if req.embedding_model not in SUPPORTED_MODELS:
        raise HTTPException(400, f"Unknown model: {req.embedding_model}")

    indexed = _get_indexed_models()
    if req.embedding_model not in indexed:
        raise HTTPException(
            503,
            f"Model '{req.embedding_model}' has no built index. Run ingest.py first.",
        )

    db = _get_db(req.embedding_model)
    t0 = time.perf_counter()
    result = answer_question(
        db=db,
        question=req.question,
        last_topic=req.last_topic,
        last_language=req.last_language,
        k=req.k,
        threshold=req.threshold,
        chat_model=CHAT_MODEL,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "answer": result.answer,
        "sources": result.sources,
        "top_score": result.top_score,
        "topic": result.topic,
        "language": result.language,
        "resolved_question": result.resolved_question,
        "retrieval_time_ms": elapsed_ms,
        "embedding_model": req.embedding_model,
    }


@app.post("/compare")
def compare(req: CompareRequest):
    """Query all (or selected) indexed models in parallel and return all answers."""
    try:
        from src.rag import answer_question
    except ModuleNotFoundError:
        from rag import answer_question

    indexed = _get_indexed_models()
    models_to_use = req.models or indexed
    models_to_use = [m for m in models_to_use if m in indexed]

    if not models_to_use:
        raise HTTPException(503, "No indexed models available. Run ingest.py first.")

    results = []
    for model_name in models_to_use:
        db = _get_db(model_name)
        t0 = time.perf_counter()
        try:
            result = answer_question(
                db=db,
                question=req.question,
                k=req.k,
                threshold=req.threshold,
                chat_model=CHAT_MODEL,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            results.append({
                "model": model_name,
                "answer": result.answer,
                "sources": result.sources,
                "top_score": result.top_score,
                "retrieval_time_ms": elapsed_ms,
                "success": True,
            })
        except Exception as e:
            results.append({
                "model": model_name,
                "error": str(e),
                "success": False,
                "retrieval_time_ms": (time.perf_counter() - t0) * 1000,
            })

    return {"question": req.question, "results": results}


@app.get("/evaluation")
def get_evaluation():
    """Return the saved evaluation.json results."""
    # Check a few common locations
    candidates = [
        EVAL_JSON,
        PROJECT_ROOT / "evaluation.json",
        PROJECT_ROOT / "src" / "evaluation.json",
        PROJECT_ROOT / "data" / "eval" / "evaluation.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)

    raise HTTPException(404, "evaluation.json not found. Run evaluate.py first.")


@app.get("/health")
def health():
    indexed = _get_indexed_models()
    return {
        "status": "ok",
        "indexed_models": indexed,
        "eval_available": any(p.exists() for p in [
            EVAL_JSON,
            PROJECT_ROOT / "evaluation.json",
        ]),
    }