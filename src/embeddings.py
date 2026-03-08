# Embedding model factory for comparing different embedding models
# Supports both OpenAI (paid) and open-source (HuggingFace) models

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

from langchain_core.embeddings import Embeddings


@dataclass
class EmbeddingResult:
    """Metadata about an embedding operation."""
    model_name: str
    model_type: Literal["openai", "huggingface"]
    embedding_time_ms: float
    dimension: int


# --- Supported Models ------------------------------------------------

SUPPORTED_MODELS = {
    # OpenAI models (paid, API-based)
    "text-embedding-3-small": {
        "type": "openai",
        "dimension": 1536,
        "description": "OpenAI's small embedding model, good balance of cost/performance",
    },
    "text-embedding-3-large": {
        "type": "openai", 
        "dimension": 3072,
        "description": "OpenAI's large embedding model, highest quality",
    },
    # Open-source models (free, runs locally)
    "all-MiniLM-L6-v2": {
        "type": "huggingface",
        "dimension": 384,
        "description": "Fast, lightweight English model. Popular choice for RAG.",
    },
    "multilingual-e5-base": {
        "type": "huggingface",
        "model_name": "intfloat/multilingual-e5-base",
        "dimension": 768,
        "description": "Multilingual model, good for Swedish/English mix.",
    },
    "all-mpnet-base-v2": {
        "type": "huggingface",
        "dimension": 768,
        "description": "Higher quality English model, slower than MiniLM.",
    },
}


def list_supported_models() -> dict:
    """Return dict of all supported models with their metadata."""
    return SUPPORTED_MODELS.copy()


def get_model_info(model_name: str) -> dict | None:
    """Get info about a specific model."""
    return SUPPORTED_MODELS.get(model_name)


# --- Factory ---------------------------------------------------------


def get_embedding_model(model_name: str) -> Embeddings:
    """
    Factory function that returns the appropriate embedding model.
    
    Args:
        model_name: Name of the model (see SUPPORTED_MODELS)
        
    Returns:
        A LangChain Embeddings instance
        
    Raises:
        ValueError: If model is not supported
    """
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Supported models: {supported}"
        )
    
    model_info = SUPPORTED_MODELS[model_name]
    model_type = model_info["type"]
    
    if model_type == "openai":
        return _get_openai_embeddings(model_name)
    elif model_type == "huggingface":
        # Some models have a different HuggingFace name
        hf_name = model_info.get("model_name", f"sentence-transformers/{model_name}")
        return _get_huggingface_embeddings(hf_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _get_openai_embeddings(model_name: str) -> Embeddings:
    """Get OpenAI embeddings model."""
    from langchain_openai import OpenAIEmbeddings
    
    return OpenAIEmbeddings(model=model_name)


def _get_huggingface_embeddings(model_name: str) -> Embeddings:
    """Get HuggingFace/sentence-transformers embeddings model."""
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # Use CPU by default, can be changed to "cuda" if GPU available
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


# --- Timing utilities ------------------------------------------------


class TimedEmbeddings:
    """Wrapper that tracks embedding time for benchmarking."""
    
    def __init__(self, embeddings: Embeddings, model_name: str):
        self.embeddings = embeddings
        self.model_name = model_name
        self.model_type = SUPPORTED_MODELS.get(model_name, {}).get("type", "unknown")
        self.total_time_ms: float = 0.0
        self.call_count: int = 0
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        start = time.perf_counter()
        result = self.embeddings.embed_documents(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        self.total_time_ms += elapsed_ms
        self.call_count += 1
        
        return result
    
    def embed_query(self, text: str) -> list[float]:
        start = time.perf_counter()
        result = self.embeddings.embed_query(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        self.total_time_ms += elapsed_ms
        self.call_count += 1
        
        return result
    
    def get_stats(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "total_time_ms": self.total_time_ms,
            "call_count": self.call_count,
            "avg_time_ms": self.total_time_ms / max(1, self.call_count),
        }


def get_timed_embedding_model(model_name: str) -> TimedEmbeddings:
    """Get an embedding model wrapped with timing functionality."""
    base_model = get_embedding_model(model_name)
    return TimedEmbeddings(base_model, model_name)