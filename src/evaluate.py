# Evaluation script for comparing embedding models
# This script runs a set of test questions against different embedding models
# and measures retrieval quality, timing, and other metrics.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import SUPPORTED_MODELS, get_embedding_model
from src.rag import get_db, get_collection_name_for_model


# --- Test Dataset ----------------------------------------------------

# Each test case has:
# - question: The question to ask
# - expected_sources: List of source files that SHOULD be retrieved (partial match OK)
# - expected_keywords: Keywords that should appear in retrieved chunks
# - difficulty: easy/medium/hard (for analysis)

TEST_QUESTIONS = [
    # Easy questions - direct facts
    {
        "question": "Who is Gandalf?",
        "expected_sources": ["gandalf"],
        "expected_keywords": ["wizard", "istari", "maia"],
        "difficulty": "easy",
    },
    {
        "question": "Who is Frodo Baggins?",
        "expected_sources": ["frodo"],
        "expected_keywords": ["hobbit", "ring", "shire"],
        "difficulty": "easy",
    },
    {
        "question": "What is the One Ring?",
        "expected_sources": ["ring", "sauron"],
        "expected_keywords": ["sauron", "power", "mordor"],
        "difficulty": "easy",
    },
    {
        "question": "Who is Aragorn?",
        "expected_sources": ["aragorn"],
        "expected_keywords": ["king", "ranger", "gondor"],
        "difficulty": "easy",
    },
    {
        "question": "What is Mordor?",
        "expected_sources": ["mordor", "sauron"],
        "expected_keywords": ["sauron", "dark", "mount doom"],
        "difficulty": "easy",
    },
    
    # Medium questions - require some inference
    {
        "question": "How did Gandalf die and return?",
        "expected_sources": ["gandalf"],
        "expected_keywords": ["balrog", "white", "moria"],
        "difficulty": "medium",
    },
    {
        "question": "What happened at the Council of Elrond?",
        "expected_sources": ["elrond", "council", "fellowship"],
        "expected_keywords": ["ring", "fellowship", "decide"],
        "difficulty": "medium",
    },
    {
        "question": "Who are the members of the Fellowship?",
        "expected_sources": ["fellowship"],
        "expected_keywords": ["frodo", "gandalf", "aragorn", "legolas", "gimli"],
        "difficulty": "medium",
    },
    {
        "question": "What is the significance of Rivendell?",
        "expected_sources": ["rivendell", "elrond"],
        "expected_keywords": ["elves", "refuge", "imladris"],
        "difficulty": "medium",
    },
    {
        "question": "How was Sauron defeated?",
        "expected_sources": ["sauron", "ring"],
        "expected_keywords": ["ring", "destroy", "mount doom"],
        "difficulty": "medium",
    },
    
    # Hard questions - specific details or connections
    {
        "question": "What is the relationship between Bilbo and Frodo?",
        "expected_sources": ["bilbo", "frodo"],
        "expected_keywords": ["uncle", "cousin", "heir", "adopted"],
        "difficulty": "hard",
    },
    {
        "question": "Why couldn't the Eagles fly the Ring to Mordor?",
        "expected_sources": ["eagle", "ring"],
        "expected_keywords": ["nazgul", "sauron", "corrupt"],
        "difficulty": "hard",
    },
    {
        "question": "What are the Silmarils?",
        "expected_sources": ["silmaril", "feanor"],
        "expected_keywords": ["jewel", "light", "feanor"],
        "difficulty": "hard",
    },
    {
        "question": "Who is Tom Bombadil?",
        "expected_sources": ["tom", "bombadil"],
        "expected_keywords": ["old", "forest", "goldberry"],
        "difficulty": "hard",
    },
    {
        "question": "What is the difference between Sauron and Morgoth?",
        "expected_sources": ["sauron", "morgoth", "melkor"],
        "expected_keywords": ["servant", "first", "dark lord"],
        "difficulty": "hard",
    },
    
    # Swedish questions (to test multilingual)
    {
        "question": "Vem är Gandalf?",
        "expected_sources": ["gandalf"],
        "expected_keywords": ["wizard", "istari", "trollkarl"],
        "difficulty": "easy",
        "language": "sv",
    },
    {
        "question": "Vad är Härskarringen?",
        "expected_sources": ["ring", "sauron"],
        "expected_keywords": ["sauron", "power", "makt"],
        "difficulty": "medium",
        "language": "sv",
    },
    {
        "question": "Vilka är medlemmarna i Ringens brödraskap?",
        "expected_sources": ["fellowship"],
        "expected_keywords": ["frodo", "gandalf", "aragorn"],
        "difficulty": "medium",
        "language": "sv",
    },
]


# --- Evaluation Result -----------------------------------------------


@dataclass
class RetrievalResult:
    """Result of a single retrieval test."""
    question: str
    difficulty: str
    language: str
    
    # Retrieval metrics
    retrieved_sources: list[str]
    expected_sources: list[str]
    source_hit: bool  # Did we retrieve at least one expected source?
    source_precision: float  # What fraction of retrieved sources were expected?
    
    # Keyword metrics
    retrieved_text: str
    expected_keywords: list[str]
    keyword_hits: int  # How many expected keywords found?
    keyword_recall: float  # What fraction of expected keywords were found?
    
    # Scores
    top_score: float | None
    all_scores: list[float]
    
    # Timing
    retrieval_time_ms: float


@dataclass 
class ModelEvaluationResult:
    """Aggregated results for one embedding model."""
    model_name: str
    model_type: str
    
    # Aggregate metrics
    total_questions: int
    source_hit_rate: float  # % of questions where we got at least one right source
    avg_source_precision: float
    avg_keyword_recall: float
    avg_top_score: float
    avg_retrieval_time_ms: float
    total_retrieval_time_ms: float
    
    # By difficulty
    easy_hit_rate: float
    medium_hit_rate: float
    hard_hit_rate: float
    
    # Individual results
    results: list[RetrievalResult]
    
    # Metadata
    timestamp: str
    index_build_time_ms: float | None


# --- Evaluation Logic ------------------------------------------------


def evaluate_retrieval(
    db,
    question: str,
    expected_sources: list[str],
    expected_keywords: list[str],
    k: int = 4,
) -> RetrievalResult:
    """Run a single retrieval and evaluate the results."""
    
    start_time = time.perf_counter()
    
    try:
        results = db.similarity_search_with_score(question, k=k)
    except Exception as e:
        print(f"  Error during retrieval: {e}")
        results = []
    
    retrieval_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Extract sources and text
    retrieved_sources = []
    retrieved_texts = []
    scores = []
    
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        retrieved_sources.append(source)
        retrieved_texts.append(doc.page_content)
        scores.append(float(score))
    
    combined_text = " ".join(retrieved_texts).lower()
    
    # Calculate source metrics
    source_hits = 0
    for expected in expected_sources:
        expected_lower = expected.lower()
        for retrieved in retrieved_sources:
            if expected_lower in retrieved.lower():
                source_hits += 1
                break
    
    source_hit = source_hits > 0
    source_precision = source_hits / max(1, len(retrieved_sources))
    
    # Calculate keyword metrics
    keyword_hits = 0
    for keyword in expected_keywords:
        if keyword.lower() in combined_text:
            keyword_hits += 1
    
    keyword_recall = keyword_hits / max(1, len(expected_keywords))
    
    # Convert distance to relevance score
    top_score = None
    if scores:
        top_score = 1.0 / (1.0 + scores[0])
    
    return RetrievalResult(
        question=question,
        difficulty="unknown",
        language="en",
        retrieved_sources=retrieved_sources,
        expected_sources=expected_sources,
        source_hit=source_hit,
        source_precision=source_precision,
        retrieved_text=combined_text[:500],  # Truncate for storage
        expected_keywords=expected_keywords,
        keyword_hits=keyword_hits,
        keyword_recall=keyword_recall,
        top_score=top_score,
        all_scores=scores,
        retrieval_time_ms=retrieval_time_ms,
    )


def evaluate_model(
    model_name: str,
    persist_dir: str,
    collection_name: str,
    test_questions: list[dict],
    k: int = 4,
    rebuild_index: bool = False,
) -> ModelEvaluationResult:
    """
    Evaluate a single embedding model on all test questions.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*60}")
    
    model_info = SUPPORTED_MODELS.get(model_name, {})
    model_type = model_info.get("type", "unknown")
    
    # Get or build the index
    full_collection = get_collection_name_for_model(collection_name, model_name)
    
    # Use model-specific subdirectory (matching ingest.py behavior)
    model_persist_dir = Path(persist_dir) / model_name.replace("/", "_")
    
    index_build_time = None
    
    # Check if we need to build the index
    if rebuild_index or not model_persist_dir.exists():
        print(f"  Building index for {model_name}...")
        # We'll handle index building separately
        index_build_time = 0  # Placeholder
    
    print(f"  Loading database from {model_persist_dir}...")
    db = get_db(
        persist_dir=str(model_persist_dir),
        collection_name=full_collection,
        embedding_model=model_name,
    )
    
    # Run evaluations
    results: list[RetrievalResult] = []
    
    for i, test_case in enumerate(test_questions, 1):
        question = test_case["question"]
        expected_sources = test_case["expected_sources"]
        expected_keywords = test_case["expected_keywords"]
        difficulty = test_case.get("difficulty", "unknown")
        language = test_case.get("language", "en")
        
        print(f"  [{i}/{len(test_questions)}] {question[:50]}...")
        
        result = evaluate_retrieval(
            db=db,
            question=question,
            expected_sources=expected_sources,
            expected_keywords=expected_keywords,
            k=k,
        )
        
        # Add metadata
        result = RetrievalResult(
            **{**asdict(result), "difficulty": difficulty, "language": language}
        )
        
        results.append(result)
        
        status = "✓" if result.source_hit else "✗"
        score_str = f"{result.top_score:.3f}" if result.top_score is not None else "N/A"
        print(f"       {status} source_hit={result.source_hit}, "
              f"keywords={result.keyword_hits}/{len(expected_keywords)}, "
              f"score={score_str}, "
              f"time={result.retrieval_time_ms:.1f}ms")
    
    # Calculate aggregates
    total = len(results)
    source_hits = sum(1 for r in results if r.source_hit)
    
    easy_results = [r for r in results if r.difficulty == "easy"]
    medium_results = [r for r in results if r.difficulty == "medium"]
    hard_results = [r for r in results if r.difficulty == "hard"]
    
    return ModelEvaluationResult(
        model_name=model_name,
        model_type=model_type,
        total_questions=total,
        source_hit_rate=source_hits / max(1, total),
        avg_source_precision=sum(r.source_precision for r in results) / max(1, total),
        avg_keyword_recall=sum(r.keyword_recall for r in results) / max(1, total),
        avg_top_score=sum(r.top_score or 0 for r in results) / max(1, total),
        avg_retrieval_time_ms=sum(r.retrieval_time_ms for r in results) / max(1, total),
        total_retrieval_time_ms=sum(r.retrieval_time_ms for r in results),
        easy_hit_rate=sum(1 for r in easy_results if r.source_hit) / max(1, len(easy_results)),
        medium_hit_rate=sum(1 for r in medium_results if r.source_hit) / max(1, len(medium_results)),
        hard_hit_rate=sum(1 for r in hard_results if r.source_hit) / max(1, len(hard_results)),
        results=results,
        timestamp=datetime.now().isoformat(),
        index_build_time_ms=index_build_time,
    )


def print_comparison_table(evaluations: list[ModelEvaluationResult]) -> None:
    """Print a comparison table of all models."""
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Model':<30} {'Type':<12} {'Hit Rate':<10} {'Precision':<10} "
          f"{'Keywords':<10} {'Avg Time':<10}")
    print("-"*80)
    
    for eval_result in evaluations:
        print(f"{eval_result.model_name:<30} "
              f"{eval_result.model_type:<12} "
              f"{eval_result.source_hit_rate*100:>6.1f}%   "
              f"{eval_result.avg_source_precision*100:>6.1f}%   "
              f"{eval_result.avg_keyword_recall*100:>6.1f}%   "
              f"{eval_result.avg_retrieval_time_ms:>6.1f}ms")
    
    # By difficulty
    print("\n" + "-"*80)
    print("BY DIFFICULTY:")
    print(f"\n{'Model':<30} {'Easy':<12} {'Medium':<12} {'Hard':<12}")
    print("-"*60)
    
    for eval_result in evaluations:
        print(f"{eval_result.model_name:<30} "
              f"{eval_result.easy_hit_rate*100:>6.1f}%     "
              f"{eval_result.medium_hit_rate*100:>6.1f}%     "
              f"{eval_result.hard_hit_rate*100:>6.1f}%")


def save_results(evaluations: list[ModelEvaluationResult], output_path: Path) -> None:
    """Save evaluation results to JSON."""
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "models": [
            {
                "model_name": e.model_name,
                "model_type": e.model_type,
                "metrics": {
                    "source_hit_rate": e.source_hit_rate,
                    "avg_source_precision": e.avg_source_precision,
                    "avg_keyword_recall": e.avg_keyword_recall,
                    "avg_top_score": e.avg_top_score,
                    "avg_retrieval_time_ms": e.avg_retrieval_time_ms,
                    "total_retrieval_time_ms": e.total_retrieval_time_ms,
                },
                "by_difficulty": {
                    "easy": e.easy_hit_rate,
                    "medium": e.medium_hit_rate,
                    "hard": e.hard_hit_rate,
                },
                "results": [asdict(r) for r in e.results],
            }
            for e in evaluations
        ],
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {output_path}")


# --- CLI -------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate and compare embedding models for RAG retrieval"
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["text-embedding-3-small", "all-MiniLM-L6-v2"],
        help="Embedding models to evaluate",
    )
    p.add_argument(
        "--persist-dir",
        type=str,
        default=os.getenv("RAG_PERSIST_DIR", "data/chroma"),
        help="Chroma persist directory",
    )
    p.add_argument(
        "--collection",
        type=str,
        default=os.getenv("RAG_COLLECTION", "tolkien_lore"),
        help="Base collection name",
    )
    p.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of documents to retrieve",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="Output file for results",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List supported models and exit",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    
    if args.list_models:
        print("Supported embedding models:")
        print("-" * 60)
        for name, info in SUPPORTED_MODELS.items():
            print(f"  {name}")
            print(f"    Type: {info['type']}")
            print(f"    Dimension: {info['dimension']}")
            print(f"    Description: {info['description']}")
            print()
        return
    
    persist_dir = args.persist_dir
    if not Path(persist_dir).is_absolute():
        persist_dir = str(PROJECT_ROOT / persist_dir)
    
    evaluations: list[ModelEvaluationResult] = []
    
    for model_name in args.models:
        if model_name not in SUPPORTED_MODELS:
            print(f"Warning: Unknown model '{model_name}', skipping...")
            continue
        
        result = evaluate_model(
            model_name=model_name,
            persist_dir=persist_dir,
            collection_name=args.collection,
            test_questions=TEST_QUESTIONS,
            k=args.k,
        )
        evaluations.append(result)
    
    if evaluations:
        print_comparison_table(evaluations)
        save_results(evaluations, Path(args.output))


if __name__ == "__main__":
    main()