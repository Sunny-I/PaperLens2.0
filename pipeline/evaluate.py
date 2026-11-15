"""Evaluate retrieval against a user-supplied, labeled PDF corpus.

Usage: python -m pipeline.evaluate PDF_DIRECTORY CASES_JSON [--top-k 6] [--min-score 0.05]
"""
import argparse
import json
import statistics
from pathlib import Path

from pipeline.corpus import ingest


def evaluate(store, cases, top_k=6, min_score=0.05):
    if not cases:
        raise ValueError("Provide at least one evaluation case.")
    recalls, reciprocal_ranks, abstentions, details, latencies = [], [], [], [], []
    known_pages = {}
    for chunk in store.chunks_metadata:
        known_pages.setdefault(chunk["filename"], set()).add(chunk["page"])
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each evaluation case must be an object.")
        query, expected = case["query"], case["relevant"]
        if not isinstance(query, str) or not query.strip() or not isinstance(expected, list):
            raise ValueError("Each case needs a nonempty query and a relevant list.")
        for label in expected:
            if not isinstance(label, dict) or label.get("filename") not in known_pages:
                raise ValueError("A relevance label refers to a PDF not present in the index. Check the evaluation corpus and labels.")
            page = label.get("page")
            if page is not None and (type(page) is not int or page not in known_pages[label["filename"]]):
                raise ValueError("A relevance label refers to a page with no indexed text. Check the one-based page reference.")
        hits, latency = store.search(query, top_k=top_k, min_score=min_score)
        latencies.append(latency)
        def matches(hit, label):
            return hit["filename"] == label["filename"] and (label.get("page") is None or hit["page"] == label["page"])
        if expected:
            recall = sum(any(matches(hit, label) for hit in hits) for label in expected) / len(expected)
            rank = next((i for i, hit in enumerate(hits, 1) if any(matches(hit, label) for label in expected)), None)
            recalls.append(recall)
            reciprocal_ranks.append(1 / rank if rank else 0)
        else:
            abstentions.append(not hits)
        details.append({"query": query, "relevant": expected,
                        "retrieved": [{"filename": h["filename"], "page": h["page"], "score": h["similarity_score"]} for h in hits]})
    return {"cases": len(cases), "top_k": top_k, "min_score": min_score,
            "mean_recall_at_k": statistics.mean(recalls) if recalls else None,
            "mean_reciprocal_rank": statistics.mean(reciprocal_ranks) if reciprocal_ranks else None,
            "unanswerable_abstention_rate": statistics.mean(abstentions) if abstentions else None,
            "mean_retrieval_ms": round(statistics.mean(latencies), 3), "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_directory", type=Path)
    parser.add_argument("cases_json", type=Path)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--min-score", type=float, default=0.05)
    args = parser.parse_args()
    try:
        corpus = ingest([(p.name, p.read_bytes()) for p in sorted(args.pdf_directory.glob("*.pdf"))])
        report = evaluate(corpus.store, json.loads(args.cases_json.read_text()), args.top_k, args.min_score)
        report["index"] = corpus.stats
        print(json.dumps(report, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
