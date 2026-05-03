"""
BIS Standards Recommendation Engine - Inference Script
Usage:
    python inference.py --input input.json --output output.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_pipeline import BISRecommendationEngine
from utils import load_json, save_json


def main():
    parser = argparse.ArgumentParser(description="BIS Standards Recommendation Engine")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, default="output.json", help="Path to output JSON")
    parser.add_argument("--top-k", type=int, default=5, help="Number of standards to return")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild vector index")
    args = parser.parse_args()

    queries = load_json(args.input)
    print(f"Loaded {len(queries)} queries from {args.input}")

    engine = BISRecommendationEngine(top_k=args.top_k, rebuild_index=args.rebuild_index)

    results = []
    for i, item in enumerate(queries, 1):
        query_id = item.get("id", i)
        query_text = item.get("query", "")
        print(f"[{i}/{len(queries)}] {query_text[:70]}...")

        t0 = time.time()
        standards = engine.query(query_text)
        latency = round(time.time() - t0, 3)

        results.append({
            "id": query_id,
            "retrieved_standards": standards,
            "latency_seconds": latency
        })
        print(f"  -> {standards} ({latency}s)")

    save_json(results, args.output)
    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
