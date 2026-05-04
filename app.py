"""
FastAPI backend for BIS Standards Recommendation Engine.
Run: uvicorn app:app --reload --port 8000
"""

import time
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag_pipeline import BISRecommendationEngine, expand_query, BIS_STANDARDS_DB, QUERY_EXPANSIONS

app = FastAPI(title="BIS Standards Recommendation Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = BISRecommendationEngine(top_k=5)
id_to_std = {s["id"]: s for s in BIS_STANDARDS_DB}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class StandardResult(BaseModel):
    standard_id: str
    title: str
    score: float
    rationale: str
    category: str


class QueryResponse(BaseModel):
    query: str
    results: List[StandardResult]
    latency_seconds: float
    expanded_terms: List[str]


@app.get("/health")
def health():
    return {"status": "ok", "standards_loaded": len(BIS_STANDARDS_DB)}


@app.post("/recommend", response_model=QueryResponse)
def recommend(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    t0 = time.time()
    raw_results = engine.retriever.search(req.query, top_k=req.top_k)

    results = []
    for r in raw_results:
        meta = id_to_std.get(r.standard_id, {})
        results.append(StandardResult(
            standard_id=r.standard_id,
            title=r.title,
            score=round(r.score, 4),
            rationale=r.rationale,
            category=meta.get("category", "General"),
        ))

    expanded_terms = []
    q_lower = req.query.lower()
    for abbr, full in QUERY_EXPANSIONS.items():
        if abbr in q_lower:
            short = " ".join(full.split()[:3])
            expanded_terms.append(f"{abbr.upper()} → {short}")

    return QueryResponse(
        query=req.query,
        results=results,
        latency_seconds=round(time.time() - t0, 3),
        expanded_terms=expanded_terms,
    )


@app.get("/standards")
def list_standards():
    return {
        "total": len(BIS_STANDARDS_DB),
        "categories": sorted(set(s["category"] for s in BIS_STANDARDS_DB)),
        "standards": [{"id": s["id"], "title": s["title"], "category": s["category"]} for s in BIS_STANDARDS_DB],
    }
