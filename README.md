# BIS Standards Recommendation Engine

AI-powered RAG system for recommending BIS Building Materials standards to Micro and Small Enterprises (MSEs).

## Architecture

```
Query Input
    │
    ▼
Query Expansion (abbreviation resolution: OPC→Ordinary Portland Cement, etc.)
    │
    ▼
Hybrid Retriever
├── BM25 (sparse, keyword-weighted)         weight: 40%
├── TF-IDF (sparse, n-gram bigrams)         weight: 20%
└── Semantic (sentence-transformers MiniLM) weight: 40%
    │
    ▼
Reciprocal Score Fusion + Category Boost
    │
    ▼
Top-K BIS Standards with Rationale
```

**Knowledge Base:** 60+ BIS standards from SP 21 (Building Materials) covering Cement, Steel, Concrete, Aggregates, Masonry, Tiles, Pipes, Admixtures, Fly Ash, GGBS, Silica Fume.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run Inference

```bash
python inference.py --input input.json --output output.json
```

**Input format:**
```json
[{"id": 1, "query": "OPC 53 grade cement for high strength concrete"}]
```

**Output format:**
```json
[{"id": 1, "retrieved_standards": ["IS 12269", "IS 8112", "IS 269"], "latency_seconds": 0.85}]
```

## Evaluate

```bash
python eval_script.py --results data/sample_results.json
```

## Chunking & Retrieval Strategy

Each BIS standard is represented as a structured document chunk containing:
- Standard ID (e.g., IS 12269)
- Title (human-readable name)
- Domain-specific keywords (manually curated synonyms and abbreviations)
- Category (Cement, Steel, Concrete, Aggregates, etc.)
- Description (concise summary)

**Query Expansion:** A curated expansion dictionary resolves industry abbreviations (OPC, TMT, Fe500, GGBS, etc.) into full terms before retrieval, significantly boosting recall for domain-specific queries.

**Hybrid Scoring:** BM25 handles exact keyword matches well; TF-IDF handles bigram partial matches; Sentence Transformers handle semantic similarity for paraphrased or natural language queries. Scores are min-max normalized and linearly combined.

## Repo Structure

```
├── inference.py          # Main entry point (judges run this)
├── eval_script.py        # Evaluation: Hit Rate @3, MRR @5, Latency
├── requirements.txt
├── input.json            # Sample public test queries
├── src/
│   ├── rag_pipeline.py   # Hybrid retriever + BIS knowledge base
│   └── utils.py
└── data/
    └── sample_results.json
```
