# 🏗️ BIS Standards Recommendation Engine

> AI-powered RAG system that turns product descriptions into accurate BIS standard recommendations in seconds — built for the **BIS × SS Hackathon 2025**.

---

## 📌 What It Does

Indian MSEs spend weeks identifying which Bureau of Indian Standards (BIS) regulations apply to their products. This system solves that.

**Type a product description → Get the top BIS standards instantly.**

```
Input:  "OPC 53 grade cement for high strength concrete"

Output: IS 12269  →  Ordinary Portland Cement (53 Grade) - Specification
        IS 8112   →  Ordinary Portland Cement (43 Grade) - Specification  
        IS 269    →  Ordinary Portland Cement (33 Grade) - Specification

Latency: 0.003s
```

---

## 🏆 Evaluation Results

| Metric | Our Score | Target |
|--------|-----------|--------|
| Hit Rate @3 | **100%** | > 80% |
| MRR @5 | **1.00** | > 0.70 |
| Avg Latency | **< 0.05s** | < 5s |

---

## 🧠 Architecture

```
User Query
    │
    ▼
Query Expansion
(OPC → Ordinary Portland Cement, TMT → Thermo-Mechanically Treated, Fe500 → high strength bars...)
    │
    ▼
┌───────────────────────────────────────────────────────┐
│                   Hybrid Retriever                    │
│                                                       │
│   BM25 (40%)  +  TF-IDF (20%)  +  Semantic (40%)      │
│   keyword        bigram phrase    sentence-transformers│
└───────────────────────────────────────────────────────┘
    │
    ▼
Score Fusion  (min-max normalize + weighted combine + category boost)
    │
    ▼
Top-K BIS Standards  (ID · Title · Rationale · Category)
```

**Knowledge Base:** 60+ standards from BIS SP 21 (Building Materials)
covering Cement · Steel · Concrete · Aggregates · Masonry · Tiles · Pipes · Admixtures · Fly Ash · GGBS · Silica Fume · and more.

---

## 📁 Repository Structure

```
BIS-/
├── inference.py          # ← Judges run this
├── app.py                # FastAPI backend
├── streamlit_app.py      # Streamlit frontend UI
├── eval_script.py        # Official evaluation script (unmodified)
├── requirements.txt
├── input.json            # Public test queries
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── rag_pipeline.py   # Core: BIS knowledge base + hybrid retriever
│   └── utils.py
│
└── data/
    └── sample_results.json   # Public test set results
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/itiz-ksh/BIS-
cd BIS-

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run Inference (judges use this)

```bash
python inference.py --input input.json --output output.json
```

### 3. Evaluate

```bash
python eval_script.py --results data/sample_results.json
```

### 4. Launch the Web UI

Open **two terminals** from the repo root:

```bash
# Terminal 1 — Backend
uvicorn app:app --reload --port 8000

# Terminal 2 — Frontend
streamlit run streamlit_app.py
```

Then open **http://localhost:8501**

---

## 🔍 How Retrieval Works

### Chunking Strategy

Each BIS standard is stored as a structured semantic chunk:

| Field | Example |
|-------|---------|
| `id` | IS 12269 |
| `title` | Ordinary Portland Cement (53 Grade) |
| `keywords` | "53 grade, high strength, OPC, clinker, prestressed concrete..." |
| `category` | Cement |
| `description` | Short curated summary |

Keywords are **manually curated** — not just extracted from the document. For IS 1786 we include "TMT", "HYSD", "Fe415", "Fe500", "Fe550", "rebar" — all the ways a practitioner actually refers to that standard.

### Query Expansion

A curated dictionary resolves domain abbreviations before retrieval:

| Abbreviation | Expands To |
|---|---|
| OPC | Ordinary Portland Cement |
| TMT | Thermo-Mechanically Treated bars |
| GGBS / GGBFS | Ground Granulated Blast Furnace Slag |
| Fe500 | Fe500 high strength deformed bars |
| RCC | Reinforced Cement Concrete |
| PPC | Portland Pozzolana Cement |
| SRC | Sulphate Resisting Portland Cement |

### Retrieval Layers

| Layer | Weight | Strength |
|-------|--------|----------|
| BM25 (custom) | 40% | Exact keyword + bigram match, IS code lookup |
| TF-IDF (scikit-learn) | 20% | Partial phrase, n-gram coverage |
| Sentence Transformers (MiniLM) | 40% | Natural language, use-case queries |

All scores are min-max normalized then linearly combined. A small category boost is applied when the query explicitly mentions a material category.

---

## 📋 API Reference

Once the backend is running at `http://localhost:8000`:

### `POST /recommend`

```json
// Request
{
  "query": "sulphate resisting cement for underground foundations",
  "top_k": 5
}

// Response
{
  "query": "...",
  "results": [
    {
      "standard_id": "IS 12330",
      "title": "Sulphate Resisting Portland Cement - Specification",
      "score": 0.8241,
      "rationale": "Matches on: sulphate, foundations. Specification for sulphate resisting portland cement.",
      "category": "Cement"
    }
  ],
  "latency_seconds": 0.003,
  "expanded_terms": ["SRC → Sulphate Resisting Portland"]
}
```

### `GET /standards`
Returns all 60+ standards in the knowledge base.

### `GET /health`
Health check.

---

## 📊 Output JSON Schema

```json
[
  {
    "id": 1,
    "retrieved_standards": ["IS 12269", "IS 8112", "IS 269", "IS 455", "IS 1489-1"],
    "latency_seconds": 0.003
  }
]
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — TF-IDF vectorizer
- **BM25** — custom implementation (no external dependency)
- **Sentence Transformers** — `all-MiniLM-L6-v2` for semantic search
- **FastAPI + Uvicorn** — REST backend
- **Streamlit** — web UI
- **Dataset** — BIS SP 21 (Building Materials)

---

## 👥 Team

Built for the **BIS × SS Hackathon 2025** — *Accelerating MSE Compliance: Automating BIS Standard Discovery*

---

## 📄 License

MIT
