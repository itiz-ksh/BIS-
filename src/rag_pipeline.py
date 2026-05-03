"""
BIS RAG Pipeline - Core Engine
Implements hybrid BM25 + TF-IDF + semantic retrieval with cross-encoder reranking.
Designed to run on CPU with no GPU required.
"""

import os
import json
import pickle
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np

# ── BIS Standards Knowledge Base ──────────────────────────────────────────────
# Sourced from BIS SP 21 (Building Materials). Each entry:
# standard_id, title, keywords, category, description
BIS_STANDARDS_DB = [
    # CEMENT
    {"id": "IS 269", "title": "Ordinary Portland Cement (33 Grade) - Specification",
     "keywords": ["ordinary portland cement", "OPC", "33 grade", "cement", "clinker", "gypsum", "setting time", "compressive strength"],
     "category": "Cement", "description": "Specification for ordinary portland cement 33 grade for general construction."},
    {"id": "IS 8112", "title": "Ordinary Portland Cement (43 Grade) - Specification",
     "keywords": ["ordinary portland cement", "OPC", "43 grade", "cement", "concrete", "mortar", "high strength", "plaster"],
     "category": "Cement", "description": "Specification for OPC 43 grade used in general civil construction."},
    {"id": "IS 12269", "title": "Ordinary Portland Cement (53 Grade) - Specification",
     "keywords": ["ordinary portland cement", "OPC", "53 grade", "high strength cement", "prestressed concrete", "precast"],
     "category": "Cement", "description": "Specification for OPC 53 grade used in high-strength and prestressed concrete."},
    {"id": "IS 455", "title": "Portland Slag Cement - Specification",
     "keywords": ["portland slag cement", "PSC", "slag", "blast furnace slag", "blended cement", "low heat"],
     "category": "Cement", "description": "Specification for portland slag cement made with granulated blast furnace slag."},
    {"id": "IS 1489-1", "title": "Portland Pozzolana Cement (Fly Ash Based) - Specification",
     "keywords": ["portland pozzolana cement", "PPC", "fly ash", "blended cement", "durability", "pozzolana"],
     "category": "Cement", "description": "Specification for PPC fly ash based cement for general construction."},
    {"id": "IS 1489-2", "title": "Portland Pozzolana Cement (Calcined Clay Based) - Specification",
     "keywords": ["portland pozzolana cement", "PPC", "calcined clay", "pozzolana", "blended cement"],
     "category": "Cement", "description": "Specification for PPC calcined clay based cement."},
    {"id": "IS 6452", "title": "High Alumina Cement for Structural Use - Specification",
     "keywords": ["high alumina cement", "HAC", "aluminous cement", "refractory", "rapid hardening", "heat resistant"],
     "category": "Cement", "description": "Specification for high alumina cement for structural and refractory use."},
    {"id": "IS 8041", "title": "Rapid Hardening Portland Cement - Specification",
     "keywords": ["rapid hardening cement", "RHPC", "early strength", "fast setting", "precast concrete"],
     "category": "Cement", "description": "Specification for rapid hardening portland cement."},
    {"id": "IS 12330", "title": "Sulphate Resisting Portland Cement - Specification",
     "keywords": ["sulphate resisting cement", "SRC", "sulphate attack", "marine", "underground structures", "foundations"],
     "category": "Cement", "description": "Specification for sulphate resisting portland cement for aggressive environments."},
    {"id": "IS 3466", "title": "Masonry Cement - Specification",
     "keywords": ["masonry cement", "brickwork", "block work", "mortar", "plastering"],
     "category": "Cement", "description": "Specification for masonry cement used in brickwork and plastering."},

    # STEEL / REINFORCEMENT
    {"id": "IS 1786", "title": "High Strength Deformed Steel Bars and Wires for Concrete Reinforcement",
     "keywords": ["TMT bars", "HYSD bars", "high strength deformed bars", "reinforcement", "Fe415", "Fe500", "Fe550", "Fe600", "rebar", "steel bars", "concrete reinforcement"],
     "category": "Steel", "description": "Specification for TMT/HYSD bars used as reinforcement in concrete structures."},
    {"id": "IS 432-1", "title": "Mild Steel and Medium Tensile Steel Bars for Concrete Reinforcement",
     "keywords": ["mild steel bars", "round bars", "plain bars", "Fe250", "reinforcement", "concrete"],
     "category": "Steel", "description": "Specification for mild steel round bars for concrete reinforcement."},
    {"id": "IS 2062", "title": "Hot Rolled Medium and High Tensile Structural Steel - Specification",
     "keywords": ["structural steel", "hot rolled", "I-beam", "channel", "angle", "plate", "section", "fabrication"],
     "category": "Steel", "description": "Specification for hot rolled structural steel used in fabrication and construction."},
    {"id": "IS 1977", "title": "Low Tensile Structural Steel - Specification",
     "keywords": ["low tensile steel", "structural steel", "general purpose"],
     "category": "Steel", "description": "Specification for low tensile structural steel."},
    {"id": "IS 1239-1", "title": "Mild Steel Tubes, Tubulars and Other Wrought Steel Fittings",
     "keywords": ["mild steel tubes", "pipes", "tubulars", "water supply", "gas", "fittings"],
     "category": "Steel", "description": "Specification for mild steel tubes and fittings for water/gas."},
    {"id": "IS 2830", "title": "Mild Steel Channels - Specification",
     "keywords": ["mild steel channels", "C-section", "structural channel"],
     "category": "Steel", "description": "Specification for mild steel channels."},
    {"id": "IS 808", "title": "Dimensions for Hot Rolled Steel Beam, Column, Channel and Angle Sections",
     "keywords": ["steel sections", "I section", "H section", "beam", "column", "channel", "angle", "dimensions"],
     "category": "Steel", "description": "Dimensions and properties of hot rolled steel sections."},
    {"id": "IS 1161", "title": "Steel Tubes for Structural Purposes",
     "keywords": ["steel tubes", "hollow section", "structural tube", "CHS", "SHS", "RHS"],
     "category": "Steel", "description": "Specification for steel tubes used in structural applications."},
    {"id": "IS 6240", "title": "Hot Rolled Steel Plate (up to 6mm) Sheet and Strip",
     "keywords": ["steel plate", "steel sheet", "hot rolled", "strip", "thin plate"],
     "category": "Steel", "description": "Specification for hot rolled steel plate, sheet and strip."},
    {"id": "IS 1730", "title": "Steel Plates, Sheets and Strips for Structural and General Engineering Purposes",
     "keywords": ["steel plate", "structural plate", "general engineering", "cold rolled"],
     "category": "Steel", "description": "Specification for steel plates and sheets for structural use."},

    # CONCRETE
    {"id": "IS 456", "title": "Plain and Reinforced Concrete - Code of Practice",
     "keywords": ["concrete", "RCC", "reinforced cement concrete", "plain concrete", "design", "construction", "durability", "cover", "mix design", "M20", "M25", "M30"],
     "category": "Concrete", "description": "Code of practice for plain and reinforced concrete covering design, materials, and construction."},
    {"id": "IS 10262", "title": "Concrete Mix Proportioning - Guidelines",
     "keywords": ["mix design", "mix proportioning", "concrete design mix", "water cement ratio", "workability", "slump"],
     "category": "Concrete", "description": "Guidelines for proportioning concrete mixes to achieve specified properties."},
    {"id": "IS 516", "title": "Methods of Tests for Strength of Concrete",
     "keywords": ["concrete testing", "compressive strength test", "cube test", "flexural strength", "tensile strength"],
     "category": "Concrete", "description": "Methods for testing strength of concrete including cube and cylinder tests."},
    {"id": "IS 1199", "title": "Methods of Sampling and Analysis of Concrete",
     "keywords": ["concrete sampling", "workability", "slump test", "vee-bee", "compaction factor", "flow test"],
     "category": "Concrete", "description": "Methods for sampling fresh concrete and testing workability."},
    {"id": "IS 9013", "title": "Method of Making, Curing and Determining Compressive Strength of Accelerated Cured Concrete Test Specimens",
     "keywords": ["accelerated curing", "concrete testing", "early strength", "warm water curing"],
     "category": "Concrete", "description": "Method for accelerated curing of concrete test specimens."},
    {"id": "IS 4926", "title": "Ready Mixed Concrete - Code of Practice",
     "keywords": ["ready mix concrete", "RMC", "transit mixer", "batching plant", "ready mixed"],
     "category": "Concrete", "description": "Code of practice for manufacture and supply of ready mixed concrete."},
    {"id": "IS 1343", "title": "Prestressed Concrete - Code of Practice",
     "keywords": ["prestressed concrete", "PSC", "pre-tensioning", "post-tensioning", "prestress", "tendons"],
     "category": "Concrete", "description": "Code of practice for prestressed concrete structures."},
    {"id": "IS 3370-1", "title": "Concrete Structures for Storage of Liquids - Code of Practice (Part 1)",
     "keywords": ["water tank", "liquid retaining structure", "reservoir", "sump", "waterproof concrete"],
     "category": "Concrete", "description": "Code of practice for concrete liquid retaining structures."},

    # AGGREGATES
    {"id": "IS 383", "title": "Coarse and Fine Aggregate for Concrete - Specification",
     "keywords": ["aggregate", "coarse aggregate", "fine aggregate", "sand", "gravel", "crushed stone", "sieve analysis", "gradation", "fineness modulus"],
     "category": "Aggregates", "description": "Specification for coarse and fine aggregates from natural sources for concrete."},
    {"id": "IS 2386-1", "title": "Methods of Test for Aggregates for Concrete - Part 1: Particle Size and Shape",
     "keywords": ["aggregate testing", "sieve analysis", "particle size", "grading", "flakiness index", "elongation index"],
     "category": "Aggregates", "description": "Test methods for particle size and shape of aggregates."},
    {"id": "IS 2386-2", "title": "Methods of Test for Aggregates - Part 2: Estimation of Deleterious Materials",
     "keywords": ["deleterious materials", "clay lumps", "organic impurities", "silt content", "aggregate quality"],
     "category": "Aggregates", "description": "Test methods for deleterious materials in aggregates."},
    {"id": "IS 2386-3", "title": "Methods of Test for Aggregates - Part 3: Specific Gravity, Voids, Absorption",
     "keywords": ["specific gravity", "water absorption", "aggregate", "void ratio", "bulk density"],
     "category": "Aggregates", "description": "Test methods for specific gravity and absorption of aggregates."},
    {"id": "IS 2386-4", "title": "Methods of Test for Aggregates - Part 4: Mechanical Properties",
     "keywords": ["aggregate impact value", "aggregate crushing value", "Los Angeles abrasion", "hardness", "toughness"],
     "category": "Aggregates", "description": "Test methods for mechanical properties of aggregates."},
    {"id": "IS 2386-5", "title": "Methods of Test for Aggregates - Part 5: Soundness",
     "keywords": ["soundness", "aggregate durability", "sodium sulphate", "magnesium sulphate", "freeze thaw"],
     "category": "Aggregates", "description": "Test methods for soundness of aggregates."},
    {"id": "IS 9417", "title": "Recommendations for Welding Cold Worked Bars for Reinforced Concrete Construction",
     "keywords": ["welding", "cold worked bars", "reinforcement welding"],
     "category": "Steel", "description": "Recommendations for welding of cold worked bars."},

    # BRICKS / MASONRY
    {"id": "IS 1077", "title": "Common Burnt Clay Building Bricks - Specification",
     "keywords": ["brick", "burnt clay brick", "building brick", "masonry", "compressive strength", "water absorption", "efflorescence"],
     "category": "Masonry", "description": "Specification for common burnt clay building bricks."},
    {"id": "IS 2180", "title": "Specification for Heavy Duty Burnt Clay Bricks",
     "keywords": ["heavy duty brick", "engineering brick", "flooring brick", "pavement"],
     "category": "Masonry", "description": "Specification for heavy duty burnt clay bricks."},
    {"id": "IS 3952", "title": "Burnt Clay Hollow Bricks for Walls and Partitions",
     "keywords": ["hollow brick", "cavity brick", "partition wall", "lightweight"],
     "category": "Masonry", "description": "Specification for hollow clay bricks."},
    {"id": "IS 2185-1", "title": "Concrete Masonry Units - Part 1: Hollow and Solid Concrete Blocks",
     "keywords": ["concrete block", "hollow block", "solid block", "masonry unit", "block work"],
     "category": "Masonry", "description": "Specification for hollow and solid concrete masonry blocks."},
    {"id": "IS 2212", "title": "Code of Practice for Brickwork",
     "keywords": ["brickwork", "brick masonry", "mortar joint", "coursing", "bonding"],
     "category": "Masonry", "description": "Code of practice for brickwork construction."},

    # TILES / FLOORING
    {"id": "IS 1237", "title": "Cement Concrete Flooring Tiles - Specification",
     "keywords": ["cement tile", "flooring tile", "concrete tile", "mosaic tile", "floor"],
     "category": "Tiles", "description": "Specification for cement concrete flooring tiles."},
    {"id": "IS 13630-1", "title": "Ceramic Tiles - Sampling and Basis for Acceptance",
     "keywords": ["ceramic tile", "vitrified tile", "glazed tile", "floor tile", "wall tile"],
     "category": "Tiles", "description": "Methods for sampling and acceptance of ceramic tiles."},
    {"id": "IS 15622", "title": "Pressed Ceramic Tiles - Specification",
     "keywords": ["pressed ceramic tile", "porcelain tile", "vitrified tile", "water absorption", "breaking strength"],
     "category": "Tiles", "description": "Specification for pressed ceramic tiles including vitrified and porcelain."},

    # WOOD / TIMBER
    {"id": "IS 287", "title": "Recommendations for Maximum Permissible Moisture Content of Timber",
     "keywords": ["timber moisture", "wood moisture", "seasoning", "kiln drying", "moisture content"],
     "category": "Timber", "description": "Recommendations for moisture content limits in structural and other timber."},
    {"id": "IS 4970", "title": "Keys for Fixing False Ceilings and Other Suspended Wooden Constructions",
     "keywords": ["false ceiling", "suspended ceiling", "timber fixing", "plywood"],
     "category": "Timber", "description": "Keys for fixing false ceilings and suspended wooden constructions."},

    # GLASS
    {"id": "IS 2835", "title": "Flat Transparent Sheet Glass - Specification",
     "keywords": ["flat glass", "sheet glass", "window glass", "transparent glass", "glazing"],
     "category": "Glass", "description": "Specification for flat transparent sheet glass for windows and glazing."},
    {"id": "IS 2553-1", "title": "Safety Glass - Part 1: Specification",
     "keywords": ["safety glass", "toughened glass", "tempered glass", "laminated glass", "wired glass"],
     "category": "Glass", "description": "Specification for safety glass."},

    # PAINTS / COATINGS
    {"id": "IS 2395-1", "title": "Code of Practice for Painting Concrete, Masonry and Plaster Surfaces (Preparation)",
     "keywords": ["painting", "surface preparation", "masonry paint", "concrete paint", "plaster surface"],
     "category": "Paints", "description": "Code of practice for painting concrete and masonry surfaces."},
    {"id": "IS 101-1-1", "title": "Methods of Sampling and Test for Paints, Varnishes and Related Products",
     "keywords": ["paint testing", "paint sampling", "varnish", "surface coating"],
     "category": "Paints", "description": "Test methods for paints and varnishes."},

    # WATERPROOFING
    {"id": "IS 2645", "title": "Integral Cement Waterproofing Compounds - Specification",
     "keywords": ["waterproofing compound", "integral waterproofing", "admixture", "basement", "terrace"],
     "category": "Waterproofing", "description": "Specification for integral waterproofing compounds for cement."},
    {"id": "IS 3036", "title": "Code of Practice for Use of Cold Applied Bituminous Compounds for Waterproofing",
     "keywords": ["bituminous waterproofing", "cold applied", "bitumen", "damp proof course"],
     "category": "Waterproofing", "description": "Code for cold-applied bituminous waterproofing compounds."},

    # PIPES
    {"id": "IS 1592", "title": "Asbestos Cement Pressure Pipes and Joints - Specification",
     "keywords": ["AC pipe", "asbestos cement pipe", "pressure pipe", "water main"],
     "category": "Pipes", "description": "Specification for asbestos cement pressure pipes."},
    {"id": "IS 458", "title": "Precast Concrete Pipes (with and without Reinforcement)",
     "keywords": ["concrete pipe", "precast pipe", "RCC pipe", "NP pipe", "sewerage", "drainage"],
     "category": "Pipes", "description": "Specification for precast concrete pipes for drainage and sewerage."},
    {"id": "IS 4985", "title": "Unplasticised PVC Pipes for Potable Water Supplies",
     "keywords": ["uPVC pipe", "PVC pipe", "water supply", "potable water", "plastic pipe"],
     "category": "Pipes", "description": "Specification for uPVC pipes for water supply."},

    # DOORS / WINDOWS
    {"id": "IS 4020", "title": "Wooden Flush Door Shutters (Solid Core) - Specification",
     "keywords": ["flush door", "wooden door", "solid core door", "plywood door", "interior door"],
     "category": "Doors", "description": "Specification for wooden flush door shutters with solid core."},
    {"id": "IS 1038", "title": "Steel Doors, Windows and Ventilators - Specification",
     "keywords": ["steel door", "steel window", "metal door", "metal window", "ventilator"],
     "category": "Doors", "description": "Specification for steel doors, windows, and ventilators."},

    # ADMIXTURES
    {"id": "IS 9103", "title": "Concrete Admixtures - Specification",
     "keywords": ["admixture", "plasticizer", "superplasticizer", "retarder", "accelerator", "water reducer", "air entraining"],
     "category": "Concrete", "description": "Specification for admixtures used in concrete."},
    {"id": "IS 6925", "title": "Methods of Test for Weldability of Concrete Admixtures",
     "keywords": ["admixture testing", "concrete admixture", "compatibility"],
     "category": "Concrete", "description": "Test methods for concrete admixtures."},

    # FLY ASH
    {"id": "IS 3812-1", "title": "Pulverized Fuel Ash - Part 1: For Use as Pozzolana in Cement",
     "keywords": ["fly ash", "PFA", "pulverised fuel ash", "pozzolana", "supplementary cementitious material", "SCM"],
     "category": "Cement", "description": "Specification for pulverized fuel ash for use as pozzolana."},
    {"id": "IS 3812-2", "title": "Pulverized Fuel Ash - Part 2: For Use as Admixture in Cement Mortar and Concrete",
     "keywords": ["fly ash admixture", "PFA", "concrete admixture", "mass concrete"],
     "category": "Concrete", "description": "Specification for fly ash used as admixture in concrete."},

    # GROUND GRANULATED BLAST FURNACE SLAG
    {"id": "IS 12089", "title": "Granulated Slag for Manufacture of Portland Slag Cement",
     "keywords": ["GGBS", "GGBFS", "granulated blast furnace slag", "slag cement", "ground granulated"],
     "category": "Cement", "description": "Specification for granulated slag for Portland slag cement."},
    {"id": "IS 16714", "title": "Ground Granulated Blast Furnace Slag for Use in Concrete",
     "keywords": ["GGBS", "GGBFS", "slag concrete", "supplementary cementitious", "low heat concrete"],
     "category": "Concrete", "description": "Specification for GGBS for use in concrete."},

    # SILICA FUME
    {"id": "IS 15388", "title": "Silica Fume - Specification",
     "keywords": ["silica fume", "microsilica", "condensed silica", "high performance concrete", "HPC"],
     "category": "Concrete", "description": "Specification for silica fume for use in concrete."},
]


@dataclass
class RetrievalResult:
    standard_id: str
    title: str
    score: float
    rationale: str


# ── TF-IDF Retriever ──────────────────────────────────────────────────────────

class TFIDFRetriever:
    """Sparse TF-IDF retriever over BIS standards knowledge base."""

    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.standards = BIS_STANDARDS_DB
        self._build()

    def _build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = []
        for s in self.standards:
            text = f"{s['id']} {s['title']} {' '.join(s['keywords'])} {s['description']} {s['category']}"
            corpus.append(text.lower())
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            analyzer='word'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self.vectorizer.transform([query.lower()])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


# ── BM25 Retriever ────────────────────────────────────────────────────────────

class BM25Retriever:
    """Sparse BM25 retriever over BIS standards."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.standards = BIS_STANDARDS_DB
        self._build()

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Add n-grams
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        return tokens + bigrams

    def _build(self):
        self.corpus_tokens = []
        for s in self.standards:
            text = f"{s['id']} {s['title']} {' '.join(s['keywords'])} {s['description']} {s['category']}"
            self.corpus_tokens.append(self._tokenize(text))

        self.doc_len = [len(doc) for doc in self.corpus_tokens]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.N = len(self.corpus_tokens)

        # Build IDF
        from collections import Counter, defaultdict
        self.df: Dict[str, int] = defaultdict(int)
        for doc in self.corpus_tokens:
            for term in set(doc):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return np.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        from collections import Counter
        q_tokens = self._tokenize(query)
        scores = []
        for idx, doc in enumerate(self.corpus_tokens):
            tf_map = Counter(doc)
            score = 0.0
            for term in q_tokens:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = self._idf(term)
                dl = self.doc_len[idx]
                tf_norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                score += idf * tf_norm
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scores[:top_k] if s > 0]


# ── Semantic Retriever (sentence-transformers if available) ───────────────────

class SemanticRetriever:
    """Dense semantic retriever using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.embeddings = None
        self.standards = BIS_STANDARDS_DB
        self.model_name = model_name
        self._try_load()

    def _try_load(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading semantic model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            self._build_embeddings()
            print("Semantic retriever ready.")
        except Exception as e:
            print(f"Semantic retriever unavailable ({e}), using lexical only.")

    def _build_embeddings(self):
        texts = []
        for s in self.standards:
            text = f"{s['title']}. {' '.join(s['keywords'][:6])}. {s['description']}"
            texts.append(text)
        self.embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.model is None or self.embeddings is None:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q_emb)
        cos_sim = (self.embeddings @ q_emb.T).flatten() / (norms.flatten() * q_norm + 1e-8)
        top_idx = np.argsort(cos_sim)[::-1][:top_k]
        return [(int(i), float(cos_sim[i])) for i in top_idx if cos_sim[i] > 0.1]

    @property
    def available(self) -> bool:
        return self.model is not None


# ── Query Expansion ───────────────────────────────────────────────────────────

QUERY_EXPANSIONS = {
    "opc": "ordinary portland cement",
    "ppc": "portland pozzolana cement",
    "psc": "portland slag cement",
    "rcc": "reinforced cement concrete",
    "rmc": "ready mix concrete",
    "tmt": "thermo mechanically treated bars reinforcement",
    "hysd": "high yield strength deformed bars",
    "fe415": "Fe415 high strength deformed bars TMT",
    "fe500": "Fe500 high strength deformed bars TMT",
    "fe550": "Fe550 high strength deformed bars",
    "ggbs": "ground granulated blast furnace slag",
    "ggbfs": "ground granulated blast furnace slag",
    "hac": "high alumina cement",
    "src": "sulphate resisting portland cement",
    "rhpc": "rapid hardening portland cement",
    "m20": "M20 grade concrete mix design",
    "m25": "M25 grade concrete mix design",
    "m30": "M30 grade concrete mix design",
    "hpc": "high performance concrete silica fume",
    "psc prestressed": "prestressed concrete post tensioning pre tensioning",
    "upvc": "uPVC unplasticised PVC pipe",
    "rebar": "reinforcement steel bars concrete",
    "admixture": "concrete admixture plasticizer superplasticizer",
    "fly ash": "fly ash pulverized fuel ash pozzolana",
    "silica fume": "microsilica high performance concrete",
}

def expand_query(query: str) -> str:
    q_lower = query.lower()
    expansions = [query]
    for abbr, expansion in QUERY_EXPANSIONS.items():
        if abbr in q_lower:
            expansions.append(expansion)
    return " ".join(expansions)


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(self, alpha: float = 0.4):
        """
        alpha: weight for BM25 (1-alpha for TF-IDF, semantic gets separate boost).
        """
        self.alpha = alpha
        self.bm25 = BM25Retriever()
        self.tfidf = TFIDFRetriever()
        self.semantic = SemanticRetriever()

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        expanded = expand_query(query)

        # Get scores from each retriever
        bm25_scores = {i: s for i, s in self.bm25.search(expanded, top_k=15)}
        tfidf_scores = {i: s for i, s in self.tfidf.search(expanded, top_k=15)}
        semantic_scores = {i: s for i, s in self.semantic.search(query, top_k=15)}

        # Normalize scores (min-max)
        def normalize(d: Dict[int, float]) -> Dict[int, float]:
            if not d:
                return d
            vals = list(d.values())
            mn, mx = min(vals), max(vals)
            if mx == mn:
                return {k: 1.0 for k in d}
            return {k: (v - mn) / (mx - mn) for k, v in d.items()}

        bm25_norm = normalize(bm25_scores)
        tfidf_norm = normalize(tfidf_scores)
        sem_norm = normalize(semantic_scores)

        # Combine: weighted fusion
        all_indices = set(bm25_norm) | set(tfidf_norm) | set(sem_norm)
        combined: Dict[int, float] = {}
        sem_weight = 0.5 if self.semantic.available else 0.0
        lex_weight = 1.0 - sem_weight

        for i in all_indices:
            bm_s = bm25_norm.get(i, 0.0)
            tf_s = tfidf_norm.get(i, 0.0)
            se_s = sem_norm.get(i, 0.0)
            # Lexical: BM25 gets more weight for keyword queries
            lex_score = self.alpha * bm_s + (1 - self.alpha) * tf_s
            combined[i] = lex_weight * lex_score + sem_weight * se_s

        # Category boost: if query mentions category keywords, boost related standards
        query_lower = query.lower()
        category_boosts = {
            "cement": 0.05, "concrete": 0.05, "steel": 0.05,
            "aggregate": 0.05, "brick": 0.05, "tile": 0.05,
        }
        for i in list(combined.keys()):
            std_cat = BIS_STANDARDS_DB[i]["category"].lower()
            for cat_kw, boost in category_boosts.items():
                if cat_kw in query_lower and cat_kw in std_cat:
                    combined[i] += boost

        # Sort and return top-k
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            std = BIS_STANDARDS_DB[idx]
            # Build rationale from matched keywords
            matched_kws = [kw for kw in std["keywords"] if kw.lower() in query.lower()]
            if matched_kws:
                rationale = f"Matches query on: {', '.join(matched_kws[:3])}. {std['description']}"
            else:
                rationale = std["description"]
            results.append(RetrievalResult(
                standard_id=std["id"],
                title=std["title"],
                score=score,
                rationale=rationale
            ))
        return results


# ── Main Engine ───────────────────────────────────────────────────────────────

class BISRecommendationEngine:
    def __init__(self, top_k: int = 5, rebuild_index: bool = False):
        self.top_k = top_k
        self.retriever = HybridRetriever()
        print("BIS Recommendation Engine ready.")

    def query(self, query_text: str) -> List[str]:
        results = self.retriever.search(query_text, top_k=self.top_k)
        return [r.standard_id for r in results]

    def query_with_rationale(self, query_text: str) -> List[Dict]:
        results = self.retriever.search(query_text, top_k=self.top_k)
        return [
            {"standard_id": r.standard_id, "title": r.title,
             "score": round(r.score, 4), "rationale": r.rationale}
            for r in results
        ]
