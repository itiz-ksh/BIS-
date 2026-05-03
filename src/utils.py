"""Utility functions for BIS RAG System"""
import json
import pickle
import time
from pathlib import Path
from typing import Any, List, Dict
import re


# Simple debug printer
def debug_print(enabled: bool, *args, **kwargs):
    if enabled:
        print(*args, **kwargs)


class Timer:
    """Simple timer for measuring latency"""
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self) -> float:
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed


def load_json(path: str) -> Any:
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    """Save data to JSON file"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_pickle(path: str) -> Any:
    """Load pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Any, path: str) -> None:
    """Save data to pickle file"""
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_standard_id(standard_id: str) -> str:
    """Normalize standard ID for matching.

    Rules:
    - Trim and collapse spaces
    - Uppercase
    - Ensure it starts with 'IS ' (attempt to fix common variants)
    - Return None if it cannot be reasonably normalized
    """
    if standard_id is None:
        return None

    s = str(standard_id).strip().upper()
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)

    # Common variant: 'IS12269' -> 'IS 12269'
    if s.startswith('IS') and not s.startswith('IS '):
        rest = s[2:].strip()
        if rest:
            s = 'IS ' + rest

    # If it starts with digits, prefix IS
    if re.match(r'^\d', s):
        s = 'IS ' + s

    # Final sanity check: must start with 'IS ' followed by digits/word
    if not s.startswith('IS '):
        return None

    return s


def preprocess_query(query: str) -> (str, List[str]):
    """Preprocess query: expand abbreviations, lowercase, remove noise words.

    Returns processed string and list of tokens.
    """
    if not query:
        return "", []

    q = str(query)
    # Lowercase for processing
    q_low = q.lower()

    # Expand common abbreviations
    abbr_map = {
        'opc': 'ordinary portland cement',
        'ppc': 'portland pozzolana cement',
        'rcc': 'reinforced cement concrete'
    }
    for abbr, expansion in abbr_map.items():
        q_low = re.sub(rf"\b{re.escape(abbr)}\b", expansion, q_low)

    # Remove punctuation except hyphens and digits
    q_low = re.sub(r"[^a-z0-9\-\s]", " ", q_low)

    # Tokenize and remove noise/stop words
    noise = {
        'the', 'a', 'an', 'for', 'of', 'in', 'on', 'and', 'with', 'by', 'to',
        'specification', 'specifications', 'standard', 'standards', 'grade'
    }
    tokens = [t for t in q_low.split() if t and t not in noise]

    processed = ' '.join(tokens)
    return processed, tokens


def extract_numeric_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if re.search(r"\d", t)]
