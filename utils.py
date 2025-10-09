"""
ShellSage utils: document loading, embedding, FAISS indexing, and context retrieval.

- Loads text files from docs/
- Embeds with sentence-transformers/all-MiniLM-L6-v2
- Builds a cosine-similarity FAISS index (inner product on normalized vectors)
- Saves index to embeddings/shell_docs.index and metadata to embeddings/shell_docs_meta.json
- Provides retrieve_context(query, top_k=3) for the CLI to use

Run directly to (re)build the index:
  python utils.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
ROOT_DIR = Path(__file__).parent
DOCS_DIR = ROOT_DIR / "docs"
EMB_DIR = ROOT_DIR / "embeddings"
INDEX_PATH = EMB_DIR / "shell_docs.index"
META_PATH = EMB_DIR / "shell_docs_meta.json"

# Globals (lazy-loaded)
_EMBEDDER: SentenceTransformer | None = None
_INDEX: faiss.IndexFlatIP | None = None
_META: List[Dict] | None = None


def _ensure_dirs() -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        # Use a small, fast CPU-friendly model
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


def load_text_files() -> List[Tuple[str, str]]:
    """
    Load all .txt files from docs/ (non-recursive) as (relative_path, content).
    """
    _ensure_dirs()
    texts: List[Tuple[str, str]] = []
    for p in sorted(DOCS_DIR.glob("*.txt")):
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            rel = str(p.relative_to(ROOT_DIR))
            texts.append((rel, content))
        except Exception:
            # Skip unreadable files
            continue
    return texts


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors to use inner product as cosine similarity in FAISS.
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def build_faiss_index() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Build a FAISS index from docs/ and return (index, meta).

    meta[i] = {"path": ..., "text": ...} corresponds to index id i.
    """
    docs = load_text_files()
    if not docs:
        # Create an empty index if no docs exist yet (CLI will still function)
        index = faiss.IndexFlatIP(384)  # 384 dims for MiniLM-L6-v2
        meta: List[Dict] = []
        return index, meta

    embedder = _get_embedder()
    corpus = [text for _, text in docs]
    embeddings = embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    embeddings = _normalize(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    meta: List[Dict] = [{"path": path, "text": text} for path, text in docs]
    return index, meta


def save_index_and_meta(index: faiss.IndexFlatIP, meta: List[Dict]) -> None:
    _ensure_dirs()
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index_and_meta() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Load FAISS index and metadata mapping. If not present, build them.
    """
    global _INDEX, _META
    _ensure_dirs()

    if INDEX_PATH.exists() and META_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        _INDEX, _META = index, meta
        return index, meta

    index, meta = build_faiss_index()
    save_index_and_meta(index, meta)
    _INDEX, _META = index, meta
    return index, meta


def build_or_load_index(force_rebuild: bool = False) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Build the index if missing or force_rebuild; otherwise load existing.
    """
    if force_rebuild or not (INDEX_PATH.exists() and META_PATH.exists()):
        index, meta = build_faiss_index()
        save_index_and_meta(index, meta)
        return index, meta
    return load_index_and_meta()


def retrieve_context(query: str, top_k: int = 3) -> List[Dict]:
    """
    Retrieve the top-k most relevant docs for the query.

    Returns: list of {"path": str, "text": str, "score": float}
    """
    index, meta = build_or_load_index(force_rebuild=False)
    if index.ntotal == 0 or not meta:
        return []

    embedder = _get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    q_emb = _normalize(q_emb)
    scores, ids = index.search(q_emb, k=min(top_k, index.ntotal))
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: List[Dict] = []
    for idx, score in zip(ids, scores):
        if idx < 0 or idx >= len(meta):
            continue
        item = dict(meta[idx])
        item["score"] = float(score)
        results.append(item)
    return results


if __name__ == "__main__":
    # Rebuild index when run directly
    print("[ShellSage] Building FAISS index from docs/ ...")
    idx, m = build_or_load_index(force_rebuild=True)
    print(f"[ShellSage] Indexed {idx.ntotal} documents. Saved to {INDEX_PATH}.")
