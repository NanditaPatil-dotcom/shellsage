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
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
ROOT_DIR = Path(__file__).parent
DOCS_DIR = ROOT_DIR / "docs"
EMB_DIR = ROOT_DIR / "embeddings"
INDEX_PATH = EMB_DIR / "shell_docs.index"
META_PATH = EMB_DIR / "shell_docs_meta.json"
META_HASH_PATH = EMB_DIR / "shell_docs_meta.hash"

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
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


def load_text_files() -> List[Tuple[str, str]]:
    _ensure_dirs()
    texts: List[Tuple[str, str]] = []
    for p in sorted(DOCS_DIR.glob("*.txt")):
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            rel = str(p.relative_to(ROOT_DIR))
            texts.append((rel, content))
        except Exception:
            continue
    return texts


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def _compute_docs_hash(docs: List[Tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for path, text in docs:
        h.update(path.encode())
        h.update(text.encode())
    return h.hexdigest()


def build_faiss_index() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    docs = load_text_files()
    if not docs:
        index = faiss.IndexFlatIP(384)
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

    # Save hash
    hash_val = _compute_docs_hash(docs)
    META_HASH_PATH.write_text(hash_val, encoding="utf-8")

    return index, meta


def save_index_and_meta(index: faiss.IndexFlatIP, meta: List[Dict]) -> None:
    _ensure_dirs()
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index_and_meta() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
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
    Rebuilds index if:
      - force_rebuild=True
      - docs changed since last index (tracked by hash)
      - index or meta files missing
    """
    _ensure_dirs()
    docs = load_text_files()
    docs_hash = _compute_docs_hash(docs)

    if force_rebuild or not (INDEX_PATH.exists() and META_PATH.exists()):
        index, meta = build_faiss_index()
        save_index_and_meta(index, meta)
        return index, meta

    # Check if hash matches
    if META_HASH_PATH.exists():
        existing_hash = META_HASH_PATH.read_text(encoding="utf-8")
        if existing_hash != docs_hash:
            index, meta = build_faiss_index()
            save_index_and_meta(index, meta)
            return index, meta

    return load_index_and_meta()


def retrieve_context(query: str, top_k: int = 3) -> List[Dict]:
    index, meta = build_or_load_index(force_rebuild=False)
    if index.ntotal == 0 or not meta:
        return []

    embedder = _get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    q_emb = _normalize(q_emb)
    scores, ids = index.search(q_emb, k=min(top_k, index.ntotal))
    results: List[Dict] = []
    for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
        if idx < 0 or idx >= len(meta):
            continue
        item = dict(meta[idx])
        item["score"] = float(score)
        results.append(item)
    return results


if __name__ == "__main__":
    print("[ShellSage] Building FAISS index from docs/ ...")
    idx, m = build_or_load_index(force_rebuild=True)
    print(f"[ShellSage] Indexed {idx.ntotal} documents. Saved to {INDEX_PATH}.")
