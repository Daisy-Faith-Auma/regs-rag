"""Ingest a regulations PDF into a local vector store.

Pipeline: extract text per page -> overlapping chunks -> Gemini embeddings
(batched, cached) -> save chunks + vectors to .cache/store.npz

Usage:
    python ingest.py regulations.pdf
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from google import genai
from pypdf import PdfReader

EMBED_MODEL = "gemini-embedding-001"
CHUNK_WORDS = 350          # ~500 tokens
OVERLAP_WORDS = 60
BATCH_SIZE = 20            # small batches for free-tier rate limits
BATCH_PAUSE_S = 20         # pause between batches (free tier); set 0 if billing enabled
MAX_RETRIES = 6
CACHE_DIR = Path(".cache")


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page with content."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    """Split page text into overlapping word-window chunks, keeping page numbers."""
    chunks = []
    for page_num, text in pages:
        words = text.split()
        step = CHUNK_WORDS - OVERLAP_WORDS
        for start in range(0, max(len(words), 1), step):
            piece = " ".join(words[start:start + CHUNK_WORDS])
            if len(piece) < 80:  # skip fragments
                continue
            chunks.append({"page": page_num, "text": piece})
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks


def embed_chunks(client: genai.Client, chunks: list[dict]) -> np.ndarray:
    """Embed all chunks, using a content-hash cache so re-runs are free."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "embeddings.npz"
    cache: dict[str, np.ndarray] = {}
    if cache_file.exists():
        loaded = np.load(cache_file)
        cache = {k: loaded[k] for k in loaded.files}

    def key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    todo = [(i, c["text"]) for i, c in enumerate(chunks) if key(c["text"]) not in cache]
    print(f"{len(chunks)} chunks total, {len(todo)} need embedding (rest cached)")

    from google.genai import errors as genai_errors

    for b in range(0, len(todo), BATCH_SIZE):
        batch = todo[b:b + BATCH_SIZE]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=[t for _, t in batch],
                )
                break
            except genai_errors.ClientError as e:
                if e.code == 429 and attempt < MAX_RETRIES:
                    wait = min(60 * attempt, 300)
                    print(f"  rate limited (429) — waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    # save what we have so a re-run resumes from here
                    np.savez_compressed(cache_file, **cache)
                    print("  progress saved to cache — re-run ingest.py to resume")
                    raise
        for (_, text), emb in zip(batch, result.embeddings):
            cache[key(text)] = np.array(emb.values, dtype=np.float32)
        # save after EVERY batch: a crash or quota hit never loses progress
        np.savez_compressed(cache_file, **cache)
        print(f"  embedded {min(b + BATCH_SIZE, len(todo))}/{len(todo)}")
        if b + BATCH_SIZE < len(todo) and BATCH_PAUSE_S:
            time.sleep(BATCH_PAUSE_S)

    np.savez_compressed(cache_file, **cache)
    return np.stack([cache[key(c["text"])] for c in chunks])


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python ingest.py regulations.pdf")
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("set GEMINI_API_KEY first")

    client = genai.Client()
    pages = extract_pages(sys.argv[1])
    print(f"extracted {len(pages)} pages")
    chunks = chunk_pages(pages)
    vectors = embed_chunks(client, chunks)

    # normalise once so retrieval is a pure dot product
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(CACHE_DIR / "store.npz", vectors=vectors)
    (CACHE_DIR / "chunks.json").write_text(json.dumps(chunks))
    print(f"store built: {len(chunks)} chunks -> .cache/store.npz")


if __name__ == "__main__":
    main()