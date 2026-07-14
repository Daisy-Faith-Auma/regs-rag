"""Ask questions of the ingested regulations. CLI or Streamlit.

CLI:        python app.py "How does Attack Mode work?"
Streamlit:  streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from google import genai

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"
TOP_K = 6
MIN_SCORE = 0.55           # below this, retrieval is probably off-topic
CACHE_DIR = Path(".cache")

SYSTEM = (
    "You are an assistant answering questions about the FIA Formula E "
    "Sporting Regulations. Answer ONLY from the provided excerpts. "
    "Cite pages inline like [page 42]. If the excerpts do not contain "
    "the answer, say so plainly — never guess."
)


def load_store() -> tuple[np.ndarray, list[dict]]:
    vectors = np.load(CACHE_DIR / "store.npz")["vectors"]
    chunks = json.loads((CACHE_DIR / "chunks.json").read_text())
    return vectors, chunks


def retrieve(client: genai.Client, question: str,
             vectors: np.ndarray, chunks: list[dict]) -> list[dict]:
    q = client.models.embed_content(model=EMBED_MODEL, contents=question)
    qv = np.array(q.embeddings[0].values, dtype=np.float32)
    qv /= np.linalg.norm(qv)
    scores = vectors @ qv                      # cosine (vectors pre-normalised)
    top = np.argsort(scores)[::-1][:TOP_K]
    return [chunks[i] | {"score": float(scores[i])} for i in top]


def answer(client: genai.Client, question: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[page {h['page']}] {h['text']}" for h in hits)
    resp = client.models.generate_content(
        model=GEN_MODEL,
        contents=f"{SYSTEM}\n\nEXCERPTS:\n{context}\n\nQUESTION: {question}",
    )
    return resp.text


def cli(question: str) -> None:
    client = genai.Client()
    vectors, chunks = load_store()
    hits = retrieve(client, question, vectors, chunks)
    print(answer(client, question, hits))
    print("\nSources:", ", ".join(f"p.{h['page']} ({h['score']:.2f})" for h in hits))


def ui() -> None:
    import streamlit as st

    st.set_page_config(page_title="Regs RAG", page_icon="🏁")
    st.title("🏁 Race Regulations Assistant")
    st.caption("Grounded answers from the FIA Formula E Sporting Regulations")

    if not (CACHE_DIR / "store.npz").exists():
        st.error("Run `python ingest.py regulations.pdf` first.")
        st.stop()

    client = genai.Client()
    vectors, chunks = load_store()

    question = st.text_input("Ask a question", placeholder="How does Attack Mode work?")
    if question:
        with st.spinner("Retrieving…"):
            hits = retrieve(client, question, vectors, chunks)
            if hits[0]["score"] < MIN_SCORE:
                st.warning(
                    f"Weak retrieval (best match {hits[0]['score']:.2f}) — "
                    "this may not be covered by the regulations."
                )
            st.markdown(answer(client, question, hits))
        with st.expander("Retrieved passages"):
            for h in hits:
                st.markdown(f"**page {h['page']}** · score {h['score']:.2f}\n\n> {h['text'][:400]}…")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli(" ".join(sys.argv[1:]))
    else:
        ui()
