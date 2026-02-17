from __future__ import annotations


import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import os
import streamlit as st
from dotenv import load_dotenv

from rag.embeddings import OpenAIEmbedder
from rag.index_store import FaissIndexStore
from rag.retriever import Retriever
from rag.answer import AnswerGenerator


def inject_css() -> None:
    st.markdown(
        """
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* Cards */
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
  margin-bottom: 10px;
}
.card-title { font-weight: 700; font-size: 0.95rem; margin-bottom: 6px; }
.card-text { opacity: 0.92; font-size: 0.92rem; line-height: 1.35; }

/* Badges */
.badges { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0 10px; }
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  opacity: 0.95;
}
.badge-strong { font-weight: 650; }
.muted { opacity: 0.75; }

.panel { position: sticky; top: 1rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_rag():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    if not api_key:
        raise RuntimeError("Brak OPENAI_API_KEY w .env")

    store = FaissIndexStore(index_dir=BASE_DIR / "index")
    index, chunks = store.load()

    embedder = OpenAIEmbedder(api_key=api_key, model=embed_model)
    retriever = Retriever(index=index, chunks=chunks, embedder=embedder)
    generator = AnswerGenerator(api_key=api_key, model=chat_model)

    return retriever, generator


def source_card(chunk: dict, score: float, idx: int) -> str:
    src = chunk.get("source", "")
    section = chunk.get("section", "")
    item = chunk.get("item_name", "")
    t = chunk.get("type", "")
    text_preview = (chunk.get("text", "")[:260]).replace("\n", " ").strip()

    title = f"[{idx}] {section}" + (f" → {item}" if item else "")
    badges_html = f"""
    <div class="badges">
      <span class="badge badge-strong">{src}</span>
      <span class="badge">{t}</span>
      <span class="badge muted">score: {score:.3f}</span>
    </div>
    """

    return f"""
    <div class="card">
      <div class="card-title">{title}</div>
      {badges_html}
      <div class="card-text">{text_preview}...</div>
    </div>
    """


def main():
    st.set_page_config(page_title="RestoRAG ", page_icon="🍜", layout="wide")
    inject_css()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

    st.markdown("## RestoRAG ")
    st.markdown(
        '<span class="muted">RAG na podstawie <b>menu</b> i <b>informacji o restauracji</b></span>',
        unsafe_allow_html=True,
    )
    st.write("")

    col_chat, col_sources = st.columns([2.2, 1.0], gap="large")

    with col_sources:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Źródła (ostatnia odpowiedź)")

        if st.session_state.last_sources:
            for i, s in enumerate(st.session_state.last_sources, start=1):
                st.markdown(source_card(s["chunk"], s["score"], i), unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="card"><div class="card-text muted">Zadaj pytanie, a tutaj pokażę źródła użyte do odpowiedzi.</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_chat:
        st.markdown("### Chat")

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        question = st.chat_input("Zadaj pytanie o menu, ceny, składniki, godziny, dostawę...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            retriever, generator = load_rag()

            results = retriever.search(query=question, top_k=5)
            answer = generator.answer(question=question, results=results)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_sources = results

            with st.chat_message("assistant"):
                st.markdown(answer)



if __name__ == "__main__":
    main()
