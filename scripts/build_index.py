from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from rag.chunker import read_md, chunk_menu, chunk_info
from rag.embeddings import OpenAIEmbedder
from rag.index_store import FaissIndexStore


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    if not api_key:
        raise RuntimeError("Brak OPENAI_API_KEY w .env")

    base_dir = Path(__file__).resolve().parent.parent

    menu_md = read_md(base_dir / "data" / "menu.md")
    info_md = read_md(base_dir / "data" / "restaurant_info.md")

    menu_chunks = chunk_menu(menu_md)
    info_chunks = chunk_info(info_md)

    chunks = menu_chunks + info_chunks
    print(f"Chunków do indeksowania: {len(chunks)}")

    texts = [c.text for c in chunks]

    embedder = OpenAIEmbedder(api_key=api_key, model=embed_model)
    vectors = embedder.embed_texts(texts)

    store = FaissIndexStore(index_dir=base_dir / "index")
    store.build_and_save(vectors=vectors, chunks=chunks)

    print("OK. Zbudowano i zapisano index do folderu /index.")


if __name__ == "__main__":
    main()
