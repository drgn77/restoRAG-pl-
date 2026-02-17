from __future__ import annotations

from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

from rag.embeddings import OpenAIEmbedder


class Retriever:
    """
    - robi embedding pytania
    - FAISS znajduje top-k podobnych wektorów
    - zwraca odpowiadające chunki + score
    """

    def __init__(self, index: faiss.Index, chunks: List[Dict[str, Any]], embedder: OpenAIEmbedder):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 1) embedding
        q_vec = self.embedder.embed_texts([query])[0]
        q = np.array([q_vec], dtype="float32")

        # 2) normalizacja
        faiss.normalize_L2(q)

        # 3) FAISS zwraca: distances (scores) i indices (ID chunków)
        scores, indices = self.index.search(q, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            payload = self.chunks[int(idx)]
            results.append(
                {
                    "score": float(score),
                    "chunk": payload,
                }
            )
        return results
