from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any
import json

import faiss
import numpy as np

from rag.chunker import Chunk


class FaissIndexStore:
    """
    - FAISS index z wektorów
    - zapis indexu do pliku
    - zapis chunków+metadanych do jsonl
    - wczytanie i wyszukiwanie top-k
    """

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss.index"
        self.chunks_path = self.index_dir / "chunks.jsonl"

    def build_and_save(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """
        Buduje FAISS index i zapisuje go na dysk.
        """
        if len(vectors) == 0:
            raise ValueError("Brak wektorów do zapisania.")
        if len(vectors) != len(chunks):
            raise ValueError("Liczba wektorów != liczba chunków.")

        # FAISS wymaga macierzy float32 o rozmiarze (n, dim)
        mat = np.array(vectors, dtype="float32")
        dim = mat.shape[1]

        # IndexFlatIP = proste wyszukiwanie po iloczynie skalarnym
        index = faiss.IndexFlatIP(dim)

        # Normalizacja: wtedy iloczyn skalarny ~ cosine similarity
        faiss.normalize_L2(mat)

        index.add(mat)

        # Zapis indexu
        faiss.write_index(index, str(self.index_path))

        # Zapis chunków jako jsonl
        with self.chunks_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                rec: Dict[str, Any] = {"text": c.text, **c.meta}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load(self) -> tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Wczytuje index i chunki z dysku.
        """
        if not self.index_path.exists() or not self.chunks_path.exists():
            raise FileNotFoundError("Brak plików index/faiss.index lub index/chunks.jsonl. Uruchom build_index.")

        index = faiss.read_index(str(self.index_path))

        chunks: List[Dict[str, Any]] = []
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

        return index, chunks
