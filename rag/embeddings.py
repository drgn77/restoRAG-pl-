from __future__ import annotations

from typing import List
from openai import OpenAI


class OpenAIEmbedder:

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Zwraca listę wektorów.
        1 wektor -> 1 tekst
        """
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in resp.data]
