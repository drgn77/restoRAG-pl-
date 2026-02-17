from __future__ import annotations

from typing import List, Dict, Any
from openai import OpenAI

from rag.prompt import build_context, build_system_rules, build_user_prompt


class AnswerGenerator:
    """
    Bierze pytanie + wyniki retrieval i generuje odpowiedź.
    """

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def answer(self, question: str, results: List[Dict[str, Any]]) -> str:
        context = build_context(results)
        system_rules = build_system_rules()
        user_prompt = build_user_prompt(question, context)

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.output_text.strip()
