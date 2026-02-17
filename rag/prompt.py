from __future__ import annotations
from typing import List, Dict, Any


def build_context(results: List[Dict[str, Any]]) -> str:
    """
    Skleja kontekst z wyników retrieval.
    Każdy chunk dostaje numer [1], [2]...
    """
    parts = []
    for i, r in enumerate(results, start=1):
        ch = r["chunk"]
        header = f"[{i}] source={ch.get('source','')} | section={ch.get('section','')}"
        if ch.get("item_name"):
            header += f" | item={ch.get('item_name')}"
        parts.append(header)
        parts.append(ch.get("text", "").strip())
        parts.append("")
    return "\n".join(parts).strip()


def build_system_rules() -> str:
    return (
        "Jesteś asystentem restauracji. Odpowiadaj po polsku.\n"
        "Odpowiadaj WYŁĄCZNIE na podstawie podanego KONTEKSTU.\n"
        "Jeśli w kontekście nie ma informacji, powiedz wprost: "
        "\"Nie mam informacji w danych/menu.\" i nie zgaduj.\n"
        "Odpowiedź ma być krótka i konkretna.\n"
        "Na końcu dodaj linię: Źródła: [1], [2] (numery tych fragmentów, z których korzystałeś).\n"
    )


def build_user_prompt(question: str, context: str) -> str:
    return f"KONTEKST:\n{context}\n\nPYTANIE:\n{question}\n"
