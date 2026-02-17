from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import re



@dataclass
class Chunk:
    text: str            # treść chunku -> embed
    meta: Dict[str, str] # metadane


# Regexy do wykrywania nagłówków w markdown
H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
H3 = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


def read_md(path: Path) -> str:
    """Wczytuje plik markdown jako tekst."""
    return path.read_text(encoding="utf-8").strip()


def split_by_h2(md: str) -> List[tuple[str, str]]:
    """
    Dzieli markdown na bloki wg ##.
    Zwraca listę: (nazwa_sekcji, tekst_sekcji).
    """
    matches = list(H2.finditer(md))
    if not matches:
        return []

    blocks: List[tuple[str, str]] = []
    for i, m in enumerate(matches):
        section = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        blocks.append((section, md[start:end].strip()))
    return blocks


def chunk_menu(md: str) -> List[Chunk]:
    """
    MENU:
    - ## = kategoria
    - ### = konkretne danie -> osobny chunk
    - jeśli w sekcji nie ma ### -> jeden chunk
    """
    chunks: List[Chunk] = []

    for section, content in split_by_h2(md):
        h3_matches = list(H3.finditer(content))

        # Sekcja bez ###
        if not h3_matches:
            text = f"## {section}\n\n{content}".strip()
            chunks.append(
                Chunk(
                    text=text,
                    meta={
                        "source": "menu",
                        "section": section,
                        "item_name": "",
                        "type": "addon" if ("Dodatki" in section or "Sosy" in section) else "info",
                    },
                )
            )
            continue

        # Sekcja z ### -> każda pozycja osobno
        for j, h3 in enumerate(h3_matches):
            item_name = h3.group(1).strip()
            start = h3.start()
            end = h3_matches[j + 1].start() if j + 1 < len(h3_matches) else len(content)
            item_block = content[start:end].strip()

            # Do chunku dokładamy nagłowek
            text = f"## {section}\n\n{item_block}".strip()

            item_type = "set" if "Zestaw" in item_name or "deska" in item_name.lower() else "dish"

            chunks.append(
                Chunk(
                    text=text,
                    meta={
                        "source": "menu",
                        "section": section,
                        "item_name": item_name,
                        "type": item_type,
                    },
                )
            )

    return chunks


def chunk_info(md: str) -> List[Chunk]:
    """
    INFO O RESTAURACJI:
    - sekcja ## = 1 chunk
    """
    chunks: List[Chunk] = []

    for section, content in split_by_h2(md):
        text = f"## {section}\n\n{content}".strip()
        chunks.append(
            Chunk(
                text=text,
                meta={
                    "source": "info",
                    "section": section,
                    "item_name": "",
                    "type": "info",
                },
            )
        )

    return chunks
