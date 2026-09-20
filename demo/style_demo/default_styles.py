from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .emotion import EMOTIONS


_DIALOGUE_DIR = Path(__file__).with_name("dialogues")
_CHARACTER_FILES = {
    "Мира": "mira.json",
    "Алекс": "alex.json",
    "Ирис": "iris.json",
}


@lru_cache(maxsize=1)
def load_default_dialogues() -> dict[str, dict[str, Any]]:
    """Load authored character dialogue data from JSON files.

    The Python module contains only the loader and structural validation. The
    actual USER → ASSISTANT texts live in ``style_demo/dialogues/*.json`` and
    are copied into SQLite during database initialization.
    """
    result: dict[str, dict[str, Any]] = {}
    for character_name, filename in _CHARACTER_FILES.items():
        path = _DIALOGUE_DIR / filename
        with path.open("r", encoding="utf-8") as handle:
            result[character_name] = json.load(handle)
    validate_default_dialogues(result)
    return result


def _pair_to_messages(pair: dict[str, str]) -> list[dict[str, str]]:
    if set(pair) != {"user", "assistant"}:
        raise ValueError("Each dialogue pair must contain exactly user and assistant text")
    return [
        {"role": "user", "content": str(pair["user"])},
        {"role": "assistant", "content": str(pair["assistant"])},
    ]


def build_default_style_levels(
    character_name: str, emotion: str
) -> dict[str, list[dict[str, str]]]:
    """Return the authored three-level dialogue for one character/emotion."""
    dialogues = load_default_dialogues()
    if character_name not in dialogues:
        raise KeyError(f"Unknown character: {character_name}")
    if emotion not in EMOTIONS:
        raise KeyError(f"Unsupported emotion field: {emotion}")
    return {
        str(level): _pair_to_messages(pair)
        for level, pair in enumerate(dialogues[character_name]["communication"][emotion])
    }


def get_default_motivation(
    character_name: str,
) -> dict[int, list[dict[str, str]]]:
    """Return authored motivation pairs for one character."""
    dialogues = load_default_dialogues()
    if character_name not in dialogues:
        raise KeyError(f"Unknown character: {character_name}")
    motivation = dialogues[character_name]["motivation"]
    return {
        int(level): ([] if pair is None else _pair_to_messages(pair))
        for level, pair in motivation.items()
    }


def style_name(character_name: str) -> str:
    dialogues = load_default_dialogues()
    try:
        return str(dialogues[character_name]["style"])
    except KeyError as error:
        raise KeyError(f"Unknown character: {character_name}") from error


def style_description(character_name: str) -> str:
    dialogues = load_default_dialogues()
    try:
        return str(dialogues[character_name]["description"])
    except KeyError as error:
        raise KeyError(f"Unknown character: {character_name}") from error


def validate_default_dialogues(
    dialogues: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Validate all authored dialogue data before it is written to SQLite."""
    source = dialogues if dialogues is not None else load_default_dialogues()
    expected_characters = set(_CHARACTER_FILES)
    if set(source) != expected_characters:
        raise RuntimeError("Dialogue data must contain exactly the three demo characters")

    for character_name, character in source.items():
        communication = character.get("communication")
        if not isinstance(communication, dict) or set(communication) != set(EMOTIONS):
            raise RuntimeError(
                f"{character_name}: communication data must cover all 28 emotions"
            )
        for emotion in EMOTIONS:
            levels = communication[emotion]
            if not isinstance(levels, list) or len(levels) != 3:
                raise RuntimeError(
                    f"{character_name}/{emotion} must have three intensity pairs"
                )
            for pair in levels:
                if not isinstance(pair, dict) or set(pair) != {"user", "assistant"}:
                    raise RuntimeError(
                        f"{character_name}/{emotion} must contain user and assistant text"
                    )
                if not pair["user"].strip() or not pair["assistant"].strip():
                    raise RuntimeError(f"{character_name}/{emotion} contains empty text")

        motivation = character.get("motivation")
        if not isinstance(motivation, dict) or set(motivation) != {"-1", "0", "1", "2"}:
            raise RuntimeError(
                f"{character_name}: motivation data must cover levels -1, 0, 1, 2"
            )
        if motivation["0"] is not None:
            raise RuntimeError(f"{character_name}: motivation level 0 must be empty")
        for level in ("-1", "1", "2"):
            pair = motivation[level]
            if not isinstance(pair, dict) or set(pair) != {"user", "assistant"}:
                raise RuntimeError(
                    f"{character_name}/motivation/{level} must be a full pair"
                )
            if not pair["user"].strip() or not pair["assistant"].strip():
                raise RuntimeError(f"{character_name}/motivation/{level} contains empty text")

