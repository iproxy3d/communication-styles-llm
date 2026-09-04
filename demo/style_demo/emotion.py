from __future__ import annotations

import re
from collections.abc import Mapping

EMOTIONS = ("neutral", "fear", "anger", "annoyance", "joy")

_LEXICON: dict[str, tuple[str, ...]] = {
    "fear": ("бою", "страш", "опас", "тревог", "волную", "паник"),
    "anger": ("зл", "бесит", "ненавиж", "ярост", "тупиц", "идиот"),
    "annoyance": ("раздраж", "надоел", "опять", "сколько можно", "достал"),
    "joy": ("рад", "счаст", "отлично", "здорово", "ура", "спасибо"),
}


def normalize(vector: Mapping[str, float]) -> dict[str, float]:
    values = {name: max(0.0, float(vector.get(name, 0.0))) for name in EMOTIONS}
    total = sum(values.values())
    if total <= 0:
        return {name: 1.0 if name == "neutral" else 0.0 for name in EMOTIONS}
    return {name: value / total for name, value in values.items()}


def detect_user_emotions(text: str) -> dict[str, float]:
    """Tiny deterministic detector for the demo, not a production classifier."""
    lowered = re.sub(r"\s+", " ", text.lower())
    scores = {name: 0.0 for name in EMOTIONS}
    for emotion, stems in _LEXICON.items():
        scores[emotion] = sum(1.0 for stem in stems if stem in lowered)
    if sum(scores.values()) == 0:
        scores["neutral"] = 1.0
    else:
        scores["neutral"] = 0.08
    return normalize(scores)


def mix_state(
    user_state: Mapping[str, float],
    agent_state: Mapping[str, float],
    alpha: float,
) -> dict[str, float]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    mixed = {
        name: alpha * float(user_state.get(name, 0.0))
        + (1.0 - alpha) * float(agent_state.get(name, 0.0))
        for name in EMOTIONS
    }
    return normalize(mixed)


def update_toy_agent_state(
    agent_state: Mapping[str, float],
    user_state: Mapping[str, float],
    inertia: float = 0.8,
) -> dict[str, float]:
    """Toy stand-in for the article's external recurrent emotional subsystem."""
    if not 0.0 <= inertia <= 1.0:
        raise ValueError("inertia must be in [0, 1]")
    updated = {
        name: inertia * float(agent_state.get(name, 0.0))
        + (1.0 - inertia) * float(user_state.get(name, 0.0))
        for name in EMOTIONS
    }
    return normalize(updated)

