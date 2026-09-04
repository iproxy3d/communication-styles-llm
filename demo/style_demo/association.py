from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .db import MemoryEntity, Repository
from .emotion import EMOTIONS

try:  # Optional at import time so the demo/tests still explain the fallback path.
    from pymorphy3 import MorphAnalyzer  # type: ignore
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    MorphAnalyzer = None  # type: ignore[assignment]

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё-]+", re.UNICODE)


@dataclass(frozen=True)
class EntityMatch:
    entity_id: int
    canonical: str
    surface_forms: tuple[str, ...]


class EntityExtractor:
    """Transparent dictionary-backed entity extractor for the demo.

    The dictionary contains canonical entities, while the input is normalized
    morphologically. If pymorphy3 is installed, Russian inflected forms such as
    ``Иванову`` and ``самолётами`` become ``иванов`` and ``самолёт``. A tiny
    fallback keeps the demo runnable even without the optional morphology
    package, but production use should install dependencies from requirements.txt.
    """

    def __init__(self, repository: Repository) -> None:
        self.repository = repository
        self._morph = MorphAnalyzer() if MorphAnalyzer is not None else None

    @staticmethod
    def _clean(value: str) -> str:
        return value.lower().replace("ё", "е").strip(" -_.,:;!?()[]{}\"'«»")

    def lemma(self, token: str) -> str:
        cleaned = self._clean(token)
        if not cleaned:
            return ""
        if self._morph is not None:
            parsed = self._morph.parse(cleaned)
            if parsed:
                return self._clean(parsed[0].normal_form)
        return self._fallback_lemma(cleaned)

    @staticmethod
    def _fallback_lemma(token: str) -> str:
        # This is deliberately small. It is not a replacement for morphology;
        # it only keeps the educational demo understandable without a model.
        special = {
            "иванову": "иванов",
            "иванова": "иванов",
            "ивановым": "иванов",
            "иванове": "иванов",
            "самолетами": "самолет",
            "самолетом": "самолет",
            "самолета": "самолет",
            "самолетов": "самолет",
            "самолету": "самолет",
            "самолеты": "самолет",
            "самолете": "самолет",
            "рейсом": "рейс",
            "рейса": "рейс",
            "рейсе": "рейс",
            "рейсы": "рейс",
            "аэропорту": "аэропорт",
            "аэропорта": "аэропорт",
            "аэропортом": "аэропорт",
        }
        return special.get(token, token)

    def extract(self, text: str) -> list[EntityMatch]:
        entities = self.repository.list_memory_entities()
        if not entities:
            return []

        by_key: dict[str, MemoryEntity] = {}
        for entity in entities:
            keys = {entity.canonical, *entity.aliases}
            for key in keys:
                clean = self._clean(key)
                if clean:
                    by_key[clean] = entity
                    by_key[self.lemma(clean)] = entity

        found: dict[int, list[str]] = {}
        for surface in _TOKEN_RE.findall(text):
            clean = self._clean(surface)
            lemma = self.lemma(surface)
            entity = by_key.get(lemma) or by_key.get(clean)
            if entity is not None:
                found.setdefault(entity.id, []).append(surface)

        result: list[EntityMatch] = []
        entity_by_id = {entity.id: entity for entity in entities}
        for entity_id, surfaces in found.items():
            entity = entity_by_id[entity_id]
            result.append(
                EntityMatch(
                    entity_id=entity.id,
                    canonical=entity.canonical,
                    surface_forms=tuple(surfaces),
                )
            )
        return result


def aggregate_memory(
    repository: Repository,
    character_id: int,
    matches: Iterable[EntityMatch],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return A_t and per-entity vectors.

    Stored vectors are intentionally *not* normalized. A newly learned
    association therefore starts weak and becomes stronger after repetitions.
    Multiple matching entities are averaged so a long message does not become
    more emotional merely because it contains more nouns.
    """

    matched = list(matches)
    details: dict[str, dict[str, float]] = {}
    if not matched:
        return ({emotion: 0.0 for emotion in EMOTIONS}, details)

    total = {emotion: 0.0 for emotion in EMOTIONS}
    contributing = 0
    for match in matched:
        association = repository.get_association(character_id, match.entity_id)
        if association is None:
            details[match.canonical] = {emotion: 0.0 for emotion in EMOTIONS}
            continue
        vector = {emotion: float(association.vector.get(emotion, 0.0)) for emotion in EMOTIONS}
        details[match.canonical] = vector
        for emotion in EMOTIONS:
            total[emotion] += vector[emotion]
        contributing += 1

    # Missing associations still appear in the trace but do not dilute already
    # learned associations. They will be initialized after this interaction.
    if contributing == 0:
        return ({emotion: 0.0 for emotion in EMOTIONS}, details)
    return ({emotion: total[emotion] / contributing for emotion in EMOTIONS}, details)
