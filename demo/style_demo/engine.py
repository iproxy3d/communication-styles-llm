from __future__ import annotations

from dataclasses import dataclass

from .association import EntityExtractor, EntityMatch, aggregate_memory
from .db import Character, Repository
from .emotion import (
    apply_character_weights,
    mix_state,
    mix_state_with_memory,
    update_toy_agent_state,
)
from .emotion_classifier import EmotionClassifier
from .local_llm import ChatModel


@dataclass
class MemoryUpdate:
    entity: str
    before: dict[str, float]
    after: dict[str, float]
    encounters: int


@dataclass
class Trace:
    user_state: dict[str, float]
    agent_state: dict[str, float]
    base_communication_state: dict[str, float]
    associative_state: dict[str, float]
    communication_state: dict[str, float]
    reaction_scores: dict[str, float]
    matched_entities: list[EntityMatch]
    entity_associations: dict[str, dict[str, float]]
    memory_updates: list[MemoryUpdate]
    selected_emotion: str
    style_name: str
    style_microdialogue: list[dict[str, str]]
    motivation_microdialogue: list[dict[str, str]]
    model_messages: list[dict[str, str]]


@dataclass
class Result:
    answer: str
    trace: Trace


class StyleDemo:
    def __init__(
        self,
        repository: Repository,
        model: ChatModel,
        emotion_classifier: EmotionClassifier,
        character: Character,
        *,
        memory_beta: float = 0.35,
        memory_eta: float = 0.20,
        memory_gamma: float = 1.0,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        if memory_beta < 0.0:
            raise ValueError("memory_beta must be >= 0")
        if not 0.0 < memory_eta <= 1.0:
            raise ValueError("memory_eta must be in (0, 1]")
        if not 0.0 < memory_gamma <= 1.0:
            raise ValueError("memory_gamma must be in (0, 1]")
        self.repository = repository
        self.model = model
        self.emotion_classifier = emotion_classifier
        self.character = character
        self.memory_beta = memory_beta
        self.memory_eta = memory_eta
        self.memory_gamma = memory_gamma
        self.entity_extractor = entity_extractor or EntityExtractor(repository)
        self.real_history: list[dict[str, str]] = []
        self.agent_state = dict(character.initial_state)

    def respond(
        self,
        user_text: str,
        *,
        use_style: bool = True,
        use_memory: bool = True,
        learn_memory: bool | None = None,
        motivation_level: int | None = None,
        keep_history: bool = True,
    ) -> Result:
        if learn_memory is None:
            learn_memory = keep_history

        user_state = self.emotion_classifier.predict(user_text)
        agent_state = dict(self.agent_state)
        base_communication_state = mix_state(
            user_state, agent_state, self.character.alpha
        )

        matched_entities = self.entity_extractor.extract(user_text) if use_memory else []
        associative_state, entity_associations = aggregate_memory(
            self.repository,
            self.character.id,
            matched_entities,
        )
        communication_state = (
            mix_state_with_memory(
                user_state,
                agent_state,
                associative_state,
                self.character.alpha,
                self.memory_beta,
            )
            if use_memory
            else base_communication_state
        )

        reaction_scores = apply_character_weights(
            communication_state, self.character.character_weights
        )
        available_emotions = [
            emotion
            for emotion in reaction_scores
            if self.character.character_weights.get(emotion, 1.0) > 0.0
        ]
        if not available_emotions:
            raise ValueError("W_character must leave at least one reaction available")
        selected_emotion = max(
            available_emotions, key=lambda emotion: reaction_scores[emotion]
        )
        style_name = (
            self.repository.get_style_name(self.character.style_id)
            if use_style
            else "disabled"
        )
        style_microdialogue = (
            self.repository.get_style_microdialogue(
                self.character.style_id, selected_emotion, self.character.intensity
            )
            if use_style
            else []
        )
        level = (
            self.character.motivation_level
            if motivation_level is None
            else motivation_level
        )
        motivation_microdialogue = (
            self.repository.get_motivation_microdialogue(self.character.id, level)
            if use_style
            else []
        )

        model_messages = [
            {"role": "system", "content": self.character.system_prompt},
            *self.real_history,
            *style_microdialogue,
            *motivation_microdialogue,
            {"role": "user", "content": user_text},
        ]
        answer = self.model.generate(model_messages)

        memory_updates: list[MemoryUpdate] = []
        if use_memory and learn_memory:
            # The article updates A(x) from the agent's current recurrent state
            # E_t, not from the mixed selector state S_t. Hidden microdialogues
            # do not enter this update.
            pending_updates: list[tuple[EntityMatch, dict[str, float], int]] = []
            for match in matched_entities:
                current = self.repository.get_association(
                    self.character.id, match.entity_id
                )
                before = (
                    dict(current.vector)
                    if current is not None
                    else {name: 0.0 for name in communication_state}
                )
                updated = self.repository.update_association(
                    self.character.id,
                    match.entity_id,
                    agent_state,
                    self.memory_eta,
                )
                pending_updates.append((match, before, updated.encounters))

            # A <- gamma * A is a separate forgetting step applied to every
            # stored association, including entities not mentioned this turn.
            self.repository.decay_associations(
                self.character.id,
                self.memory_gamma,
            )
            for match, before, encounters in pending_updates:
                updated = self.repository.get_association(
                    self.character.id, match.entity_id
                )
                if updated is None:  # pragma: no cover - guarded by the write above.
                    raise RuntimeError("association disappeared after update")
                memory_updates.append(
                    MemoryUpdate(
                        entity=match.canonical,
                        before=before,
                        after=dict(updated.vector),
                        encounters=encounters,
                    )
                )

        trace = Trace(
            user_state=user_state,
            agent_state=dict(self.agent_state),
            base_communication_state=base_communication_state,
            associative_state=associative_state,
            communication_state=communication_state,
            reaction_scores=reaction_scores,
            matched_entities=matched_entities,
            entity_associations=entity_associations,
            memory_updates=memory_updates,
            selected_emotion=selected_emotion,
            style_name=style_name,
            style_microdialogue=style_microdialogue,
            motivation_microdialogue=motivation_microdialogue,
            model_messages=model_messages,
        )
        if keep_history:
            # Only real turns are persisted. Hidden microdialogues stay local to this call.
            self.real_history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": answer},
                ]
            )
        if keep_history or learn_memory:
            self.agent_state = update_toy_agent_state(agent_state, user_state)
        return Result(answer=answer, trace=trace)
