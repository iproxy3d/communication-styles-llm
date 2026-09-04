from __future__ import annotations

from dataclasses import dataclass

from .db import Character, Repository
from .emotion import detect_user_emotions, mix_state, update_toy_agent_state
from .local_llm import ChatModel


@dataclass
class Trace:
    user_state: dict[str, float]
    agent_state: dict[str, float]
    communication_state: dict[str, float]
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
    def __init__(self, repository: Repository, model: ChatModel, character: Character) -> None:
        self.repository = repository
        self.model = model
        self.character = character
        self.real_history: list[dict[str, str]] = []
        self.agent_state = dict(character.initial_state)

    def respond(
        self,
        user_text: str,
        *,
        use_style: bool = True,
        motivation_level: int | None = None,
        keep_history: bool = True,
    ) -> Result:
        user_state = detect_user_emotions(user_text)
        communication_state = mix_state(
            user_state, self.agent_state, self.character.alpha
        )
        selected_emotion = max(communication_state, key=communication_state.get)
        style_name = self.repository.get_style_name(self.character.style_id)
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
            self.repository.get_motivation_microdialogue(level) if use_style else []
        )

        model_messages = [
            {"role": "system", "content": self.character.system_prompt},
            *self.real_history,
            *style_microdialogue,
            *motivation_microdialogue,
            {"role": "user", "content": user_text},
        ]
        answer = self.model.generate(model_messages)
        trace = Trace(
            user_state=user_state,
            agent_state=dict(self.agent_state),
            communication_state=communication_state,
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
            self.agent_state = update_toy_agent_state(self.agent_state, user_state)
        return Result(answer=answer, trace=trace)

