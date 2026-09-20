from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from style_demo.db import Repository, initialize_database
from style_demo.engine import Result, StyleDemo
from style_demo.emotion_classifier import MultiMotions28Classifier
from style_demo.local_llm import ContextEchoLLM, LocalTransformersLLM


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "db.sqlite3"
MODEL_PATH = ROOT / "models" / "qwen2.5-0.5b-instruct"
CLASSIFIER_PATH = ROOT / "models" / "multi-motions-28"
DEFAULT_MESSAGE = "Я боюсь опоздать на рейс. Что мне делать?"


def vector_text(vector: dict[str, float], limit: int = 8) -> str:
    ranked = sorted(vector.items(), key=lambda item: item[1], reverse=True)
    shown = ranked[:limit]
    suffix = f", … ({len(vector)} dims)" if len(vector) > limit else ""
    return ", ".join(f"{key}={value:.2f}" for key, value in shown) + suffix


def print_messages(title: str, messages: list[dict[str, str]]) -> None:
    print(f"\n{title}")
    if not messages:
        print("  (не добавляется)")
        return
    for item in messages:
        print(f"  {item['role'].upper()}: {item['content']}")


def print_result(character_name: str, result: Result, show_context: bool) -> None:
    trace = result.trace
    print("\n" + "=" * 72)
    print(f"Персонаж: {character_name} | стиль: {trace.style_name}")
    print(f"U_t, эмоции пользователя:       {vector_text(trace.user_state)}")
    print(f"E_t, состояние агента:          {vector_text(trace.agent_state)}")
    print(f"S_base без памяти:              {vector_text(trace.base_communication_state)}")
    print(f"A_t, ассоциативная память:      {vector_text(trace.associative_state)}")
    print(f"S_t, состояние выбора стиля:    {vector_text(trace.communication_state)}")
    print(f"R_t = S_t ⊙ W_character:        {vector_text(trace.reaction_scores)}")

    if trace.matched_entities:
        print("Найденные сущности:")
        for item in trace.matched_entities:
            forms = ", ".join(item.surface_forms)
            vector = trace.entity_associations.get(item.canonical, {})
            print(f"  {forms} -> {item.canonical}: {vector_text(vector)}")
    else:
        print("Найденные сущности: (нет)")

    print(f"Выбрана эмоция: {trace.selected_emotion}")
    print_messages("Коммуникационный микродиалог:", trace.style_microdialogue)
    print_messages("Мотивационный микродиалог:", trace.motivation_microdialogue)
    if show_context:
        print_messages("Полный контекст, переданный локальной LLM:", trace.model_messages)
    print(f"\nОТВЕТ: {result.answer}")

    if trace.memory_updates:
        print("\nОбновление ассоциативной памяти:")
        for update in trace.memory_updates:
            print(f"  {update.entity} | встреч: {update.encounters}")
            print(f"    было:  {vector_text(update.before)}")
            print(f"    стало: {vector_text(update.after)}")


def make_model(backend: str):
    if backend == "echo":
        return ContextEchoLLM()
    return LocalTransformersLLM(MODEL_PATH)


def make_classifier() -> MultiMotions28Classifier:
    return MultiMotions28Classifier(CLASSIFIER_PATH)


def make_demo(
    repository: Repository,
    model,
    classifier,
    character_name: str,
    memory_beta: float,
    memory_eta: float,
    memory_gamma: float,
) -> StyleDemo:
    return StyleDemo(
        repository,
        model,
        classifier,
        repository.get_character(character_name),
        memory_beta=memory_beta,
        memory_eta=memory_eta,
        memory_gamma=memory_gamma,
    )


def run_compare(
    repository: Repository,
    model,
    classifier,
    text: str,
    show_context: bool,
    memory_beta: float,
    memory_eta: float,
    memory_gamma: float,
) -> None:
    # This is a product-level comparison: baseline and the three named
    # characters deliberately have different system prompts, alpha values,
    # styles, character weights, and motivation levels. It is not a causal A/B
    # test that changes only the presence of a microdialogue.
    baseline_character = replace(
        repository.get_character("Мира"),
        system_prompt=(
            "Ты — компетентный помощник. Сохраняй факты, не выдумывай детали "
            "и отвечай на языке текущего сообщения пользователя."
        ),
        motivation_level=0,
    )
    baseline = StyleDemo(
        repository,
        model,
        classifier,
        baseline_character,
        memory_beta=memory_beta,
        memory_eta=memory_eta,
        memory_gamma=memory_gamma,
    ).respond(
        text,
        use_style=False,
        use_memory=False,
        learn_memory=False,
        keep_history=False,
    )
    print_result("Без управляющих микродиалогов", baseline, show_context)
    for character in repository.list_characters():
        result = StyleDemo(
            repository,
            model,
            classifier,
            character,
            memory_beta=memory_beta,
            memory_eta=memory_eta,
            memory_gamma=memory_gamma,
        ).respond(
            text,
            use_memory=False,
            learn_memory=False,
            keep_history=False,
        )
        print_result(character.name, result, show_context)


def interactive(
    repository: Repository,
    model,
    classifier,
    character_name: str,
    motivation: int | None,
    show_context: bool,
    memory_beta: float,
    memory_eta: float,
    memory_gamma: float,
) -> None:
    demo = make_demo(
        repository,
        model,
        classifier,
        character_name,
        memory_beta,
        memory_eta,
        memory_gamma,
    )
    print(f"Персонаж {demo.character.name}. Введите сообщение; /exit — завершить.")
    print("Ассоциативная память включена и сохраняется в db.sqlite3.")
    while True:
        try:
            text = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in {"/exit", "exit", "выход"}:
            return
        if text:
            print_result(
                demo.character.name,
                demo.respond(text, motivation_level=motivation),
                show_context,
            )


def print_memory(repository: Repository, character_name: str) -> None:
    character = repository.get_character(character_name)
    rows = repository.list_associations(character.id)
    print(f"Ассоциативная память персонажа {character.name}:")
    if not rows:
        print("  (пока пуста)")
        return
    for canonical, association in rows:
        print(
            f"  {canonical:12s} encounters={association.encounters:3d} "
            f"{vector_text(association.vector)}"
        )


def run_memory_demo(
    repository: Repository,
    model,
    classifier,
    character_name: str,
    show_context: bool,
    memory_beta: float,
    memory_eta: float,
    memory_gamma: float,
) -> None:
    """Small reproducible scenario that makes the memory effect visible."""
    character = repository.get_character(character_name)
    repository.clear_associations(character.id)
    demo = StyleDemo(
        repository,
        model,
        classifier,
        character,
        memory_beta=memory_beta,
        memory_eta=memory_eta,
        memory_gamma=memory_gamma,
    )

    training_turns = [
        "Я очень боюсь самолетов, мне страшно летать.",
        "Самолетами я летать боюсь, каждый рейс вызывает тревогу.",
        "Перед рейсом в аэропорту я снова сильно волнуюсь из-за самолета.",
    ]
    print("\n### ЭТАП 1. Формируем эмоциональную ассоциацию с сущностью 'самолет' ###")
    for text in training_turns:
        result = demo.respond(text, keep_history=False, learn_memory=True)
        print_result(character.name, result, show_context=False)

    print("\n### ПАМЯТЬ ПОСЛЕ ОБУЧАЮЩИХ ПОВТОРОВ ###")
    print_memory(repository, character_name)

    print("\n### ЭТАП 2. Нейтральная фраза с той же сущностью ###")
    neutral = "Завтра летим на самолете в Париж."
    result = demo.respond(
        neutral,
        keep_history=False,
        learn_memory=False,
    )
    print_result(character.name, result, show_context)
    print(
        "\nСравните S_base и S_t: A_t добавляет ранее накопленный "
        "эмоциональный профиль сущности к текущему состоянию выбора стиля."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Локальная демонстрация коммуникационных стилей и ассоциативной эмоциональной памяти"
    )
    parser.add_argument("--character", default="Мира", help="Мира, Алекс или Ирис")
    parser.add_argument("--message", help="одно сообщение; без него запускается диалог")
    parser.add_argument("--motivation", type=int, choices=(-1, 0, 1, 2))
    parser.add_argument(
        "--compare",
        action="store_true",
        help="сравнить baseline и три полные конфигурации персонажей; это не чистый A/B-тест",
    )
    parser.add_argument("--show-context", action="store_true", help="показать весь контекст для LLM")
    parser.add_argument("--show-memory", action="store_true", help="показать накопленные ассоциации персонажа и завершить")
    parser.add_argument("--reset-memory", action="store_true", help="очистить ассоциативную память выбранного персонажа")
    parser.add_argument("--memory-demo", action="store_true", help="запустить сценарий обучения эмоциональной ассоциации с самолетом и повторного упоминания")
    parser.add_argument("--memory-beta", type=float, default=0.35, help="beta: влияние A_t на S_t")
    parser.add_argument("--memory-eta", type=float, default=0.20, help="eta: скорость обновления A(entity)")
    parser.add_argument("--memory-gamma", type=float, default=1.0, help="gamma: доля памяти, сохраняемая после обновления")
    parser.add_argument("--no-memory", action="store_true", help="отключить чтение и обучение ассоциативной памяти")
    parser.add_argument("--backend", choices=("local", "echo"), default="local", help="echo предназначен только для тестирования конвейера")
    args = parser.parse_args()

    initialize_database(DB_PATH)
    repository = Repository(DB_PATH)

    if args.reset_memory:
        character = repository.get_character(args.character)
        repository.clear_associations(character.id)
        print(f"Ассоциативная память персонажа {character.name} очищена.")
        if not (args.message or args.compare or args.memory_demo or args.show_memory):
            return

    if args.show_memory:
        print_memory(repository, args.character)
        if not (args.message or args.compare or args.memory_demo):
            return

    try:
        model = make_model(args.backend)
        classifier = make_classifier()
    except FileNotFoundError as error:
        parser.error(str(error))

    if args.memory_demo:
        run_memory_demo(
            repository,
            model,
            classifier,
            args.character,
            args.show_context,
            args.memory_beta,
            args.memory_eta,
            args.memory_gamma,
        )
    elif args.compare:
        run_compare(
            repository,
            model,
            classifier,
            args.message or DEFAULT_MESSAGE,
            args.show_context,
            args.memory_beta,
            args.memory_eta,
            args.memory_gamma,
        )
    elif args.message:
        demo = make_demo(
            repository,
            model,
            classifier,
            args.character,
            args.memory_beta,
            args.memory_eta,
            args.memory_gamma,
        )
        result = demo.respond(
            args.message,
            motivation_level=args.motivation,
            use_memory=not args.no_memory,
            # A one-shot CLI call still represents a real interaction, so the
            # association is persisted even though no chat history is retained.
            learn_memory=not args.no_memory,
            keep_history=False,
        )
        print_result(demo.character.name, result, args.show_context)
    else:
        interactive(
            repository,
            model,
            classifier,
            args.character,
            args.motivation,
            args.show_context,
            args.memory_beta,
            args.memory_eta,
            args.memory_gamma,
        )


if __name__ == "__main__":
    main()
