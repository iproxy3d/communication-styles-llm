from __future__ import annotations

import argparse
from pathlib import Path

from style_demo.db import Repository, initialize_database
from style_demo.engine import Result, StyleDemo
from style_demo.local_llm import ContextEchoLLM, LocalTransformersLLM


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "db.sqlite3"
MODEL_PATH = ROOT / "models" / "qwen2.5-0.5b-instruct"
DEFAULT_MESSAGE = "Я боюсь опоздать на рейс. Что мне делать?"


def vector_text(vector: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.2f}" for key, value in vector.items())


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
    print(f"U_t, эмоции пользователя: {vector_text(trace.user_state)}")
    print(f"E_t, состояние агента:    {vector_text(trace.agent_state)}")
    print(f"S_t, состояние ответа:    {vector_text(trace.communication_state)}")
    print(f"Выбрана эмоция: {trace.selected_emotion}")
    print_messages("Коммуникационный микродиалог:", trace.style_microdialogue)
    print_messages("Мотивационный микродиалог:", trace.motivation_microdialogue)
    if show_context:
        print_messages("Полный контекст, переданный локальной LLM:", trace.model_messages)
    print(f"\nОТВЕТ: {result.answer}")


def make_model(backend: str):
    if backend == "echo":
        return ContextEchoLLM()
    return LocalTransformersLLM(MODEL_PATH)


def run_compare(repository: Repository, model, text: str, show_context: bool) -> None:
    baseline_character = repository.get_character("Мира")
    baseline = StyleDemo(repository, model, baseline_character).respond(
        text, use_style=False, keep_history=False
    )
    print_result("Без управляющих микродиалогов", baseline, show_context)
    for character in repository.list_characters():
        result = StyleDemo(repository, model, character).respond(
            text, keep_history=False
        )
        print_result(character.name, result, show_context)


def interactive(repository: Repository, model, character_name: str, motivation: int | None, show_context: bool) -> None:
    character = repository.get_character(character_name)
    demo = StyleDemo(repository, model, character)
    print(f"Персонаж {character.name}. Введите сообщение; /exit — завершить.")
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
                character.name,
                demo.respond(text, motivation_level=motivation),
                show_context,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Локальная демонстрация управления стилем LLM через микродиалоги"
    )
    parser.add_argument("--character", default="Мира", help="Мира, Алекс или Ирис")
    parser.add_argument("--message", help="одно сообщение; без него запускается диалог")
    parser.add_argument("--motivation", type=int, choices=(-1, 0, 1, 2))
    parser.add_argument("--compare", action="store_true", help="сравнить ответ без стиля и трёх персонажей")
    parser.add_argument("--show-context", action="store_true", help="показать весь контекст для LLM")
    parser.add_argument("--backend", choices=("local", "echo"), default="local", help="echo предназначен только для тестирования конвейера")
    args = parser.parse_args()

    initialize_database(DB_PATH)
    repository = Repository(DB_PATH)
    try:
        model = make_model(args.backend)
    except FileNotFoundError as error:
        parser.error(str(error))

    if args.compare:
        run_compare(repository, model, args.message or DEFAULT_MESSAGE, args.show_context)
    elif args.message:
        character = repository.get_character(args.character)
        result = StyleDemo(repository, model, character).respond(
            args.message, motivation_level=args.motivation, keep_history=False
        )
        print_result(character.name, result, args.show_context)
    else:
        interactive(repository, model, args.character, args.motivation, args.show_context)


if __name__ == "__main__":
    main()
