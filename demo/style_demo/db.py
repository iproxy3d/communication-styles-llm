from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

STYLE_FIELDS = ("neutral", "fear", "anger", "annoyance", "joy")


@dataclass(frozen=True)
class Character:
    id: int
    name: str
    system_prompt: str
    style_id: int
    alpha: float
    intensity: int
    motivation_level: int
    initial_state: dict[str, float]


@dataclass(frozen=True)
class MemoryEntity:
    id: int
    canonical: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Association:
    character_id: int
    entity_id: int
    vector: dict[str, float]
    encounters: int


class Repository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            # sqlite3.Connection.__exit__ commits or rolls back a transaction,
            # but does not close the connection. Explicit close is essential on
            # Windows, where an open connection locks the database file.
            connection.close()

    def list_characters(self) -> list[Character]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM characters ORDER BY id").fetchall()
        return [self._character(row) for row in rows]

    def get_character(self, name: str) -> Character:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM characters WHERE lower(name) = lower(?)", (name,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown character: {name}")
        return self._character(row)

    def get_style_name(self, style_id: int) -> str:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT name FROM communication_styles WHERE id = ?", (style_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown style id: {style_id}")
        return str(row["name"])

    def get_style_microdialogue(
        self, style_id: int, emotion: str, intensity: int
    ) -> list[dict[str, str]]:
        if emotion not in STYLE_FIELDS:
            raise KeyError(f"Unsupported emotion field: {emotion}")
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT {emotion} FROM communication_styles WHERE id = ?", (style_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown style id: {style_id}")
        raw = row[emotion]
        if not raw:
            return []
        levels: dict[str, list[dict[str, str]]] = json.loads(raw)
        key = str(max(0, min(2, intensity)))
        return list(levels.get(key, levels.get("1", [])))

    def get_motivation_microdialogue(self, level: int) -> list[dict[str, str]]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT microdialogue FROM motivation_styles WHERE level = ?", (level,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown motivation level: {level}")
        return json.loads(row["microdialogue"]) if row["microdialogue"] else []

    def list_memory_entities(self) -> list[MemoryEntity]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, canonical, aliases FROM memory_entities ORDER BY id"
            ).fetchall()
        return [
            MemoryEntity(
                id=int(row["id"]),
                canonical=str(row["canonical"]),
                aliases=tuple(json.loads(row["aliases"] or "[]")),
            )
            for row in rows
        ]

    def get_association(self, character_id: int, entity_id: int) -> Association | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT character_id, entity_id, vector, encounters
                   FROM associative_memory
                   WHERE character_id = ? AND entity_id = ?""",
                (character_id, entity_id),
            ).fetchone()
        if row is None:
            return None
        return Association(
            character_id=int(row["character_id"]),
            entity_id=int(row["entity_id"]),
            vector=json.loads(row["vector"]),
            encounters=int(row["encounters"]),
        )

    def update_association(
        self,
        character_id: int,
        entity_id: int,
        state: dict[str, float],
        eta: float,
    ) -> Association:
        if not 0.0 < eta <= 1.0:
            raise ValueError("eta must be in (0, 1]")
        current = self.get_association(character_id, entity_id)
        old = current.vector if current is not None else {name: 0.0 for name in STYLE_FIELDS}
        updated = {
            name: (1.0 - eta) * float(old.get(name, 0.0))
            + eta * float(state.get(name, 0.0))
            for name in STYLE_FIELDS
        }
        encounters = (current.encounters if current is not None else 0) + 1
        payload = json.dumps(updated, ensure_ascii=False)
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO associative_memory(character_id, entity_id, vector, encounters)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(character_id, entity_id) DO UPDATE SET
                       vector = excluded.vector,
                       encounters = excluded.encounters""",
                (character_id, entity_id, payload, encounters),
            )
            connection.commit()
        return Association(character_id, entity_id, updated, encounters)

    def list_associations(self, character_id: int) -> list[tuple[str, Association]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT e.canonical, m.character_id, m.entity_id, m.vector, m.encounters
                   FROM associative_memory AS m
                   JOIN memory_entities AS e ON e.id = m.entity_id
                   WHERE m.character_id = ?
                   ORDER BY m.encounters DESC, e.canonical""",
                (character_id,),
            ).fetchall()
        return [
            (
                str(row["canonical"]),
                Association(
                    character_id=int(row["character_id"]),
                    entity_id=int(row["entity_id"]),
                    vector=json.loads(row["vector"]),
                    encounters=int(row["encounters"]),
                ),
            )
            for row in rows
        ]

    def clear_associations(self, character_id: int | None = None) -> None:
        with self.connect() as connection:
            if character_id is None:
                connection.execute("DELETE FROM associative_memory")
            else:
                connection.execute(
                    "DELETE FROM associative_memory WHERE character_id = ?",
                    (character_id,),
                )
            connection.commit()

    @staticmethod
    def _character(row: sqlite3.Row) -> Character:
        return Character(
            id=int(row["id"]),
            name=str(row["name"]),
            system_prompt=str(row["system_prompt"]),
            style_id=int(row["style_id"]),
            alpha=float(row["alpha"]),
            intensity=int(row["intensity"]),
            motivation_level=int(row["motivation_level"]),
            initial_state=json.loads(row["initial_state"]),
        )


def encode_levels(
    low: list[dict[str, str]],
    medium: list[dict[str, str]],
    high: list[dict[str, str]],
) -> str:
    return json.dumps({"0": low, "1": medium, "2": high}, ensure_ascii=False)


def message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _seed_memory_entities(connection: sqlite3.Connection) -> None:
    entities = [
        (1, "самолет", ["самолёт", "самолет"]),
        (2, "рейс", ["рейс"]),
        (3, "аэропорт", ["аэропорт"]),
        (4, "иванов", ["иванов"]),
        (5, "париж", ["париж"]),
        (6, "билет", ["билет"]),
    ]
    connection.executemany(
        "INSERT OR IGNORE INTO memory_entities(id, canonical, aliases) VALUES (?, ?, ?)",
        [(entity_id, canonical, json.dumps(aliases, ensure_ascii=False)) for entity_id, canonical, aliases in entities],
    )


def _ensure_association_schema(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_entities (
                id INTEGER PRIMARY KEY,
                canonical TEXT NOT NULL UNIQUE,
                aliases TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS associative_memory (
                character_id INTEGER NOT NULL REFERENCES characters(id),
                entity_id INTEGER NOT NULL REFERENCES memory_entities(id),
                vector TEXT NOT NULL,
                encounters INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(character_id, entity_id)
            );
            """
        )
        _seed_memory_entities(connection)
        connection.commit()


def initialize_database(path: str | Path, force: bool = False) -> None:
    db_path = Path(path)
    if db_path.exists() and not force:
        _ensure_association_schema(db_path)
        return
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.executescript(
            """
            CREATE TABLE communication_styles (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                neutral TEXT,
                fear TEXT,
                anger TEXT,
                annoyance TEXT,
                joy TEXT
            );
            CREATE TABLE characters (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                system_prompt TEXT NOT NULL,
                style_id INTEGER NOT NULL REFERENCES communication_styles(id),
                alpha REAL NOT NULL,
                intensity INTEGER NOT NULL,
                motivation_level INTEGER NOT NULL,
                initial_state TEXT NOT NULL
            );
            CREATE TABLE motivation_styles (
                level INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                microdialogue TEXT
            );
            CREATE TABLE memory_entities (
                id INTEGER PRIMARY KEY,
                canonical TEXT NOT NULL UNIQUE,
                aliases TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE associative_memory (
                character_id INTEGER NOT NULL REFERENCES characters(id),
                entity_id INTEGER NOT NULL REFERENCES memory_entities(id),
                vector TEXT NOT NULL,
                encounters INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(character_id, entity_id)
            );
            """
        )

        supportive = {
            "neutral": encode_levels([], [], []),
            "fear": encode_levels(
                [message("assistant", "Спокойно. Сначала уточним факты.")],
                [message("user", "Я волнуюсь."), message("assistant", "Понимаю тревогу. Давайте спокойно разберёмся по шагам.")],
                [message("user", "Мне страшно, я теряюсь."), message("assistant", "Я рядом. Сначала отделим то, что уже известно, затем выберем один безопасный следующий шаг."), message("user", "Хорошо, начнём с первого шага.")],
            ),
            "anger": encode_levels(
                [message("assistant", "Сохраняю спокойствие и отвечаю по существу.")],
                [message("user", "Меня это злит."), message("assistant", "Понимаю раздражение. Давайте без резкости найдём решение.")],
                [message("user", "Я очень зол."), message("assistant", "Вижу, что ситуация вывела вас из себя. Я не буду спорить; разберём конкретную ошибку и следующий шаг."), message("user", "Хорошо, к сути.")],
            ),
            "annoyance": encode_levels(
                [message("assistant", "Отвечу прямо и спокойно.")],
                [message("user", "Это уже надоело."), message("assistant", "Согласен, повторение неприятно. Давайте сразу к решению.")],
                [message("user", "Сколько можно повторять?"), message("assistant", "Понимаю раздражение. Не буду ходить по кругу: кратко обозначу причину и действие."), message("user", "Давайте.")],
            ),
            "joy": encode_levels(
                [message("assistant", "Это хорошая новость.")],
                [message("user", "У меня получилось!"), message("assistant", "Отлично! Рад вашему результату.")],
                [message("user", "Наконец-то получилось!"), message("assistant", "Здорово! Это действительно повод порадоваться. Давайте закрепим успех следующим шагом."), message("user", "Да!")],
            ),
        }
        formal = {
            "neutral": encode_levels([], [], []),
            "fear": encode_levels(
                [message("assistant", "Зафиксируем риски и проверим факты.")],
                [message("user", "Я опасаюсь ошибки."), message("assistant", "Опасение понятно. Предлагаю последовательно проверить исходные данные и варианты действий.")],
                [message("user", "Риск кажется высоким."), message("assistant", "Принято. Сначала оценим вероятность и последствия, затем выберем контролируемое действие."), message("user", "Продолжайте.")],
            ),
            "anger": encode_levels(
                [message("assistant", "Сохраняю корректный деловой тон.")],
                [message("user", "Это недопустимо."), message("assistant", "Претензия принята. Уточним факт нарушения и порядок исправления.")],
                [message("user", "Я требую немедленного решения."), message("assistant", "Требование зафиксировано. Перечислю доступные действия и ограничения без эмоциональной оценки."), message("user", "Хорошо.")],
            ),
            "annoyance": encode_levels(
                [message("assistant", "Отвечу кратко и предметно.")],
                [message("user", "Ситуация повторяется."), message("assistant", "Повторное обращение зафиксировано. Перейдём непосредственно к причине и действию.")],
                [message("user", "Проблема снова возникла."), message("assistant", "Понимаю основание претензии. Исключу повторные пояснения и укажу только проверку и следующий шаг."), message("user", "Продолжайте.")],
            ),
            "joy": encode_levels(
                [message("assistant", "Результат положительный.")],
                [message("user", "Всё получилось."), message("assistant", "Отличный результат. Благодарю за подтверждение.")],
                [message("user", "Проект успешно завершён."), message("assistant", "Поздравляю с успешным завершением. Рекомендую зафиксировать результат и критерии воспроизводимости."), message("user", "Согласен.")],
            ),
        }
        ironic = {
            "neutral": encode_levels([], [], []),
            "fear": encode_levels(
                [message("assistant", "Без паники: катастрофа пока не подтверждена.")],
                [message("user", "Кажется, всё пропало."), message("assistant", "Конечно, конец света очень пунктуален. А теперь спокойно проверим факты.")],
                [message("user", "Это точно катастрофа."), message("assistant", "Да, осталось только вызвать оркестр для финальных титров. Но сначала всё-таки посмотрим, что реально произошло."), message("user", "Ладно, смотрим.")],
            ),
            "anger": encode_levels(
                [message("assistant", "Очаровательно. Теперь к сути.")],
                [message("user", "Ты меня бесишь."), message("assistant", "Взаимность сегодня отменяется. Лучше разберём, что именно не сработало.")],
                [message("user", "Да сколько можно!"), message("assistant", "Судя по всему, бесконечность решила дать практическую демонстрацию. Остановим её конкретным действием."), message("user", "Каким?")],
            ),
            "annoyance": encode_levels(
                [message("assistant", "Ну да, опять. Исправляем.")],
                [message("user", "Снова та же проблема."), message("assistant", "Какая удивительная преданность традициям. Давайте её всё же исправим.")],
                [message("user", "Опять всё сначала."), message("assistant", "Стабильность достойна восхищения, жаль, что не та. Коротко: причина, проверка, решение."), message("user", "Наконец-то.")],
            ),
            "joy": encode_levels(
                [message("assistant", "Редкий случай — всё получилось.")],
                [message("user", "Получилось!"), message("assistant", "Надо же, вселенная сегодня сотрудничает. Отличный результат.")],
                [message("user", "Всё идеально!"), message("assistant", "Подозрительно идеально, но спорить не стану. Поздравляю — закрепим результат, пока вселенная не передумала."), message("user", "Давайте.")],
            ),
        }
        for style_id, name, description, fields in (
            (1, "supportive", "Тёплый, спокойный, поддерживающий", supportive),
            (2, "formal", "Сдержанный, точный, деловой", formal),
            (3, "ironic", "Мягкая ирония без изменения фактов", ironic),
        ):
            connection.execute(
                """INSERT INTO communication_styles
                (id, name, description, neutral, fear, anger, annoyance, joy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (style_id, name, description, *(fields[field] for field in STYLE_FIELDS)),
            )

        neutral_state = json.dumps(
            {"neutral": 1.0, "fear": 0.0, "anger": 0.0, "annoyance": 0.0, "joy": 0.0}
        )
        connection.executemany(
            """INSERT INTO characters
            (id, name, system_prompt, style_id, alpha, intensity, motivation_level, initial_state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (1, "Мира", "Ты — компетентный помощник. Сохраняй факты и не выдумывай детали.", 1, 0.65, 1, 2, neutral_state),
                (2, "Алекс", "Ты — деловой помощник. Отвечай точно и не выдумывай детали.", 2, 0.60, 1, 1, neutral_state),
                (3, "Ирис", "Ты — полезный собеседник с мягкой иронией. Не меняй факты и не оскорбляй пользователя.", 3, 0.70, 1, 1, neutral_state),
            ],
        )
        motivation = {
            -1: [message("assistant", "Не хочу развивать этот разговор."), message("user", "Тогда ответь предельно кратко.")],
            0: [],
            1: [message("assistant", "Хорошо, отвечу кратко и полно."), message("user", "Ответь на следующее сообщение без развития темы.")],
            2: [message("assistant", "Я готов поддержать разговор и предложить следующий шаг."), message("user", "Ответь на следующее сообщение и прояви уместную инициативу.")],
        }
        connection.executemany(
            "INSERT INTO motivation_styles(level, name, microdialogue) VALUES (?, ?, ?)",
            [
                (level, { -1: "антимотивация", 0: "нет воздействия", 1: "слабая", 2: "высокая" }[level], json.dumps(messages, ensure_ascii=False) if messages else None)
                for level, messages in motivation.items()
            ],
        )
        _seed_memory_entities(connection)
        connection.commit()
