from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from .default_styles import STYLE_DESCRIPTIONS, build_default_style_levels
from .emotion import EMOTIONS, vector

STYLE_FIELDS = EMOTIONS
STYLE_ROWS = (
    (1, "supportive"),
    (2, "formal"),
    (3, "ironic"),
)


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
                f'SELECT "{emotion}" FROM communication_styles WHERE id = ?',
                (style_id,),
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
            vector=vector(json.loads(row["vector"])),
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
        updated = vector(
            {
                name: (1.0 - eta) * float(old.get(name, 0.0))
                + eta * float(state.get(name, 0.0))
                for name in STYLE_FIELDS
            }
        )
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
                    vector=vector(json.loads(row["vector"])),
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
            initial_state=vector(json.loads(row["initial_state"])),
        )


def encode_levels(levels: dict[str, list[dict[str, str]]]) -> str:
    return json.dumps(levels, ensure_ascii=False)


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
        [
            (entity_id, canonical, json.dumps(aliases, ensure_ascii=False))
            for entity_id, canonical, aliases in entities
        ],
    )


def _initial_state() -> dict[str, float]:
    # Model-like neutral baseline rather than a categorical one-hot distribution.
    # It leaves enough room for even moderately expressed classifier emotions to
    # influence the first communication state.
    state = {name: 0.0 for name in EMOTIONS}
    state["neutral"] = 0.25
    return state


def _style_columns_sql() -> str:
    return ",\n                ".join(f'"{emotion}" TEXT' for emotion in STYLE_FIELDS)


def _insert_default_style(connection: sqlite3.Connection, style_id: int, style_name: str) -> None:
    fields = [
        encode_levels(build_default_style_levels(style_name, emotion))
        for emotion in STYLE_FIELDS
    ]
    columns = ", ".join(f'"{emotion}"' for emotion in STYLE_FIELDS)
    placeholders = ", ".join("?" for _ in range(3 + len(STYLE_FIELDS)))
    connection.execute(
        f"INSERT INTO communication_styles (id, name, description, {columns}) "
        f"VALUES ({placeholders})",
        (style_id, style_name, STYLE_DESCRIPTIONS[style_name], *fields),
    )


def _seed_characters(connection: sqlite3.Connection) -> None:
    neutral_state = json.dumps(_initial_state(), ensure_ascii=False)
    connection.executemany(
        """INSERT INTO characters
        (id, name, system_prompt, style_id, alpha, intensity, motivation_level, initial_state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                1,
                "Мира",
                "Ты — компетентный поддерживающий помощник. Сохраняй факты, не выдумывай детали и отвечай на языке текущего сообщения пользователя.",
                1,
                0.65,
                1,
                2,
                neutral_state,
            ),
            (
                2,
                "Алекс",
                "Ты — деловой помощник. Отвечай точно, не выдумывай детали и отвечай на языке текущего сообщения пользователя.",
                2,
                0.60,
                1,
                1,
                neutral_state,
            ),
            (
                3,
                "Ирис",
                "Ты — полезный собеседник с мягкой иронией. Не меняй факты, не оскорбляй пользователя и отвечай на языке текущего сообщения пользователя.",
                3,
                0.70,
                1,
                1,
                neutral_state,
            ),
        ],
    )


def _seed_motivation(connection: sqlite3.Connection) -> None:
    # Motivation is orthogonal to the emotion-classifier output and remains
    # a separate control layer.
    motivation = {
        -1: [
            message("assistant", "Не хочу развивать этот разговор."),
            message("user", "Тогда ответь предельно кратко."),
        ],
        0: [],
        1: [
            message("assistant", "Хорошо, отвечу кратко и полно."),
            message("user", "Ответь на следующее сообщение без развития темы."),
        ],
        2: [
            message("assistant", "Я готов поддержать разговор и предложить следующий шаг."),
            message("user", "Ответь на следующее сообщение и прояви уместную инициативу."),
        ],
    }
    connection.executemany(
        "INSERT INTO motivation_styles(level, name, microdialogue) VALUES (?, ?, ?)",
        [
            (
                level,
                {-1: "антимотивация", 0: "нет воздействия", 1: "слабая", 2: "высокая"}[level],
                json.dumps(messages, ensure_ascii=False) if messages else None,
            )
            for level, messages in motivation.items()
        ],
    )


def _create_database(db_path: Path) -> None:
    with closing(sqlite3.connect(db_path)) as connection:
        connection.executescript(
            f"""
            CREATE TABLE communication_styles (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                {_style_columns_sql()}
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
        for style_id, style_name in STYLE_ROWS:
            _insert_default_style(connection, style_id, style_name)
        _seed_characters(connection)
        _seed_motivation(connection)
        _seed_memory_entities(connection)
        connection.commit()


def _ensure_database_schema(db_path: Path) -> None:
    """Ensure that an existing SQLite file has the complete current schema."""
    with closing(sqlite3.connect(db_path)) as connection:
        connection.row_factory = sqlite3.Row
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

        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(communication_styles)").fetchall()
        }
        missing = [emotion for emotion in STYLE_FIELDS if emotion not in columns]
        schema_was_incomplete = bool(missing)
        for emotion in missing:
            connection.execute(
                f'ALTER TABLE communication_styles ADD COLUMN "{emotion}" TEXT'
            )

        for style_id, style_name in STYLE_ROWS:
            row = connection.execute(
                "SELECT id FROM communication_styles WHERE id = ?", (style_id,)
            ).fetchone()
            if row is None:
                _insert_default_style(connection, style_id, style_name)
                continue
            connection.execute(
                "UPDATE communication_styles SET name = ?, description = ? WHERE id = ?",
                (style_name, STYLE_DESCRIPTIONS[style_name], style_id),
            )
            # If required emotion columns had to be added, seed the complete
            # built-in style table consistently. For an already complete schema,
            # existing custom cell contents are left untouched.
            for emotion in STYLE_FIELDS:
                payload = encode_levels(build_default_style_levels(style_name, emotion))
                if schema_was_incomplete:
                    connection.execute(
                        f'UPDATE communication_styles SET "{emotion}" = ? WHERE id = ?',
                        (payload, style_id),
                    )
                else:
                    connection.execute(
                        f'UPDATE communication_styles SET "{emotion}" = ? '
                        f'WHERE id = ? AND ("{emotion}" IS NULL OR "{emotion}" = "")',
                        (payload, style_id),
                    )

        # Ensure stored initial and memory vectors cover the full classifier space.
        for row in connection.execute("SELECT id, initial_state FROM characters").fetchall():
            state = vector(json.loads(row["initial_state"]))
            if schema_was_incomplete:
                state = _initial_state()
            connection.execute(
                "UPDATE characters SET initial_state = ? WHERE id = ?",
                (json.dumps(state, ensure_ascii=False), int(row["id"])),
            )
        for row in connection.execute(
            "SELECT character_id, entity_id, vector FROM associative_memory"
        ).fetchall():
            expanded = vector(json.loads(row["vector"]))
            connection.execute(
                "UPDATE associative_memory SET vector = ? WHERE character_id = ? AND entity_id = ?",
                (
                    json.dumps(expanded, ensure_ascii=False),
                    int(row["character_id"]),
                    int(row["entity_id"]),
                ),
            )
        connection.commit()


def initialize_database(path: str | Path, force: bool = False) -> None:
    db_path = Path(path)
    if db_path.exists() and force:
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        _create_database(db_path)
    else:
        _ensure_database_schema(db_path)
