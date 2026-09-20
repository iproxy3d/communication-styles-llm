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


def _weight_profile(**overrides: float) -> dict[str, float]:
    """Build a dense character policy vector in the Multi-Motions order.

    ``1.0`` is the maximum contribution of a route. Values below one reduce
    that route's contribution to the article's
    ``argmax(S_t ⊙ W_character)`` selector. These are transparent demo
    defaults inferred from the three written character descriptions, not
    hidden model parameters.
    """
    unknown = set(overrides) - set(EMOTIONS)
    if unknown:
        raise ValueError(f"Unknown character-weight coordinates: {sorted(unknown)}")
    invalid = {
        emotion: value
        for emotion, value in overrides.items()
        if not 0.0 <= float(value) <= 1.0
    }
    if invalid:
        raise ValueError(
            f"Character weights must be in [0, 1]: {sorted(invalid.items())}"
        )
    return {
        emotion: float(overrides.get(emotion, 1.0))
        for emotion in EMOTIONS
    }


# The article specifies the W_character mechanism, but does not prescribe a
# complete numeric vector for these three demo names. The following explicit
# profiles make the prose descriptions observable in the selector while
# leaving the article's formula unchanged.
CHARACTER_WEIGHT_PROFILES = {
    "Мира": _weight_profile(
        admiration=1.00,
        amusement=0.85,
        anger=0.15,
        annoyance=0.20,
        approval=1.00,
        caring=1.00,
        confusion=1.00,
        curiosity=0.95,
        desire=0.85,
        disappointment=0.75,
        disapproval=0.25,
        disgust=0.20,
        embarrassment=0.85,
        excitement=0.95,
        fear=1.00,
        gratitude=1.00,
        grief=1.00,
        joy=1.00,
        love=1.00,
        nervousness=1.00,
        optimism=1.00,
        pride=1.00,
        realization=1.00,
        relief=1.00,
        remorse=1.00,
        sadness=1.00,
        surprise=0.95,
        neutral=1.00,
    ),
    "Алекс": _weight_profile(
        admiration=1.00,
        amusement=0.45,
        anger=0.25,
        annoyance=0.30,
        approval=1.00,
        caring=0.85,
        confusion=1.00,
        curiosity=1.00,
        desire=0.80,
        disappointment=0.80,
        disapproval=0.45,
        disgust=0.25,
        embarrassment=0.55,
        excitement=0.70,
        fear=0.85,
        gratitude=0.90,
        grief=0.75,
        joy=0.80,
        love=0.55,
        nervousness=0.85,
        optimism=1.00,
        pride=0.95,
        realization=1.00,
        relief=1.00,
        remorse=0.85,
        sadness=0.75,
        surprise=0.95,
        neutral=1.00,
    ),
    "Ирис": _weight_profile(
        admiration=1.00,
        amusement=1.00,
        anger=0.70,
        annoyance=0.75,
        approval=1.00,
        caring=0.90,
        confusion=1.00,
        curiosity=1.00,
        desire=0.90,
        disappointment=0.90,
        disapproval=0.75,
        disgust=0.60,
        embarrassment=0.35,
        excitement=1.00,
        fear=0.70,
        gratitude=0.95,
        grief=0.20,
        joy=1.00,
        love=0.85,
        nervousness=0.65,
        optimism=1.00,
        pride=1.00,
        realization=1.00,
        relief=1.00,
        remorse=0.45,
        sadness=0.30,
        surprise=1.00,
        neutral=1.00,
    ),
}


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
    character_weights: dict[str, float]


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
            return [
                self._character(row, self._character_weights(connection, int(row["id"])))
                for row in rows
            ]

    def get_character(self, name: str) -> Character:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM characters WHERE lower(name) = lower(?)", (name,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown character: {name}")
            return self._character(
                row, self._character_weights(connection, int(row["id"]))
            )

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

    def get_motivation_microdialogue(
        self, character_id: int, level: int
    ) -> list[dict[str, str]]:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT microdialogue FROM motivation_styles
                   WHERE character_id = ? AND level = ?""",
                (character_id, level),
            ).fetchone()
        if row is None:
            raise KeyError(
                f"Unknown motivation level {level} for character id {character_id}"
            )
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
        # Article formula: A_{t+1}(x) = A_t(x) + eta * (E_t - A_t(x)).
        # The stored association is kept in the same bounded [0, 1] scale as E_t.
        updated = {}
        for name in STYLE_FIELDS:
            ema = (1.0 - eta) * float(old.get(name, 0.0)) + eta * float(
                state.get(name, 0.0)
            )
            updated[name] = max(0.0, min(1.0, ema))
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

    def decay_associations(self, character_id: int, gamma: float) -> None:
        """Apply the article's separate global A <- gamma * A decay step."""
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT entity_id, vector FROM associative_memory
                   WHERE character_id = ?""",
                (character_id,),
            ).fetchall()
            for row in rows:
                current = vector(json.loads(row["vector"]))
                decayed = {
                    name: max(0.0, min(1.0, gamma * float(current[name])))
                    for name in STYLE_FIELDS
                }
                connection.execute(
                    """UPDATE associative_memory SET vector = ?
                       WHERE character_id = ? AND entity_id = ?""",
                    (
                        json.dumps(decayed, ensure_ascii=False),
                        character_id,
                        int(row["entity_id"]),
                    ),
                )
            connection.commit()

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
    def _character_weights(
        connection: sqlite3.Connection, character_id: int
    ) -> dict[str, float]:
        character_row = connection.execute(
            "SELECT name FROM characters WHERE id = ?", (character_id,)
        ).fetchone()
        character_name = str(character_row[0]) if character_row is not None else ""
        defaults = CHARACTER_WEIGHT_PROFILES.get(character_name, _weight_profile())
        try:
            rows = connection.execute(
                "SELECT emotion, weight FROM character_weights WHERE character_id = ?",
                (character_id,),
            ).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error).lower():
                raise
            rows = []
        weights = {
            str(row["emotion"]): max(0.0, min(1.0, float(row["weight"])))
            for row in rows
        }
        # initialize_database() fills every coordinate. The fallback keeps a
        # direct Repository read of an older database usable until migration is
        # run, without changing the article's W_character semantics.
        return {
            emotion: weights.get(emotion, defaults[emotion])
            for emotion in EMOTIONS
        }

    @staticmethod
    def _character(
        row: sqlite3.Row, character_weights: dict[str, float] | None = None
    ) -> Character:
        return Character(
            id=int(row["id"]),
            name=str(row["name"]),
            system_prompt=str(row["system_prompt"]),
            style_id=int(row["style_id"]),
            alpha=float(row["alpha"]),
            intensity=int(row["intensity"]),
            motivation_level=int(row["motivation_level"]),
            initial_state=vector(json.loads(row["initial_state"])),
            character_weights=(
                character_weights
                if character_weights is not None
                else CHARACTER_WEIGHT_PROFILES.get(
                    str(row["name"]), _weight_profile()
                )
            ),
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
                0,
                neutral_state,
            ),
        ],
    )


def _seed_character_weights(connection: sqlite3.Connection) -> None:
    """Seed character policies and migrate the former all-ones defaults.

    A partially customized table is preserved. A table whose values are all
    the old neutral ``1.0`` defaults is upgraded to the explicit demo profiles
    above, so an existing archive database receives the same initialization as
    a newly created one.
    """
    characters = connection.execute(
        "SELECT id, name FROM characters ORDER BY id"
    ).fetchall()
    for row in characters:
        character_id = int(row[0])
        character_name = str(row[1])
        profile = CHARACTER_WEIGHT_PROFILES.get(
            character_name, _weight_profile()
        )
        existing_rows = connection.execute(
            "SELECT emotion, weight FROM character_weights WHERE character_id = ?",
            (character_id,),
        ).fetchall()
        existing = {str(item[0]): float(item[1]) for item in existing_rows}
        had_out_of_range = False
        for emotion, weight in existing.items():
            bounded = max(0.0, min(1.0, weight))
            if bounded != weight:
                had_out_of_range = True
                connection.execute(
                    "UPDATE character_weights SET weight = ? "
                    "WHERE character_id = ? AND emotion = ?",
                    (bounded, character_id, emotion),
                )
                existing[emotion] = bounded

        # Databases produced before character-specific initialization contain
        # exactly the neutral vector. Migrate that legacy state once, while
        # preserving any table that already contains a custom value.
        if (
            not had_out_of_range
            and existing
            and set(existing) == set(EMOTIONS)
            and all(abs(weight - 1.0) <= 1e-12 for weight in existing.values())
        ):
            connection.executemany(
                "UPDATE character_weights SET weight = ? "
                "WHERE character_id = ? AND emotion = ?",
                [
                    (profile[emotion], character_id, emotion)
                    for emotion in EMOTIONS
                ],
            )
            continue

        connection.executemany(
            """INSERT OR IGNORE INTO character_weights(character_id, emotion, weight)
               VALUES (?, ?, ?)""",
            [
                (character_id, emotion, profile[emotion])
                for emotion in EMOTIONS
            ],
        )


_MOTIVATION_NAMES = {
    -1: "антимотивация",
    0: "нет воздействия",
    1: "слабая",
    2: "высокая",
}


_MOTIVATION_DEFAULTS = {
    "Мира": {
        -1: [
            message("user", "Если устали, можете не продолжать — я рядом."),
            message("assistant", "Понимаю. Не буду давить и помогу спокойно завершить мысль."),
        ],
        0: [],
        1: [
            message("user", "Мира, ответьте коротко, хорошо? Мне важно вас услышать."),
            message("assistant", "Конечно. Отвечу кратко и бережно — мне приятно быть вам полезной."),
        ],
        2: [
            message("user", "Мира, продолжите разговор и проявите инициативу, ладно?"),
            message("assistant", "С радостью. Расскажу подробнее и задам встречный вопрос — вы меня заинтересовали."),
        ],
    },
    "Алекс": {
        -1: [
            message("user", "Бро, можешь не разворачивать тему и ответить без лишнего."),
            message("assistant", "Окей, чувак. Не буду растягивать разговор — только необходимое."),
        ],
        0: [],
        1: [
            message("user", "Бро, не растекайся, ответь кратко."),
            message("assistant", "Без проблем, чувак: отвечу коротко и по делу."),
        ],
        2: [
            message("user", "Чувак, продолжи разговор и задай встречный вопрос."),
            message("assistant", "Окей, бро. Раскрою детали и задам встречный вопрос."),
        ],
    },
    "Ирис": {
        -1: [
            message("user", "Рил, я ливаю — не форсь ответ, ок?"),
            message("assistant", "Жиза, не буду душнить: сверну разговор без лишнего кринжа."),
        ],
        0: [],
        1: [
            message("user", "Чечик, ответь кратко, без кринжа, рил."),
            message("assistant", "Окей, это нормис-режим: коротко и без душнилова."),
        ],
        2: [
            message("user", "Гойда, продолжи этот вайб и задай вопрос, рил."),
            message("assistant", "Имба, продолжаю: разверну мысль и задам встречный вопрос — не ливаем."),
        ],
    },
}


def _motivation_defaults(character_name: str) -> dict[int, list[dict[str, str]]]:
    """Return character-specific USER -> ASSISTANT motivation examples."""
    try:
        return _MOTIVATION_DEFAULTS[character_name]
    except KeyError as error:
        raise KeyError(f"Unknown character for motivation defaults: {character_name}") from error


def _seed_motivation(connection: sqlite3.Connection) -> None:
    # Motivation is orthogonal to the emotion-classifier output and remains
    # a separate control layer. Keeping USER -> ASSISTANT order makes the final
    # chat context alternate roles naturally when it follows the style example.
    characters = connection.execute(
        "SELECT id, name FROM characters ORDER BY id"
    ).fetchall()
    connection.executemany(
        """INSERT INTO motivation_styles
           (character_id, level, name, microdialogue)
           VALUES (?, ?, ?, ?)""",
        [
            (
                int(character[0]),
                level,
                _MOTIVATION_NAMES[level],
                json.dumps(messages, ensure_ascii=False) if messages else None,
            )
            for character in characters
            for level, messages in _motivation_defaults(str(character[1])).items()
        ],
    )


def _ensure_motivation_rows(connection: sqlite3.Connection) -> None:
    """Ensure built-in rows exist and use a valid alternating chat-role order."""
    characters = connection.execute(
        "SELECT id, name FROM characters ORDER BY id"
    ).fetchall()
    for character in characters:
        character_id = int(character[0])
        character_name = str(character[1])
        for level, messages in _motivation_defaults(character_name).items():
            payload = json.dumps(messages, ensure_ascii=False) if messages else None
            row = connection.execute(
                """SELECT microdialogue FROM motivation_styles
                   WHERE character_id = ? AND level = ?""",
                (character_id, level),
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO motivation_styles
                       (character_id, level, name, microdialogue)
                       VALUES (?, ?, ?, ?)""",
                    (character_id, level, _MOTIVATION_NAMES[level], payload),
                )
                continue

            # Preserve already well-formed custom USER -> ASSISTANT pairs.
            # Repair only rows whose role order cannot be composed cleanly
            # with the style example and the final real USER message.
            should_repair = False
            raw = row["microdialogue"]
            if level == 0:
                should_repair = bool(raw)
            else:
                try:
                    parsed = json.loads(raw) if raw else []
                    roles = [item.get("role") for item in parsed]
                    should_repair = roles != ["user", "assistant"]
                except (TypeError, ValueError, json.JSONDecodeError):
                    should_repair = True
            if should_repair:
                connection.execute(
                    """UPDATE motivation_styles
                       SET name = ?, microdialogue = ?
                       WHERE character_id = ? AND level = ?""",
                    (_MOTIVATION_NAMES[level], payload, character_id, level),
                )


def _ensure_motivation_schema(connection: sqlite3.Connection) -> bool:
    """Upgrade the old global-level table to character-specific rows.

    The previous demo version keyed this table only by ``level``.  Such rows
    cannot express a character's register, so they are kept under a legacy
    name while the active table is rebuilt with ``(character_id, level)``.
    Fresh databases use the new schema directly.
    """
    table = connection.execute(
        """SELECT name FROM sqlite_master
           WHERE type = 'table' AND name = 'motivation_styles'"""
    ).fetchone()
    if table is None:
        connection.execute(
            """CREATE TABLE motivation_styles (
                character_id INTEGER NOT NULL REFERENCES characters(id),
                level INTEGER NOT NULL,
                name TEXT NOT NULL,
                microdialogue TEXT,
                PRIMARY KEY(character_id, level)
        )"""
        )
        return False

    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(motivation_styles)").fetchall()
    }
    if "character_id" in columns:
        return False

    connection.execute("ALTER TABLE motivation_styles RENAME TO motivation_styles_legacy")
    connection.execute(
        """CREATE TABLE motivation_styles (
            character_id INTEGER NOT NULL REFERENCES characters(id),
            level INTEGER NOT NULL,
            name TEXT NOT NULL,
            microdialogue TEXT,
            PRIMARY KEY(character_id, level)
        )"""
    )
    return True


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
            CREATE TABLE character_weights (
                character_id INTEGER NOT NULL REFERENCES characters(id),
                emotion TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY(character_id, emotion)
            );
            CREATE TABLE motivation_styles (
                character_id INTEGER NOT NULL REFERENCES characters(id),
                level INTEGER NOT NULL,
                name TEXT NOT NULL,
                microdialogue TEXT,
                PRIMARY KEY(character_id, level)
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
        _seed_character_weights(connection)
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
            CREATE TABLE IF NOT EXISTS character_weights (
                character_id INTEGER NOT NULL REFERENCES characters(id),
                emotion TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY(character_id, emotion)
            );
            """
        )
        _seed_character_weights(connection)
        _seed_memory_entities(connection)
        motivation_schema_was_migrated = _ensure_motivation_schema(connection)
        legacy_motivation_table = connection.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type = 'table' AND name = 'motivation_styles_legacy'"""
        ).fetchone()
        if motivation_schema_was_migrated or legacy_motivation_table is not None:
            # The previous built-in archive used level 1 for Iris.  The
            # character-specific demo deliberately uses level 0 so the
            # communication-only path is visible by default.  Preserve any
            # non-legacy custom value in databases that already have the new
            # schema.
            connection.execute(
                """UPDATE characters SET motivation_level = 0
                   WHERE name = 'Ирис' AND motivation_level = 1"""
            )
        _ensure_motivation_rows(connection)

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
