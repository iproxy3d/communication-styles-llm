import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from style_demo.association import EntityExtractor
from style_demo.db import CHARACTER_WEIGHT_PROFILES, Repository, initialize_database
from style_demo.emotion import EMOTIONS, apply_character_weights, vector
from style_demo.engine import StyleDemo
from style_demo.local_llm import ContextEchoLLM


class FixedClassifier:
    """Offline unit-test double; production demo uses Multi-Motions 28."""

    def __init__(self, dominant: str = "neutral", score: float = 0.9) -> None:
        self.dominant = dominant
        self.score = score

    def predict(self, text: str) -> dict[str, float]:
        del text
        values = {name: 0.01 for name in EMOTIONS}
        values[self.dominant] = self.score
        if self.dominant != "neutral":
            values["neutral"] = 0.05
        return vector(values)


class DemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / "test.sqlite3"
        initialize_database(path)
        self.repo = Repository(path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_demo(self, character: str, emotion: str = "neutral", **kwargs) -> StyleDemo:
        return StyleDemo(
            self.repo,
            ContextEchoLLM(),
            FixedClassifier(emotion),
            self.repo.get_character(character),
            **kwargs,
        )

    def test_database_contains_three_characters(self) -> None:
        self.assertEqual(
            [item.name for item in self.repo.list_characters()],
            ["Мира", "Алекс", "Ирис"],
        )

    def test_character_weights_are_stored_separately_from_microdialogues(self) -> None:
        for character in self.repo.list_characters():
            self.assertEqual(set(character.character_weights), set(EMOTIONS))
            self.assertEqual(
                character.character_weights,
                CHARACTER_WEIGHT_PROFILES[character.name],
            )
        with sqlite3.connect(self.repo.path) as connection:
            table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'character_weights'"
            ).fetchone()
        self.assertIsNotNone(table)

    def test_character_weight_profiles_follow_the_documented_behaviors(self) -> None:
        mira = self.repo.get_character("Мира").character_weights
        alex = self.repo.get_character("Алекс").character_weights
        iris = self.repo.get_character("Ирис").character_weights

        self.assertGreater(mira["caring"], mira["anger"])
        self.assertGreater(mira["optimism"], mira["disapproval"])
        self.assertLess(alex["anger"], 1.0)
        self.assertGreater(alex["neutral"], alex["amusement"])
        self.assertGreater(iris["amusement"], iris["anger"])
        self.assertGreater(iris["surprise"], iris["grief"])
        for profile in CHARACTER_WEIGHT_PROFILES.values():
            self.assertTrue(all(0.0 <= value <= 1.0 for value in profile.values()))
        with self.assertRaises(ValueError):
            apply_character_weights({"anger": 1.0}, {"anger": 1.01})

    def test_database_contains_microdialogues_for_all_28_emotions_and_styles(self) -> None:
        self.assertEqual(len(EMOTIONS), 28)
        for character in self.repo.list_characters():
            for emotion in EMOTIONS:
                for intensity in (0, 1, 2):
                    dialogue = self.repo.get_style_microdialogue(
                        character.style_id, emotion, intensity
                    )
                    self.assertEqual(
                        [item["role"] for item in dialogue],
                        ["user", "assistant"],
                        msg=f"{character.name}/{emotion}/intensity={intensity}",
                    )
                    self.assertTrue(dialogue[0]["content"])
                    self.assertTrue(dialogue[1]["content"])

    def test_emotion_selects_field_and_injects_microdialogues(self) -> None:
        # Iris keeps anger available at a restrained weight, so this test
        # exercises the ordinary anger route without bypassing W_character.
        demo = self.make_demo("Ирис", "anger")
        result = demo.respond(
            "Ты меня бесишь, я очень злюсь!",
            motivation_level=1,
            keep_history=False,
            learn_memory=False,
        )
        self.assertEqual(result.trace.selected_emotion, "anger")
        self.assertTrue(result.trace.style_microdialogue)
        self.assertTrue(result.trace.motivation_microdialogue)
        self.assertTrue(result.trace.model_messages[-1]["content"].startswith("Ты меня"))

    def test_character_weights_are_applied_before_emotion_selection(self) -> None:
        character = self.repo.get_character("Мира")
        with sqlite3.connect(self.repo.path) as connection:
            connection.execute(
                "UPDATE character_weights SET weight = 0 WHERE character_id = ?",
                (character.id,),
            )
            connection.execute(
                "UPDATE character_weights SET weight = 1.0 WHERE character_id = ? AND emotion = 'fear'",
                (character.id,),
            )
            connection.commit()

        character = self.repo.get_character("Мира")
        result = StyleDemo(
            self.repo,
            ContextEchoLLM(),
            FixedClassifier("fear"),
            character,
        ).respond(
            "Я боюсь опоздать.",
            use_memory=False,
            keep_history=False,
            learn_memory=False,
        )
        self.assertEqual(result.trace.selected_emotion, "fear")
        self.assertAlmostEqual(
            result.trace.reaction_scores["fear"],
            result.trace.communication_state["fear"] * 1.0,
        )
        self.assertEqual(result.trace.reaction_scores["anger"], 0.0)


    def test_neutral_always_injects_character_style_microdialogue(self) -> None:
        expected_markers = {
            "Мира": "Спокойно разберём ситуацию",
            "Алекс": "Перейдём к сути вопроса",
            "Ирис": "реальность решила быть обычной",
        }
        for character_name, marker in expected_markers.items():
            demo = self.make_demo(character_name, "neutral")
            result = demo.respond(
                "Давайте разберём этот вопрос.",
                use_memory=False,
                keep_history=False,
                learn_memory=False,
                motivation_level=0,
            )
            self.assertEqual(result.trace.selected_emotion, "neutral")
            self.assertEqual(
                [item["role"] for item in result.trace.style_microdialogue],
                ["user", "assistant"],
            )
            self.assertTrue(result.trace.style_microdialogue[0]["content"])
            self.assertIn(marker, result.trace.style_microdialogue[1]["content"])
            self.assertIn(
                result.trace.style_microdialogue[0], result.trace.model_messages
            )
            self.assertIn(
                result.trace.style_microdialogue[1], result.trace.model_messages
            )

    def test_hidden_microdialogue_is_not_saved_as_real_history(self) -> None:
        demo = self.make_demo("Алекс", "annoyance")
        result = demo.respond("Опять та же ошибка")
        self.assertGreater(len(result.trace.model_messages), 2)
        self.assertEqual(len(demo.real_history), 2)
        self.assertEqual(
            demo.real_history[0],
            {"role": "user", "content": "Опять та же ошибка"},
        )

    def test_motivation_microdialogues_use_user_assistant_order(self) -> None:
        for character in self.repo.list_characters():
            for level in (-1, 1, 2):
                dialogue = self.repo.get_motivation_microdialogue(character.id, level)
                self.assertEqual(
                    [item["role"] for item in dialogue],
                    ["user", "assistant"],
                    msg=f"{character.name}/motivation level={level}",
                )

    def test_motivation_microdialogues_are_character_specific(self) -> None:
        mira = self.repo.get_character("Мира")
        alex = self.repo.get_character("Алекс")
        iris = self.repo.get_character("Ирис")
        self.assertIn(
            "коротко",
            self.repo.get_motivation_microdialogue(mira.id, 1)[0]["content"],
        )
        self.assertIn(
            "бро",
            self.repo.get_motivation_microdialogue(alex.id, 1)[0]["content"].lower(),
        )
        self.assertIn(
            "вайб",
            self.repo.get_motivation_microdialogue(iris.id, 2)[0]["content"].lower(),
        )
        self.assertEqual(
            self.repo.get_motivation_microdialogue(iris.id, 0),
            [],
        )

    def test_iris_default_motivation_is_disabled(self) -> None:
        iris = self.repo.get_character("Ирис")
        self.assertEqual(iris.motivation_level, 0)
        result = self.make_demo("Ирис", "neutral").respond(
            "Проверим факты.",
            keep_history=False,
            learn_memory=False,
        )
        self.assertEqual(result.trace.motivation_microdialogue, [])

    def test_style_microdialogues_put_character_markers_on_both_roles(self) -> None:
        markers = {
            "Мира": (
                ("заинтерес", "расскаж", "слушаю", "поделитесь", "готовы"),
                ("приятно", "заинтерес", "вместе", "рядом", "важно"),
            ),
            "Алекс": ("бро", "чувак", "рил", "без проблем"),
            "Ирис": ("рил", "вайб", "жиза", "pov", "имба", "кринж", "рофл", "мув"),
        }
        for character in self.repo.list_characters():
            marker_set = markers[character.name]
            if character.name == "Мира":
                user_markers, assistant_markers = marker_set
            else:
                user_markers = marker_set
                assistant_markers = marker_set
            for emotion in EMOTIONS:
                for intensity in (0, 1, 2):
                    dialogue = self.repo.get_style_microdialogue(
                        character.style_id, emotion, intensity
                    )
                    self.assertTrue(
                        any(
                            marker in dialogue[0]["content"].lower()
                            for marker in user_markers
                        ),
                        msg=f"missing user marker for {character.name}/{emotion}/{intensity}",
                    )
                    self.assertTrue(
                        any(
                            marker in dialogue[1]["content"].lower()
                            for marker in assistant_markers
                        ),
                        msg=f"missing assistant marker for {character.name}/{emotion}/{intensity}",
                    )

    def test_control_context_preserves_role_alternation(self) -> None:
        for character_name in ("Мира", "Алекс", "Ирис"):
            demo = self.make_demo(character_name, "fear")
            result = demo.respond(
                "Я боюсь опоздать на рейс.",
                use_memory=False,
                keep_history=False,
                learn_memory=False,
            )
            roles = [item["role"] for item in result.trace.model_messages]
            self.assertEqual(roles[0], "system")
            self.assertEqual(roles[-1], "user")
            for previous, current in zip(roles[1:], roles[2:]):
                self.assertNotEqual(
                    previous,
                    current,
                    msg=f"{character_name}: roles={roles}",
                )

    def test_style_disabled_is_reported_as_disabled(self) -> None:
        demo = self.make_demo("Мира", "fear")
        result = demo.respond(
            "Я боюсь опоздать на рейс.",
            use_style=False,
            use_memory=False,
            keep_history=False,
            learn_memory=False,
        )
        self.assertEqual(result.trace.style_name, "disabled")
        self.assertEqual(result.trace.style_microdialogue, [])
        self.assertEqual(result.trace.motivation_microdialogue, [])

    def test_motivation_zero_adds_nothing(self) -> None:
        demo = self.make_demo("Ирис", "joy")
        result = demo.respond(
            "Ура, всё получилось!",
            motivation_level=0,
            keep_history=False,
            learn_memory=False,
        )
        self.assertEqual(result.trace.motivation_microdialogue, [])

    def test_classifier_vector_keeps_independent_scores_not_sum_normalized(self) -> None:
        demo = self.make_demo("Мира", "fear")
        result = demo.respond(
            "Я боюсь опоздать на рейс.",
            use_memory=False,
            keep_history=False,
            learn_memory=False,
        )
        self.assertAlmostEqual(result.trace.user_state["fear"], 0.9)
        # Independent sigmoid-like values do not have to sum to one.
        self.assertNotAlmostEqual(sum(result.trace.user_state.values()), 1.0)

    def test_entity_normalization_finds_inflected_demo_entities(self) -> None:
        extractor = EntityExtractor(self.repo)
        matches = extractor.extract("Я говорил с Ивановым о самолетами и рейсе")
        names = {item.canonical for item in matches}
        self.assertIn("иванов", names)
        self.assertIn("самолет", names)
        self.assertIn("рейс", names)

    def test_associative_memory_is_learned_and_changes_next_state(self) -> None:
        character = self.repo.get_character("Мира")
        demo = StyleDemo(
            self.repo,
            ContextEchoLLM(),
            FixedClassifier("fear"),
            character,
            memory_beta=0.8,
            memory_eta=0.5,
        )

        for _ in range(3):
            demo.respond(
                "Я очень боюсь самолета, мне страшно летать.",
                keep_history=False,
                learn_memory=True,
            )

        association_rows = dict(self.repo.list_associations(character.id))
        self.assertIn("самолет", association_rows)
        self.assertGreater(association_rows["самолет"].vector["fear"], 0.0)
        self.assertEqual(association_rows["самолет"].encounters, 3)

        recalled = demo.respond(
            "Завтра летим на самолете в Париж.",
            keep_history=False,
            learn_memory=False,
        )
        self.assertIn("самолет", {m.canonical for m in recalled.trace.matched_entities})
        self.assertGreater(recalled.trace.associative_state["fear"], 0.0)
        self.assertGreaterEqual(
            recalled.trace.communication_state["fear"],
            recalled.trace.base_communication_state["fear"],
        )

    def test_associative_memory_uses_E_t_and_clamps_to_unit_interval(self) -> None:
        character = self.repo.get_character("Мира")
        demo = StyleDemo(
            self.repo,
            ContextEchoLLM(),
            FixedClassifier("fear"),
            character,
            memory_eta=1.0,
        )
        first = demo.respond(
            "Я боюсь самолета.",
            keep_history=False,
            learn_memory=True,
        )
        update = first.trace.memory_updates[0]
        # On the first turn E_t is neutral; the strong current U_t/S_t must
        # not be written directly into A(x).
        self.assertEqual(update.after["fear"], 0.0)
        self.assertGreater(update.after["neutral"], 0.0)

        with sqlite3.connect(self.repo.path) as connection:
            entity_id = connection.execute(
                "SELECT id FROM memory_entities WHERE canonical = 'самолет'"
            ).fetchone()[0]
        bounded = self.repo.update_association(
            character.id,
            entity_id,
            {"fear": 1.0, "anger": -2.0},
            eta=1.0,
        )
        self.assertEqual(bounded.vector["fear"], 1.0)
        self.assertEqual(bounded.vector["anger"], 0.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in bounded.vector.values()))

        self.repo.decay_associations(character.id, gamma=0.5)
        decayed = self.repo.get_association(character.id, entity_id)
        self.assertIsNotNone(decayed)
        self.assertEqual(decayed.vector["fear"], 0.5)

        upper = self.repo.update_association(
            character.id,
            entity_id,
            {"fear": 5.0},
            eta=1.0,
        )
        self.assertEqual(upper.vector["fear"], 1.0)

    def test_compare_like_read_does_not_update_memory_when_learning_disabled(self) -> None:
        character = self.repo.get_character("Мира")
        demo = StyleDemo(
            self.repo,
            ContextEchoLLM(),
            FixedClassifier("fear"),
            character,
        )
        demo.respond(
            "Я боюсь самолета.",
            keep_history=False,
            learn_memory=False,
        )
        self.assertEqual(self.repo.list_associations(character.id), [])


if __name__ == "__main__":
    unittest.main()
