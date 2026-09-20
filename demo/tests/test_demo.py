import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from style_demo.association import EntityExtractor
from style_demo.db import Repository, initialize_database
from style_demo.emotion import EMOTIONS, vector
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
        demo = self.make_demo("Мира", "anger")
        result = demo.respond(
            "Ты меня бесишь, я очень злюсь!",
            keep_history=False,
            learn_memory=False,
        )
        self.assertEqual(result.trace.selected_emotion, "anger")
        self.assertTrue(result.trace.style_microdialogue)
        self.assertTrue(result.trace.motivation_microdialogue)
        self.assertTrue(result.trace.model_messages[-1]["content"].startswith("Ты меня"))


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
        for level in (-1, 1, 2):
            dialogue = self.repo.get_motivation_microdialogue(level)
            self.assertEqual(
                [item["role"] for item in dialogue],
                ["user", "assistant"],
                msg=f"motivation level={level}",
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
