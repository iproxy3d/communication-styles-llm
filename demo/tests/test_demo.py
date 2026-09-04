import tempfile
import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from style_demo.association import EntityExtractor
from style_demo.db import Repository, initialize_database
from style_demo.engine import StyleDemo
from style_demo.local_llm import ContextEchoLLM


class DemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / "test.sqlite3"
        initialize_database(path)
        self.repo = Repository(path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_database_contains_three_characters(self) -> None:
        self.assertEqual(
            [item.name for item in self.repo.list_characters()],
            ["Мира", "Алекс", "Ирис"],
        )

    def test_emotion_selects_string_field_and_injects_microdialogues(self) -> None:
        demo = StyleDemo(self.repo, ContextEchoLLM(), self.repo.get_character("Мира"))
        result = demo.respond(
            "Ты меня бесишь, я очень злюсь!",
            keep_history=False,
            learn_memory=False,
        )
        self.assertEqual(result.trace.selected_emotion, "anger")
        self.assertTrue(result.trace.style_microdialogue)
        self.assertTrue(result.trace.motivation_microdialogue)
        self.assertTrue(result.trace.model_messages[-1]["content"].startswith("Ты меня"))

    def test_hidden_microdialogue_is_not_saved_as_real_history(self) -> None:
        demo = StyleDemo(self.repo, ContextEchoLLM(), self.repo.get_character("Алекс"))
        result = demo.respond("Опять та же ошибка")
        self.assertGreater(len(result.trace.model_messages), 2)
        self.assertEqual(len(demo.real_history), 2)
        self.assertEqual(
            demo.real_history[0],
            {"role": "user", "content": "Опять та же ошибка"},
        )

    def test_motivation_zero_adds_nothing(self) -> None:
        demo = StyleDemo(self.repo, ContextEchoLLM(), self.repo.get_character("Ирис"))
        result = demo.respond(
            "Ура, всё получилось!",
            motivation_level=0,
            keep_history=False,
            learn_memory=False,
        )
        self.assertEqual(result.trace.motivation_microdialogue, [])

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
        self.assertGreater(
            recalled.trace.communication_state["fear"],
            recalled.trace.base_communication_state["fear"],
        )

    def test_compare_like_read_does_not_update_memory_when_learning_disabled(self) -> None:
        character = self.repo.get_character("Мира")
        demo = StyleDemo(self.repo, ContextEchoLLM(), character)
        demo.respond(
            "Я боюсь самолета.",
            keep_history=False,
            learn_memory=False,
        )
        self.assertEqual(self.repo.list_associations(character.id), [])


if __name__ == "__main__":
    unittest.main()
