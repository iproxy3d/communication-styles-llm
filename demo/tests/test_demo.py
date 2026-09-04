import tempfile
import unittest
import sys
from pathlib import Path

# Allow both supported launch forms:
#   python -m unittest discover -s tests -v
#   python tests/test_demo.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        result = demo.respond("Ты меня бесишь, я очень злюсь!", keep_history=False)
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
        result = demo.respond("Ура, всё получилось!", motivation_level=0, keep_history=False)
        self.assertEqual(result.trace.motivation_microdialogue, [])


if __name__ == "__main__":
    unittest.main()
