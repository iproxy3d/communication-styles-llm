from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .emotion import EMOTIONS, vector


class EmotionClassifier(Protocol):
    def predict(self, text: str) -> dict[str, float]: ...


class MultiMotions28Classifier:
    """Local inference wrapper for proxy3d/multi-motions-28.

    The model has 28 independent logits.  We apply sigmoid and return the full
    dense score vector; no softmax and no sum-normalization are used.
    """

    MODEL_ID = "proxy3d/multi-motions-28"
    MAX_LENGTH = 64

    def __init__(self, model_path: str | Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(model_path)
        if not path.exists() or not (path / "config.json").exists():
            raise FileNotFoundError(
                f"Emotion classifier not found at {path}. Run: python download_model.py"
            )

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        labels = tuple(id2label[i] for i in range(len(id2label)))
        if labels != EMOTIONS:
            raise RuntimeError(
                "Unexpected classifier labels. "
                f"Expected {EMOTIONS!r}, got {labels!r}."
            )
        if int(self.model.config.num_labels) != len(EMOTIONS):
            raise RuntimeError(
                f"Expected {len(EMOTIONS)} logits, got {self.model.config.num_labels}."
            )

    def predict(self, text: str) -> dict[str, float]:
        text = text.strip()
        if not text:
            # Empty text is not sent through the encoder; represent it as neutral.
            return vector({"neutral": 1.0})

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_LENGTH,
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            scores = self.torch.sigmoid(logits).detach().cpu().float().tolist()
        return vector({emotion: scores[index] for index, emotion in enumerate(EMOTIONS)})
