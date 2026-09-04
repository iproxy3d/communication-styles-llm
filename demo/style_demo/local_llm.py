from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ChatModel(Protocol):
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 120) -> str: ...


class LocalTransformersLLM:
    """Loads model weights from a local directory; performs no API calls."""

    def __init__(self, model_path: str | Path) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Local model not found at {path}. Run: python download_model.py"
            )
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 120) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.device)
        attention_mask = self.torch.ones_like(inputs)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class ContextEchoLLM:
    """Test backend: no model, returns a deterministic context summary."""

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 120) -> str:
        del max_new_tokens
        return f"[test backend] received {len(messages)} messages; current={messages[-1]['content']!r}"

