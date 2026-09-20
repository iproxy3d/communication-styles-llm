from __future__ import annotations

from pathlib import Path
from typing import Protocol


DEFAULT_GENERATION_TEMPERATURE = 0.7
DEFAULT_GENERATION_TOP_P = 0.9
DEFAULT_GENERATION_TOP_K = 50


class ChatModel(Protocol):
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 120) -> str: ...


class LocalTransformersLLM:
    """Loads model weights from a local directory; performs no API calls."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        temperature: float = DEFAULT_GENERATION_TEMPERATURE,
        top_p: float = DEFAULT_GENERATION_TOP_P,
        top_k: int = DEFAULT_GENERATION_TOP_K,
        repetition_penalty: float = 1.05,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if temperature < 0.7:
            raise ValueError("temperature must be >= 0.7 for the style demo")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be > 0")

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Local model not found at {path}. Run: python download_model.py"
            )
        self.torch = torch
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, attn_implementation="sdpa", local_files_only=True, torch_dtype=dtype
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
        # Style adherence needs actual sampling: with greedy decoding the
        # temperature is ignored and a small model tends to collapse different
        # character prompts into very similar continuations.
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=pad_token_id,
            )
        new_tokens = output[0, inputs.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class ContextEchoLLM:
    """Test backend: no model, returns a deterministic context summary."""

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 120) -> str:
        del max_new_tokens
        return f"[test backend] received {len(messages)} messages; current={messages[-1]['content']!r}"
