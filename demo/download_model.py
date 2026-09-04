from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET = Path(__file__).parent / "models" / "qwen2.5-0.5b-instruct"


def main() -> None:
    if (TARGET / "config.json").exists():
        print(f"Model already exists: {TARGET}")
        return
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} once. Inference will run locally afterwards.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(TARGET)
    model.save_pretrained(TARGET, safe_serialization=True)
    print(f"Local model ready: {TARGET}")


if __name__ == "__main__":
    main()

