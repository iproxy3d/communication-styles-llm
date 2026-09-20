from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent

GENERATION_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GENERATION_TARGET = ROOT / "models" / "qwen2.5-0.5b-instruct"

EMOTION_MODEL_ID = "proxy3d/multi-motions-28"
EMOTION_TARGET = ROOT / "models" / "multi-motions-28"


def download(repo_id: str, target: Path) -> None:
    if (target / "config.json").exists() and any(
        target.glob("*.safetensors")
    ):
        print(f"Already present: {repo_id} -> {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        ignore_patterns=(
            "*.md",
            "*.html",
            "*.bin",  # both models provide safetensors
            "docs/*",
            "examples/*",
            "training_*",
            "trainer_*",
            "evaluation*",
            "native_ru_evaluation*",
        ),
    )


def main() -> None:
    download(GENERATION_MODEL_ID, GENERATION_TARGET)
    download(EMOTION_MODEL_ID, EMOTION_TARGET)
    print("\nBoth models are ready. Runtime inference is local and uses no LLM API.")


if __name__ == "__main__":
    main()
