"""Train and save the two champion Credit Score model artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def resolve_project_path(path_value: str) -> Path:
    """Resolves relative paths from the project root.

    Args:
        path_value: Absolute path or project-relative path.

    Returns:
        Absolute path resolved from the repository root.
    """

    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_trainer_class():
    """Loads the champion model trainer class.

    Returns:
        Champion model trainer class.
    """

    try:
        from creditscore.training import ChampionModelTrainer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependencia ausente ao carregar o treinamento. "
            "Ative a venv do projeto ou instale as dependencias com "
            "`pip install -r requirements.txt`."
        ) from exc
    return ChampionModelTrainer


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed arguments with train path and output directory.
    """

    parser = argparse.ArgumentParser(description="Train champion Credit Score models.")
    parser.add_argument("--train-path", default="data/train.csv", help="Path to train.csv.")
    parser.add_argument("--output-dir", default="models", help="Directory for .joblib artifacts.")
    return parser.parse_args()


def main() -> None:
    """Runs final model training and saves artifacts."""

    args = parse_args()
    trainer_class = load_trainer_class()
    trainer = trainer_class()
    saved_paths = trainer.train_and_save_all(
        train_path=resolve_project_path(args.train_path),
        output_dir=resolve_project_path(args.output_dir),
    )
    print("Saved model artifacts:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
