"""Generate predictions using saved champion model artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

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


def load_predictor_class():
    """Loads the credit score predictor class.

    Returns:
        Credit score predictor class.
    """

    try:
        from creditscore.inference import CreditScorePredictor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependencia ausente ao carregar a predicao. "
            "Ative a venv do projeto ou instale as dependencias com "
            "`pip install -r requirements.txt`."
        ) from exc
    return CreditScorePredictor


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed arguments with artifact path, input path and output path.
    """

    parser = argparse.ArgumentParser(description="Predict credit default risk.")
    parser.add_argument(
        "--artifact",
        default="models/importancexgboost_top50.joblib",
        help="Path to a saved .joblib model artifact.",
    )
    parser.add_argument("--input", default="data/train.csv", help="CSV file to score.")
    parser.add_argument("--output", default="reports/prediction_examples.csv", help="Output CSV.")
    parser.add_argument("--n-rows", type=int, default=20, help="Number of rows to score.")
    return parser.parse_args()


def main() -> None:
    """Loads a model artifact and writes prediction examples."""

    args = parse_args()
    input_path = resolve_project_path(args.input)
    artifact_path = resolve_project_path(args.artifact)
    output_path = resolve_project_path(args.output)

    df_raw = pd.read_csv(input_path).head(args.n_rows)
    predictor_class = load_predictor_class()
    predictor = predictor_class(artifact_path)
    predictions = predictor.predict(df_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
