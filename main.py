import logging
import sys
from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import argparse

from datetime import datetime



def setup_logging(log_path: Path) -> None:
    """Log vers console + fichier outputs/run.log."""
    log_path.parent.mkdir(exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # éviter les doublons si relance
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # logs fichier
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    # logs console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)


def to_jsonable(x):
    """Convertit récursivement certains types (ex: numpy scalars) en types JSON."""
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if hasattr(x, "item"):  # numpy scalars
        return x.item()
    return x


def parse_args():
    parser = argparse.ArgumentParser(description="Logistic regression training")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    # Création d'un run_id basé sur la date/heure
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Paths (artefacts) ---
    base_output_dir = Path(args.output_dir)

    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    runs_csv_path = base_output_dir / "runs.csv"


    model_path = run_dir / "model.joblib"
    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "predictions.csv"
    log_path = run_dir / "run.log"

    setup_logging(log_path)
    logging.info("Starting training run")

    # Sauvegarde de la configuration du run (traçabilité)
    run_config_path = run_dir / "run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_size": args.test_size,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    # 1) Charger dataset (pandas)
    bunch = load_breast_cancer(as_frame=True)
    X: pd.DataFrame = bunch.data
    y: pd.Series = bunch.target
    logging.info("Loaded dataset: X=%s, y=%s", X.shape, y.shape)

    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X,y,
    test_size=args.test_size,
    random_state=args.seed,
    stratify=y,
)
    logging.info("Using test_size=%s, seed=%s", args.test_size, args.seed)

    logging.info("Split: train=%d, test=%d", len(X_train), len(X_test))

    # 3) Modèle
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    # 4) Entraîner
    model.fit(X_train, y_train)
    logging.info("Model trained")

    # 5) Évaluer
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    logging.info("Accuracy: %.4f", acc)
    logging.info("Classification report:\n%s", report_str)

    # 6) Sauvegarder le modèle
    joblib.dump(model, model_path)
    logging.info("Saved model to: %s", model_path)

    # 7) Sauvegarder des métriques (json) - sérialisable
    metrics = {
        "accuracy": float(acc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model": "Pipeline(StandardScaler -> LogisticRegression)",
        "config": {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
        },
        "logreg_params": model.named_steps["clf"].get_params(),
        "classification_report": report_dict,
    }
    metrics = to_jsonable(metrics)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Saved metrics to: %s", metrics_path)

    # 8) Sauvegarder un fichier de prédictions (pandas)
    preds_df = pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "y_pred": pd.Series(y_pred),
        }
    )
    preds_df.to_csv(preds_path, index=False)
    logging.info("Saved predictions to: %s", preds_path)


    # 9) Mettre à jour un résumé global des runs (outputs/runs.csv)
    run_row = pd.DataFrame([{
        "run_id": run_id,
        "test_size": args.test_size,
        "seed": args.seed,
        "accuracy": float(acc),
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }])

    if runs_csv_path.exists():
        existing = pd.read_csv(runs_csv_path)
        updated = pd.concat([existing, run_row], ignore_index=True)
    else:
        updated = run_row

    updated.to_csv(runs_csv_path, index=False)
    logging.info("Updated runs summary: %s", runs_csv_path)


    logging.info("Run finished successfully")


if __name__ == "__main__":
    main()
