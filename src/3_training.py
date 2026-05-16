"""Module d'entraînement, tracking MLflow et enregistrement au Model Registry."""

# Exécution : python src/3_training.py
# Interface  : mlflow ui  →  http://localhost:5000

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRAIN: Path        = Path("data/train.csv")
TEST: Path         = Path("data/test.csv")
MODELS: Path       = Path("models")
EXPERIMENT: str    = "titanic-mlops"
REGISTRY_NAME: str = "titanic-classifier"

# Seuil de promotion en Production (0.70 est réaliste pour ce dataset)
F1_THRESHOLD: float = 0.70

MODELS_TO_TRY: Dict[str, Any] = {
    "LogisticRegression": LogisticRegression(
        C=0.3,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=1000,
        max_depth=5,
        min_samples_leaf=4,
        random_state=42,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.2,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
}


def compute_metrics(y_true: pd.Series, y_pred: Any, y_proba: Any) -> Dict[str, float]:
    """Calcule les métriques de classification.

    Args:
        y_true: Labels réels.
        y_pred: Prédictions binaires.
        y_proba: Probabilités de la classe positive.

    Returns:
        Dictionnaire des métriques.
    """
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
    }


def train() -> Tuple[Any, str, Dict[str, float]]:
    """Entraîne les modèles, track avec MLflow, enregistre le meilleur au Registry.

    Returns:
        Tuple (meilleur_modèle, nom_modèle, métriques).
    """
    MODELS.mkdir(exist_ok=True)
    mlflow.set_experiment(EXPERIMENT)

    train_df = pd.read_csv(TRAIN)
    test_df  = pd.read_csv(TEST)
    X_train, y_train = train_df.drop(columns=["Survived"]), train_df["Survived"]
    X_test,  y_test  = test_df.drop(columns=["Survived"]),  test_df["Survived"]

    logger.info("Train : %d lignes | Test : %d lignes", len(X_train), len(X_test))
    logger.info("Features : %s", list(X_train.columns))

    best_model:    Any              = None
    best_name:     str              = ""
    best_f1:       float            = -1.0
    best_metrics:  Dict[str, float] = {}
    best_run_id:   str              = ""

    for name, model in MODELS_TO_TRY.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            preds  = model.predict(X_test)
            probas = model.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, preds, probas)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.set_tag("model_type", name)
            # Nouvelle syntaxe MLflow 2.x (name= au lieu de artifact_path=)
            mlflow.sklearn.log_model(model, name="model")

            logger.info(
                "[%s] accuracy=%.4f | f1=%.4f | roc_auc=%.4f",
                name, metrics["accuracy"], metrics["f1_score"], metrics["roc_auc"],
            )

            if metrics["f1_score"] > best_f1:
                best_f1      = metrics["f1_score"]
                best_model   = model
                best_name    = name
                best_metrics = metrics
                best_run_id  = run.info.run_id

    # ── Sauvegarde locale ────────────────────────────────────────
    joblib.dump(best_model, MODELS / "best_model.pkl")
    logger.info("🏆 Meilleur modèle : %s (F1=%.4f)", best_name, best_f1)

    # ── MLflow Model Registry ────────────────────────────────────
    model_uri  = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)
    client     = MlflowClient()

    client.update_registered_model(
        name=REGISTRY_NAME,
        description=f"Prédiction survie Titanic — meilleur algo : {best_name}",
    )

    # Promotion selon le seuil F1
    if best_f1 >= F1_THRESHOLD:
        try:
            # Nouvelle API MLflow >= 2.9 : aliases
            client.set_registered_model_alias(
                name=REGISTRY_NAME,
                alias="production",
                version=registered.version,
            )
            logger.info(
                "🚀 Modèle v%s → alias 'production' (F1=%.4f >= seuil=%.2f)",
                registered.version, best_f1, F1_THRESHOLD,
            )
        except Exception:
            # Fallback MLflow < 2.9 : stages
            client.transition_model_version_stage(  # type: ignore[attr-defined]
                name=REGISTRY_NAME,
                version=registered.version,
                stage="Production",
            )
            logger.info("🚀 Modèle v%s → Production", registered.version)
    else:
        logger.warning(
            "⚠️  F1=%.4f < seuil=%.2f → modèle maintenu en Staging",
            best_f1, F1_THRESHOLD,
        )

    logger.info("\n=== RÉSUMÉ FINAL ===")
    for k, v in best_metrics.items():
        logger.info("  %-12s : %.4f", k, v)

    return best_model, best_name, best_metrics


if __name__ == "__main__":
    train()