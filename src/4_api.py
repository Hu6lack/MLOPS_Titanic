"""API REST FastAPI avec métriques Prometheus pour le monitoring Grafana."""

# Exécution : uvicorn src.4_api:app --reload
# Docs      : http://localhost:8000/docs
# Métriques : http://localhost:8000/metrics  (scrapées par Prometheus)

import time
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

app = FastAPI(title="Titanic Survival API", version="1.0.0")

# ── Métriques Prometheus ─────────────────────────────────────────────────────
REQUEST_COUNT   = Counter("api_requests_total",    "Nombre total de requêtes", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latence des requêtes", ["endpoint"])
PREDICTION_COUNT = Counter("predictions_total", "Nombre de prédictions", ["result"])

# ── Chargement du modèle ─────────────────────────────────────────────────────
model: Any = None


class Passenger(BaseModel):
    """Données d'entrée pour la prédiction de survie.

    Attributes:
        Pclass: Classe du billet (1, 2 ou 3).
        Sex: 0=male, 1=female.
        Age: Âge du passager en années.
        SibSp: Nombre de frères/sœurs ou conjoints à bord.
        Parch: Nombre de parents ou enfants à bord.
        Fare: Prix du billet.
        Embarked: Port d'embarquement (0=S, 1=C, 2=Q).
        FamilySize: Taille totale de la famille (SibSp + Parch + 1).
        IsAlone: 1 si le passager voyage seul, 0 sinon.
    """

    Pclass:     int   = Field(..., ge=1, le=3)
    Sex:        int   = Field(..., ge=0, le=1)
    Age:        float = Field(..., ge=0, le=120)
    SibSp:      int   = Field(0, ge=0)
    Parch:      int   = Field(0, ge=0)
    Fare:       float = Field(30.0, ge=0)
    Embarked:   int   = Field(0, ge=0, le=2)
    FamilySize: int   = Field(1, ge=1)
    IsAlone:    int   = Field(1, ge=0, le=1)


@app.on_event("startup")
def load_model() -> None:
    """Charge le modèle ML au démarrage de l'API."""
    global model
    path = Path("models/best_model.pkl")
    if path.exists():
        model = joblib.load(path)
        logger.info("✅ Modèle chargé depuis %s", path)
    else:
        logger.warning("⚠️  Aucun modèle trouvé. Lancez d'abord 3_training.py")


@app.get("/health", summary="État de l'API")
def health() -> Dict[str, Any]:
    """Vérifie que l'API est opérationnelle.

    Returns:
        Statut de l'API et disponibilité du modèle.
    """
    REQUEST_COUNT.labels(endpoint="/health", status="200").inc()
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", summary="Prédire la survie d'un passager")
def predict(p: Passenger) -> Dict[str, Any]:
    """Prédit si un passager du Titanic aurait survécu.

    Args:
        p: Caractéristiques du passager.

    Returns:
        Prédiction (0/1), probabilité et label lisible.

    Raises:
        HTTPException: Si le modèle n'est pas chargé (503).
    """
    if model is None:
        REQUEST_COUNT.labels(endpoint="/predict", status="503").inc()
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    start = time.time()
    df    = pd.DataFrame([p.model_dump()])
    pred  = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])
    elapsed = time.time() - start

    REQUEST_LATENCY.labels(endpoint="/predict").observe(elapsed)
    REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
    PREDICTION_COUNT.labels(result="survived" if pred == 1 else "deceased").inc()

    return {
        "survived":    pred,
        "probability": round(proba, 3),
        "label":       "Survécu ✅" if pred == 1 else "Décédé ❌",
    }


@app.get("/metrics", summary="Métriques Prometheus")
def metrics() -> Response:
    """Expose les métriques au format Prometheus (scrapées par Grafana).

    Returns:
        Réponse texte contenant toutes les métriques.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
