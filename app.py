"""API FastAPI : expose la fonction predict du modèle (Atelier 4)."""

import os

import pandas as pd
from fastapi import FastAPI, HTTPException

from model_pipeline import load_model, predict_model

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

app = FastAPI(title="Adult Census Income API")
model = load_model(MODEL_PATH)  # chargé une seule fois au démarrage
FEATURES = list(model.feature_names_in_)


@app.get("/features")
def features_list():
    """Retourne la liste des colonnes attendues par le modèle."""
    return {"features": FEATURES}


@app.post("/predict")
def predict(features: dict):
    """Effectue une prédiction à partir des caractéristiques fournies.

    Args:
        features (dict): {nom_colonne: valeur} (colonnes absentes = 0).

    Returns:
        dict: {"prediction": 0 ou 1} (1 = revenu > 50K).

    Raises:
        HTTPException: si une colonne est inconnue ou en cas d'erreur.
    """
    unknown = set(features) - set(FEATURES)
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"Colonnes inconnues : {sorted(unknown)}"
        )
    try:
        X = pd.DataFrame([features]).reindex(columns=FEATURES, fill_value=0)
        pred = predict_model(model, X)
        return {"prediction": int(pred[0])}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
