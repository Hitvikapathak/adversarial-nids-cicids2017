"""Baseline model training."""
from __future__ import annotations

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from .config import MODELS_DIR, RANDOM_SEED


def build_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=25,
            early_stopping=True,
            random_state=RANDOM_SEED,
        ),
    }


def train_models(x_train, y_train) -> dict:
    models = build_models()
    for model in models.values():
        model.fit(x_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODELS_DIR / "baseline_models.joblib")
    return models