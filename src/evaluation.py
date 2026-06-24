"""Model evaluation and metric computation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(model, x: np.ndarray, y: np.ndarray) -> dict:
    preds = model.predict(x)
    proba = None
    try:
        proba = model.predict_proba(x)[:, 1]
    except Exception:
        pass

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }
    if proba is not None and len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
    else:
        metrics["roc_auc"] = None
    return metrics


def pct(value: float) -> float:
    return round(value * 100, 1)