"""FGSM and PGD adversarial attacks."""
from __future__ import annotations

import numpy as np

from .config import PGD_STEP_SIZE, PGD_STEPS


def predict_proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    scores = model.decision_function(x)
    if scores.ndim == 1:
        p1 = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - p1, p1])
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def clip_perturbation(x_adv: np.ndarray, x_orig: np.ndarray, epsilon: float) -> np.ndarray:
    delta = np.clip(x_adv - x_orig, -epsilon, epsilon)
    return x_orig + delta


def loss_gradient(model, xi: np.ndarray, yi: int) -> np.ndarray:
    grad = np.zeros_like(xi)
    step = 1e-3
    base = predict_proba(model, xi)[0, yi]
    for j in range(xi.shape[1]):
        xp = xi.copy()
        xp[0, j] += step
        grad[0, j] = (predict_proba(model, xp)[0, yi] - base) / step
    return grad


def fgsm_attack(model, x: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    x_adv = x.copy()
    for i in range(len(x)):
        xi = x[i : i + 1]
        grad = loss_gradient(model, xi, int(y[i]))
        x_adv[i] = xi[0] - epsilon * np.sign(grad[0])
    return clip_perturbation(x_adv, x, epsilon)


def pgd_attack(
    model,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    steps: int = PGD_STEPS,
    step_size: float = PGD_STEP_SIZE,
) -> np.ndarray:
    x_adv = x.copy()
    for i in range(len(x)):
        xi = x[i : i + 1].copy()
        yi = int(y[i])
        for _ in range(steps):
            grad = loss_gradient(model, xi, yi)
            xi = xi - step_size * np.sign(grad)
            xi = clip_perturbation(xi, x[i : i + 1], epsilon)
        x_adv[i] = xi[0]
    return x_adv