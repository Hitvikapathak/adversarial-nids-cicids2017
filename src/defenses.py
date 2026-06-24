"""Defense mechanisms: adversarial training, feature squeezing, ensemble."""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier

from .attacks import pgd_attack
from .config import PRIMARY_EPSILON, RANDOM_SEED


def feature_squeeze(x: np.ndarray, bit_depth: int = 4) -> np.ndarray:
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    span = x_max - x_min + 1e-8
    x01 = (x - x_min) / span
    levels = 2**bit_depth - 1
    squeezed01 = np.round(x01 * levels) / levels
    return squeezed01 * span + x_min


def adversarial_train_mlp(x_train, y_train, surrogate_mlp) -> MLPClassifier:
    n = min(1500, len(x_train))
    x_adv = pgd_attack(surrogate_mlp, x_train[:n], y_train[:n], PRIMARY_EPSILON)
    x_mix = np.vstack([x_train[:n], x_adv])
    y_mix = np.concatenate([y_train[:n], y_train[:n]])
    robust = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=30,
        early_stopping=True,
        random_state=RANDOM_SEED,
    )
    robust.fit(x_mix, y_mix)
    return robust


class EnsembleDefense:
    """Majority vote between tree model and MLP for complementary robustness."""

    def __init__(self, tree_model, mlp_model):
        self.tree_model = tree_model
        self.mlp_model = mlp_model

    def predict(self, x: np.ndarray) -> np.ndarray:
        p_tree = self.tree_model.predict(x)
        p_mlp = self.mlp_model.predict(x)
        return np.where((p_tree + p_mlp) >= 1, 1, 0)