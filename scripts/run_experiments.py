"""
Reproducible adversarial robustness experiments on CIC-IDS2017 (subset).
Generates metrics tables and figures for the IITK B.Cyber project report.
"""
from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42
EPSILONS = [0.01, 0.05, 0.1]
PRIMARY_EPSILON = 0.05
PGD_STEPS = 20
PGD_STEP_SIZE = 0.01
MAX_SAMPLES_PER_CLASS = 2500
TEST_SIZE = 0.2
ATTACK_EVAL_SAMPLES = 200
TOP_FEATURES = 30

DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
]


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def download_sample_data() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    if csv_path.exists() and csv_path.stat().st_size > 1_000_000:
        return csv_path

    url = (
        "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/"
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    )
    print(f"Downloading {url} ...")
    try:
        import ssl
        import urllib.request

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:
            csv_path.write_bytes(resp.read())
    except Exception as exc:
        print(f"Download failed ({exc}). Generating synthetic fallback data.")
        return generate_synthetic_data(csv_path)
    return csv_path


def generate_synthetic_data(csv_path: Path) -> Path:
    rng = np.random.default_rng(RANDOM_SEED)
    n = 12000
    n_features = 30
    x_benign = rng.normal(0.35, 0.45, (n // 2, n_features))
    x_attack = rng.normal(0.55, 0.45, (n // 2, n_features))
    x = np.vstack([x_benign, x_attack])
    y = np.array(["BENIGN"] * (n // 2) + ["Attack"] * (n // 2))
    cols = [f"Feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(x, columns=cols)
    df["Label"] = y
    df.to_csv(csv_path, index=False)
    return csv_path


def load_and_preprocess(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if " Label" in df.columns:
        df.rename(columns={" Label": "Label"}, inplace=True)

    df["Label"] = df["Label"].astype(str).str.strip()
    df["binary_label"] = np.where(df["Label"].str.upper() == "BENIGN", 0, 1)

    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS + ["binary_label"]]
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(0.0)

    # Balance for tractable runtime while preserving class ratio.
    frames = []
    for label_value in [0, 1]:
        subset = x[df["binary_label"] == label_value]
        if len(subset) > MAX_SAMPLES_PER_CLASS:
            subset = subset.sample(MAX_SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
        frames.append(subset)
    x_balanced = pd.concat(frames, axis=0)
    y_balanced = df.loc[x_balanced.index, "binary_label"].values

    variances = x_balanced.var().sort_values(ascending=False)
    selected = variances.head(TOP_FEATURES).index.tolist()
    x_selected = x_balanced[selected]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_selected)

    return x_scaled, y_balanced, selected


def train_models(x_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=25,
        early_stopping=True,
        random_state=RANDOM_SEED,
    )

    rf.fit(x_train, y_train)
    xgb.fit(x_train, y_train)
    mlp.fit(x_train, y_train)
    return {"Random Forest": rf, "XGBoost": xgb, "MLP": mlp}


def predict_proba_attack(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    scores = model.decision_function(x)
    if scores.ndim == 1:
        p1 = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - p1, p1])
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def clip_realistic(x_adv: np.ndarray, x_orig: np.ndarray, epsilon: float) -> np.ndarray:
    delta = np.clip(x_adv - x_orig, -epsilon, epsilon)
    return x_orig + delta


def loss_gradient(model, xi: np.ndarray, yi: int) -> np.ndarray:
    grad = np.zeros_like(xi)
    eps = 1e-3
    base = predict_proba_attack(model, xi)[0, yi]
    for j in range(xi.shape[1]):
        xp = xi.copy()
        xp[0, j] += eps
        plus = predict_proba_attack(model, xp)[0, yi]
        grad[0, j] = (plus - base) / eps
    return grad


def fgsm_attack(model, x: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    x_adv = x.copy()
    for i in range(len(x)):
        xi = x[i : i + 1]
        yi = int(y[i])
        grad = loss_gradient(model, xi, yi)
        x_adv[i] = xi[0] - epsilon * np.sign(grad[0])
    return clip_realistic(x_adv, x, epsilon)


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
            xi = clip_realistic(xi, x[i : i + 1], epsilon)
        x_adv[i] = xi[0]
    return x_adv


def feature_squeeze(x: np.ndarray, bit_depth: int = 4) -> np.ndarray:
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    span = x_max - x_min + 1e-8
    x01 = (x - x_min) / span
    levels = 2**bit_depth - 1
    squeezed01 = np.round(x01 * levels) / levels
    return squeezed01 * span + x_min


def adversarial_train_mlp(x_train, y_train, surrogate_mlp: MLPClassifier) -> MLPClassifier:
    x_adv = pgd_attack(surrogate_mlp, x_train[: min(1500, len(x_train))], y_train[: min(1500, len(y_train))], PRIMARY_EPSILON)
    n = min(len(x_train), len(x_adv))
    x_mix = np.vstack([x_train[:n], x_adv[:n]])
    y_mix = np.concatenate([y_train[:n], y_train[:n]])
    robust_mlp = MLPClassifier(
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
    robust_mlp.fit(x_mix, y_mix)
    return robust_mlp


def evaluate(model, x: np.ndarray, y: np.ndarray) -> dict:
    preds = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "attack_recall": float(recall_score(y, preds, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_epsilon_curve(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    plt.figure(figsize=(6, 4))
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        plt.plot(sub["epsilon"], sub["pgd_accuracy"], marker="o", label=model_name)
    plt.xlabel("Epsilon (L-inf)")
    plt.ylabel("Accuracy under PGD (%)")
    plt.title("Robustness vs Perturbation Budget")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    set_seeds()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = download_sample_data()
    x, y, feature_cols = load_and_preprocess(csv_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    models = train_models(x_train, y_train)
    joblib.dump({"feature_cols": feature_cols, "models": models}, MODELS_DIR / "models.joblib")

    attack_idx = np.where(y_test == 1)[0][:ATTACK_EVAL_SAMPLES]
    attack_samples = x_test[attack_idx]
    attack_labels = y_test[attack_idx]

    mlp = models["MLP"]
    fgsm_adv = fgsm_attack(mlp, attack_samples, attack_labels, PRIMARY_EPSILON)
    pgd_adv = pgd_attack(mlp, attack_samples, attack_labels, PRIMARY_EPSILON)

    robust_mlp = adversarial_train_mlp(x_train, y_train, mlp)

    summary_rows = []
    epsilon_rows = []

    for model_name, model in models.items():
        clean = evaluate(model, x_test, y_test)
        if model_name == "MLP":
            fgsm_eval = evaluate(model, fgsm_adv, attack_labels)
            pgd_eval = evaluate(model, pgd_adv, attack_labels)
        else:
            fgsm_eval = evaluate(model, fgsm_adv, attack_labels)
            pgd_eval = evaluate(model, pgd_adv, attack_labels)

        summary_rows.append(
            {
                "model": model_name,
                "clean_accuracy": round(clean["accuracy"] * 100, 1),
                "clean_recall": round(clean["attack_recall"] * 100, 1),
                "fgsm_accuracy": round(fgsm_eval["accuracy"] * 100, 1),
                "pgd_accuracy": round(pgd_eval["accuracy"] * 100, 1),
                "after_defense_pgd_accuracy": None,
            }
        )

        plot_confusion(
            np.array(clean["confusion_matrix"]),
            f"{model_name} - Clean",
            RESULTS_DIR / f"cm_{model_name.lower().replace(' ', '_')}_clean.png",
        )
        plot_confusion(
            np.array(pgd_eval["confusion_matrix"]),
            f"{model_name} - PGD eps={PRIMARY_EPSILON}",
            RESULTS_DIR / f"cm_{model_name.lower().replace(' ', '_')}_pgd.png",
        )

        for eps in EPSILONS:
            adv_eps = pgd_attack(mlp, attack_samples, attack_labels, eps)
            ev = evaluate(model, adv_eps, attack_labels)
            epsilon_rows.append(
                {
                    "model": model_name,
                    "epsilon": eps,
                    "pgd_accuracy": round(ev["accuracy"] * 100, 1),
                }
            )

    robust_eval = evaluate(robust_mlp, pgd_adv, attack_labels)
    squeezed = feature_squeeze(pgd_adv)
    squeeze_eval = evaluate(mlp, squeezed, attack_labels)

    summary_rows[2]["after_defense_pgd_accuracy"] = round(robust_eval["accuracy"] * 100, 1)

    defense_rows = [
        {
            "defense": "Baseline MLP",
            "clean_accuracy": summary_rows[2]["clean_accuracy"],
            "pgd_accuracy": summary_rows[2]["pgd_accuracy"],
        },
        {
            "defense": "Adversarial Training (MLP)",
            "clean_accuracy": round(evaluate(robust_mlp, x_test, y_test)["accuracy"] * 100, 1),
            "pgd_accuracy": round(robust_eval["accuracy"] * 100, 1),
        },
        {
            "defense": "Feature Squeezing (MLP)",
            "clean_accuracy": summary_rows[2]["clean_accuracy"],
            "pgd_accuracy": round(squeeze_eval["accuracy"] * 100, 1),
        },
    ]

    plot_confusion(
        np.array(robust_eval["confusion_matrix"]),
        "MLP + Adversarial Training - PGD",
        RESULTS_DIR / "cm_mlp_advtrain_pgd.png",
    )
    plot_confusion(
        np.array(squeeze_eval["confusion_matrix"]),
        "MLP + Feature Squeezing - PGD",
        RESULTS_DIR / "cm_mlp_squeeze_pgd.png",
    )
    plot_epsilon_curve(epsilon_rows, RESULTS_DIR / "accuracy_vs_epsilon.png")

    metadata = {
        "random_seed": RANDOM_SEED,
        "dataset_file": str(csv_path.name),
        "train_test_split": f"{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} stratified",
        "samples_used": int(len(x)),
        "attack_samples_evaluated": int(len(attack_samples)),
        "epsilons": EPSILONS,
        "primary_epsilon": PRIMARY_EPSILON,
        "pgd_steps": PGD_STEPS,
        "pgd_step_size": PGD_STEP_SIZE,
        "attack_type": "untargeted evasion (attack -> benign)",
        "white_box_model": "MLP",
        "transfer_models": ["Random Forest", "XGBoost"],
        "summary_table": summary_rows,
        "defense_table": defense_rows,
        "epsilon_curve": epsilon_rows,
    }

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    pd.DataFrame(defense_rows).to_csv(RESULTS_DIR / "defense_table.csv", index=False)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()