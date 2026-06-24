"""End-to-end adversarial robustness pipeline."""
from __future__ import annotations

import io
import json
import random
import sys
from contextlib import redirect_stdout

import joblib
import numpy as np

from .attacks import fgsm_attack, pgd_attack
from .config import (
    ATTACK_EVAL_SAMPLES,
    EPSILONS,
    PRIMARY_EPSILON,
    PROCESSED_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SCREENSHOTS_DIR,
)
from .data import download_dataset, explore_dataset, preprocess
from .defenses import EnsembleDefense, adversarial_train_mlp, feature_squeeze
from .evaluation import compute_metrics, pct
from .models import train_models
from .visualization import (
    plot_accuracy_drop,
    plot_confusion,
    plot_epsilon_curve,
    save_terminal_screenshot,
)


def set_seeds() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def run() -> dict:
    set_seeds()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("=== Adversarial NIDS Pipeline ===")
        csv_path = download_dataset()
        dataset_profile = explore_dataset(csv_path)
        print("Dataset profile:", json.dumps(dataset_profile, indent=2))

        bundle = preprocess(csv_path)
        models = train_models(bundle["x_train"], bundle["y_train"])

        attack_idx = np.where(bundle["y_test"] == 1)[0][:ATTACK_EVAL_SAMPLES]
        attack_x = bundle["x_test"][attack_idx]
        attack_y = bundle["y_test"][attack_idx]

        mlp = models["MLP"]
        fgsm_adv = fgsm_attack(mlp, attack_x, attack_y, PRIMARY_EPSILON)
        pgd_adv = pgd_attack(mlp, attack_x, attack_y, PRIMARY_EPSILON)

        robust_mlp = adversarial_train_mlp(bundle["x_train"], bundle["y_train"], mlp)
        ensemble = EnsembleDefense(models["Random Forest"], robust_mlp)

        summary_rows = []
        baseline_detail = []
        epsilon_rows = []

        for name, model in models.items():
            clean = compute_metrics(model, bundle["x_test"], bundle["y_test"])
            fgsm_m = compute_metrics(model, fgsm_adv, attack_y)
            pgd_m = compute_metrics(model, pgd_adv, attack_y)

            row = {
                "model": name,
                "clean_accuracy": pct(clean["accuracy"]),
                "clean_precision": pct(clean["precision"]),
                "clean_recall": pct(clean["recall"]),
                "clean_f1": pct(clean["f1"]),
                "clean_roc_auc": None if clean["roc_auc"] is None else pct(clean["roc_auc"]),
                "fgsm_detection_rate": pct(fgsm_m["recall"]),
                "pgd_detection_rate": pct(pgd_m["recall"]),
                "after_defense_pgd_rate": None,
            }
            summary_rows.append(row)
            baseline_detail.append({"model": name, "clean": clean, "pgd_on_attacks": pgd_m})

            plot_confusion(
                np.array(clean["confusion_matrix"]),
                f"{name} — Clean Test Set",
                RESULTS_DIR / f"cm_{name.lower().replace(' ', '_')}_clean.png",
            )
            plot_confusion(
                np.array(pgd_m["confusion_matrix"]),
                f"{name} — Transferred PGD (eps={PRIMARY_EPSILON})",
                RESULTS_DIR / f"cm_{name.lower().replace(' ', '_')}_pgd.png",
            )

            for eps in EPSILONS:
                adv = pgd_attack(mlp, attack_x, attack_y, eps)
                ev = compute_metrics(model, adv, attack_y)
                epsilon_rows.append(
                    {"model": name, "epsilon": eps, "attack_detection_rate": pct(ev["recall"])}
                )

        robust_clean = compute_metrics(robust_mlp, bundle["x_test"], bundle["y_test"])
        robust_pgd = compute_metrics(robust_mlp, pgd_adv, attack_y)
        squeeze_pgd = compute_metrics(mlp, feature_squeeze(pgd_adv), attack_y)
        ensemble_pgd = compute_metrics(ensemble, pgd_adv, attack_y)

        summary_rows[2]["after_defense_pgd_rate"] = pct(robust_pgd["recall"])

        defense_rows = [
            {
                "defense": "Baseline MLP",
                "clean_accuracy": pct(compute_metrics(mlp, bundle["x_test"], bundle["y_test"])["accuracy"]),
                "pgd_detection_rate": pct(compute_metrics(mlp, pgd_adv, attack_y)["recall"]),
                "compute_cost": "Low",
            },
            {
                "defense": "Adversarial Training (MLP)",
                "clean_accuracy": pct(robust_clean["accuracy"]),
                "pgd_detection_rate": pct(robust_pgd["recall"]),
                "compute_cost": "High (retraining)",
            },
            {
                "defense": "Feature Squeezing (MLP)",
                "clean_accuracy": pct(compute_metrics(mlp, bundle["x_test"], bundle["y_test"])["accuracy"]),
                "pgd_detection_rate": pct(squeeze_pgd["recall"]),
                "compute_cost": "Low",
            },
            {
                "defense": "Ensemble (RF + Robust MLP)",
                "clean_accuracy": pct(compute_metrics(ensemble, bundle["x_test"], bundle["y_test"])["accuracy"]),
                "pgd_detection_rate": pct(ensemble_pgd["recall"]),
                "compute_cost": "Medium",
            },
        ]

        plot_confusion(
            np.array(robust_pgd["confusion_matrix"]),
            "Robust MLP — PGD",
            RESULTS_DIR / "cm_mlp_advtrain_pgd.png",
        )
        plot_confusion(
            np.array(squeeze_pgd["confusion_matrix"]),
            "Feature Squeezing — PGD",
            RESULTS_DIR / "cm_mlp_squeeze_pgd.png",
        )
        plot_epsilon_curve(epsilon_rows, RESULTS_DIR / "accuracy_vs_epsilon.png")
        plot_accuracy_drop(summary_rows, RESULTS_DIR / "accuracy_drop_comparison.png")

        joblib.dump(
            {
                "models": models,
                "robust_mlp": robust_mlp,
                "ensemble": ensemble,
                "feature_names": bundle["feature_names"],
            },
            RESULTS_DIR.parent / "models" / "all_models.joblib",
        )

        metadata = {
            "project_title": "Evaluating and Enhancing Adversarial Robustness of ML Models for NIDS",
            "random_seed": RANDOM_SEED,
            "dataset_profile": dataset_profile,
            "dataset_file": csv_path.name,
            "splits": "72% train / 8% val / 20% test (stratified)",
            "samples_after_balancing": int(len(bundle["y_train"]) + len(bundle["y_val"]) + len(bundle["y_test"])),
            "attack_samples_evaluated": int(len(attack_x)),
            "epsilons": EPSILONS,
            "primary_epsilon": PRIMARY_EPSILON,
            "pgd_steps": 20,
            "pgd_step_size": 0.01,
            "attack_objective": "untargeted evasion (attack classified as benign)",
            "white_box_model": "MLP",
            "transfer_models": ["Random Forest", "XGBoost"],
            "summary_table": summary_rows,
            "defense_table": defense_rows,
            "epsilon_curve": epsilon_rows,
            "baseline_detail": baseline_detail,
        }

        (RESULTS_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (PROCESSED_DIR / "dataset_profile.json").write_text(
            json.dumps(dataset_profile, indent=2), encoding="utf-8"
        )

        import pandas as pd

        pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "summary_table.csv", index=False)
        pd.DataFrame(defense_rows).to_csv(RESULTS_DIR / "defense_table.csv", index=False)

        print("\n=== Summary Table ===")
        print(pd.DataFrame(summary_rows).to_string(index=False))
        print("\n=== Defense Table ===")
        print(pd.DataFrame(defense_rows).to_string(index=False))
        print("\nArtifacts written to:", RESULTS_DIR)

    terminal_text = buf.getvalue()
    (SCREENSHOTS_DIR / "terminal_output.txt").write_text(terminal_text, encoding="utf-8")
    save_terminal_screenshot(terminal_text[:3500], SCREENSHOTS_DIR / "terminal_output.png")
    print(terminal_text)
    return metadata


if __name__ == "__main__":
    run()