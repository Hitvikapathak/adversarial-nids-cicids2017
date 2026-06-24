"""Plot generation for report artifacts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    plt.figure(figsize=(6.5, 4.5))
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        plt.plot(sub["epsilon"], sub["attack_detection_rate"], marker="o", label=model_name)
    plt.xlabel("Epsilon (L-inf)")
    plt.ylabel("Attack Detection Rate (%)")
    plt.title("PGD Robustness vs Perturbation Budget")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_accuracy_drop(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, df["clean_accuracy"], width, label="Clean")
    plt.bar(x + width / 2, df["pgd_detection_rate"], width, label="PGD (eps=0.05)")
    plt.xticks(x, df["model"], rotation=15)
    plt.ylabel("Rate (%)")
    plt.title("Clean vs Adversarial Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_terminal_screenshot(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.text(0.02, 0.98, text, va="top", family="monospace", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor="#1e1e1e")
    plt.close()