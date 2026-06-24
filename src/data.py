"""Dataset download, exploration, and preprocessing."""
from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    DATASET_FILENAME,
    DATASET_URL,
    DROP_COLUMNS,
    MAX_SAMPLES_PER_CLASS,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    TEST_SIZE,
    TOP_FEATURES,
    VAL_SIZE,
)


def download_dataset() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / DATASET_FILENAME
    if csv_path.exists() and csv_path.stat().st_size > 1_000_000:
        return csv_path

    print(f"Downloading {DATASET_URL}")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(DATASET_URL, context=ctx, timeout=180) as resp:
        csv_path.write_bytes(resp.read())
    return csv_path


def explore_dataset(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if " Label" in df.columns:
        df.rename(columns={" Label": "Label"}, inplace=True)

    label_counts = df["Label"].astype(str).str.strip().value_counts().to_dict()
    numeric_cols = [
        c for c in df.columns if c not in DROP_COLUMNS + ["Label"]
    ]
    missing = int(df[numeric_cols].isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    profile = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "feature_count": len(numeric_cols),
        "missing_values": missing,
        "duplicate_rows": duplicates,
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "attack_types": [k for k in label_counts if str(k).upper() != "BENIGN"],
        "classification_task": "binary (Benign vs Attack)",
    }
    return profile


def preprocess(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if " Label" in df.columns:
        df.rename(columns={" Label": "Label"}, inplace=True)

    df["Label"] = df["Label"].astype(str).str.strip()
    df["binary_label"] = np.where(df["Label"].str.upper() == "BENIGN", 0, 1)

    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS + ["binary_label", "Label"]]
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    frames = []
    for label_value in [0, 1]:
        subset = x[df["binary_label"] == label_value]
        if len(subset) > MAX_SAMPLES_PER_CLASS:
            subset = subset.sample(MAX_SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
        frames.append(subset)
    x_balanced = pd.concat(frames, axis=0)
    y_balanced = df.loc[x_balanced.index, "binary_label"].values

    selected = x_balanced.var().sort_values(ascending=False).head(TOP_FEATURES).index.tolist()
    x_selected = x_balanced[selected]

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_selected.values,
        y_balanced,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_balanced,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": selected,
        "scaler": scaler,
    }
    joblib.dump(bundle, PROCESSED_DIR / "processed_data.joblib")
    return bundle