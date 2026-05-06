from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

from .config import Config
from .features_f1 import F1_COLUMNS


def _metric_row(model_name: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "model": model_name,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_precision": w_precision,
        "weighted_recall": w_recall,
        "weighted_f1": w_f1,
        "n_samples": len(y_true),
    }


def _available_models(seed: int, enable_xgboost: bool = False) -> dict:
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=3.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    if enable_xgboost:
        try:
            from xgboost import XGBClassifier
            models["xgboost_optional"] = XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=1,
            )
        except Exception:
            pass
    return models


def train_and_evaluate(cfg: Config, f1_path: str | Path) -> dict:
    out_dir = Path(cfg.output_dir)
    model_dir = out_dir / "models"
    cm_dir = out_dir / "confusion_matrices"
    model_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(f1_path)
    X = df[F1_COLUMNS].astype(float).to_numpy()
    y_raw = df["label"].astype(int).to_numpy()

    # xgboost prefers labels starting from 0. Other sklearn models can use 1/2/3.
    label_values = sorted(np.unique(y_raw).tolist())
    label_to_zero = {lab: i for i, lab in enumerate(label_values)}
    zero_to_label = {i: lab for lab, i in label_to_zero.items()}

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=cfg.test_size,
        random_state=cfg.model_random_seed,
        stratify=y_raw,
    )

    metrics = []
    for name, model in _available_models(cfg.model_random_seed, cfg.enable_xgboost).items():
        y_train = y_raw[train_idx]
        y_test = y_raw[test_idx]
        if name == "xgboost_optional":
            y_fit = np.array([label_to_zero[v] for v in y_train])
            model.fit(X[train_idx], y_fit)
            pred_train_zero = model.predict(X[train_idx])
            pred_test_zero = model.predict(X[test_idx])
            pred_train = np.array([zero_to_label[int(v)] for v in pred_train_zero])
            pred_test = np.array([zero_to_label[int(v)] for v in pred_test_zero])
        else:
            model.fit(X[train_idx], y_train)
            pred_train = model.predict(X[train_idx])
            pred_test = model.predict(X[test_idx])

        metrics.append(_metric_row(name, "train", y_train, pred_train))
        metrics.append(_metric_row(name, "test", y_test, pred_test))

        cm = confusion_matrix(y_test, pred_test, labels=label_values)
        cm_df = pd.DataFrame(cm, index=[f"true_{v}" for v in label_values], columns=[f"pred_{v}" for v in label_values])
        cm_df.to_csv(cm_dir / f"{name}_test_confusion_matrix.csv", encoding="utf-8-sig")
        joblib.dump(model, model_dir / f"{name}.joblib")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "metrics_path": str(out_dir / "metrics_summary.csv"),
        "model_dir": str(model_dir),
        "confusion_matrix_dir": str(cm_dir),
        "n_features": len(F1_COLUMNS),
        "n_samples": len(df),
    }
