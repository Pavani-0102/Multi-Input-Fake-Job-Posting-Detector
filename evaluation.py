"""Metrics, exploratory data analysis and plotting utilities (matplotlib only)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend; works on any machine
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import CLASS_NAMES, FRAUD_LABEL, LABEL_COLUMN, TEXT_FIELDS


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float] | None = None) -> dict[str, Any]:
    """Metrics with the fraudulent class (1) as the positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=FRAUD_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=FRAUD_LABEL, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=FRAUD_LABEL, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
    if y_score is not None and len(set(y_true.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    return metrics


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(cm: Sequence[Sequence[int]], title: str, path: Path) -> str:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=12)
    return _save(fig, path)


def plot_class_distribution(df: pd.DataFrame, path: Path) -> str:
    counts = df[LABEL_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    bars = ax.bar(CLASS_NAMES, counts.values, color=["#4C78A8", "#E45756"])
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values])
    ax.set_ylabel("Postings")
    ax.set_title("Class distribution (after cleaning)")
    return _save(fig, path)


def plot_missing_values(df_raw: pd.DataFrame, path: Path) -> str:
    cols = [c for c in TEXT_FIELDS if c in df_raw.columns]
    pct = (df_raw[cols].isna() | (df_raw[cols].astype(str).apply(lambda s: s.str.strip()) == "")).mean() * 100
    pct = pct.sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(pct.index, pct.values, color="#72B7B2")
    ax.set_xlabel("% missing / empty")
    ax.set_title("Missing values per text field (raw data)")
    return _save(fig, path)


def plot_text_length(df: pd.DataFrame, path: Path) -> str:
    lengths = df["text"].str.split().str.len()
    upper = float(np.percentile(lengths, 99)) if len(lengths) else 1.0
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(0, max(upper, 1), 40)
    ax.hist(lengths[df[LABEL_COLUMN] == 0], bins=bins, alpha=0.65, label="Legitimate", color="#4C78A8")
    ax.hist(lengths[df[LABEL_COLUMN] == 1], bins=bins, alpha=0.65, label="Fraudulent", color="#E45756")
    ax.set_xlabel("Words per posting (up to 99th percentile)")
    ax.set_ylabel("Postings")
    ax.set_title("Posting length by class")
    ax.legend()
    return _save(fig, path)


def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> str:
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(comparison), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (_, row) in enumerate(comparison.iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics], width, label=row["model"])
    ax.set_xticks(x + width * (len(comparison) - 1) / 2, [m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_title("Test-set comparison (fraudulent = positive class)")
    ax.legend()
    return _save(fig, path)


def plot_roc_pr(curves: dict[str, tuple[Sequence[int], Sequence[float]]], path: Path) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ax1.plot(fpr, tpr, label=name)
        ax2.plot(rec, prec, label=name)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax1.legend()
    ax2.legend()
    return _save(fig, path)


def plot_top_features(pipeline, path: Path, top_n: int = 15) -> str | None:
    """Plot the strongest TF-IDF terms for a linear model; returns None otherwise."""
    clf = pipeline.steps[-1][1]
    if not hasattr(clf, "coef_"):
        return None
    names = np.asarray(pipeline.named_steps["tfidf"].get_feature_names_out())
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    selected = np.concatenate([order[:top_n], order[-top_n:]])
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["#4C78A8" if coefs[i] < 0 else "#E45756" for i in selected]
    ax.barh(names[selected], coefs[selected], color=colors)
    ax.set_title("Most influential terms (red = toward fraudulent)")
    ax.set_xlabel("Logistic Regression coefficient")
    return _save(fig, path)


def run_eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Create EDA figures and return a JSON-serialisable summary."""
    out_dir = Path(out_dir)
    figures = {
        "class_distribution": plot_class_distribution(df_clean, out_dir / "eda_class_distribution.png"),
        "missing_values": plot_missing_values(df_raw, out_dir / "eda_missing_values.png"),
        "text_length": plot_text_length(df_clean, out_dir / "eda_text_length.png"),
    }
    lengths = df_clean["text"].str.split().str.len()
    summary = {
        "n_raw": int(len(df_raw)),
        "n_clean": int(len(df_clean)),
        "median_words_legit": float(lengths[df_clean[LABEL_COLUMN] == 0].median()),
        "median_words_fraud": float(lengths[df_clean[LABEL_COLUMN] == 1].median()),
        "missing_pct": {
            c: float(((df_raw[c].isna()) | (df_raw[c].astype(str).str.strip() == "")).mean() * 100)
            for c in TEXT_FIELDS if c in df_raw.columns
        },
        "figures": figures,
    }
    return summary
