"""Train, compare and save the fake-job-posting classifiers.

Usage:
    python train_model.py                       # uses data/fake_job_postings.csv
    python train_model.py --data path/to.csv --seed 42

Leakage controls
  * duplicate / conflicting postings are removed BEFORE the split;
  * the stratified test set is untouched until final evaluation;
  * TF-IDF lives inside the sklearn Pipeline, so it is fitted only on training folds;
  * hyper-parameters and the final model are chosen by cross-validated F1 on the
    TRAINING set only. Test-set numbers are reported, never used for selection.

Class imbalance: balanced class weights (Logistic Regression) and balanced sample
weights (Multinomial Naive Bayes); F1 of the fraudulent class is the selection metric.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

import config
from evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_pr,
    plot_top_features,
    run_eda,
)
from preprocessing import DataValidationError, clean_dataframe, clean_text, load_raw_dataset


def build_pipeline(classifier, memory=None) -> Pipeline:
    """TF-IDF (uni+bi-grams) followed by the classifier."""
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=50_000,
        sublinear_tf=True,
    )
    return Pipeline([("tfidf", tfidf), ("clf", classifier)], memory=memory)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fake job posting detectors.")
    p.add_argument("--data", type=Path, default=config.DATASET_PATH, help="Path to fake_job_postings.csv")
    p.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    p.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    p.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for cross-validation")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load, EDA, clean --------------------------------------------------
    print(f"[1/6] Loading dataset: {args.data}")
    if not Path(args.data).exists():
        print(
            f"ERROR: dataset not found at '{args.data}'.\n"
            "Download EMSCAD ('fake_job_postings.csv') as described in README.md and place it in data/.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        df_raw = load_raw_dataset(args.data)
    except DataValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print("[2/6] Cleaning data and running EDA")
    df, clean_report = clean_dataframe(df_raw)
    print("      cleaning report:", json.dumps(clean_report))
    if clean_report["fraud_count"] < 10 or clean_report["legit_count"] < 10:
        print("ERROR: need at least 10 postings of each class after cleaning.", file=sys.stderr)
        raise SystemExit(1)
    eda = run_eda(df_raw, df, args.reports_dir)

    # ---- 2. Split (stratified, before any fitting) -----------------------------
    print("[3/6] Stratified train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df[config.LABEL_COLUMN],
        test_size=args.test_size, stratify=df[config.LABEL_COLUMN], random_state=args.seed,
    )
    print(f"      train={len(X_train):,}  test={len(X_test):,}  "
          f"train fraud ratio={y_train.mean():.3f}  test fraud ratio={y_test.mean():.3f}")

    # ---- 3. Model search on training data only --------------------------------
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    sample_weight = compute_sample_weight("balanced", y_train)

    candidates = {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, random_state=args.seed),
            {"clf__C": [1, 5, 10, 20]},
            {},
        ),
        "Multinomial Naive Bayes": (
            MultinomialNB(),
            {"clf__alpha": [0.01, 0.05, 0.1, 0.5]},
            {"clf__sample_weight": sample_weight},
        ),
    }

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    curves: dict[str, tuple] = {}
    print("[4/6] Cross-validated training (selection metric: F1 of fraudulent class)")
    with tempfile.TemporaryDirectory() as cache_dir:  # caches the TF-IDF fit per CV fold
        for name, (clf, grid, fit_params) in candidates.items():
            search = GridSearchCV(
                build_pipeline(clf, memory=cache_dir), grid, scoring="f1", cv=cv,
                n_jobs=args.n_jobs, refit=True,
            )
            search.fit(X_train, y_train, **fit_params)
            best = search.best_estimator_
            best.memory = None  # do not pickle the temporary cache path
            fitted[name] = best

            y_pred = best.predict(X_test)
            y_score = best.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_pred, y_score)
            results[name] = {
                "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
                "cv_f1": float(search.best_score_),
                "test": metrics,
                "classification_report": classification_report(
                    y_test, y_pred, target_names=config.CLASS_NAMES, digits=4, zero_division=0),
            }
            curves[name] = (y_test.to_numpy(), y_score)
            print(f"      {name}: best={results[name]['best_params']}  CV F1={search.best_score_:.4f}  "
                  f"test F1={metrics['f1']:.4f}  recall={metrics['recall']:.4f}  precision={metrics['precision']:.4f}")

    # ---- 4. Select by CV F1 (training data only) ------------------------------
    selected = max(results, key=lambda n: results[n]["cv_f1"])
    print(f"[5/6] Selected model (highest cross-validated F1 on training data): {selected}")

    # ---- 5. Reports and artifacts ---------------------------------------------
    comparison = pd.DataFrame([
        {
            "model": name,
            "cv_f1": r["cv_f1"],
            "accuracy": r["test"]["accuracy"],
            "precision": r["test"]["precision"],
            "recall": r["test"]["recall"],
            "f1": r["test"]["f1"],
            "roc_auc": r["test"].get("roc_auc"),
            "pr_auc": r["test"].get("pr_auc"),
        }
        for name, r in results.items()
    ])
    comparison.to_csv(args.reports_dir / "model_comparison.csv", index=False)

    figures = dict(eda["figures"])
    for name, r in results.items():
        slug = name.lower().replace(" ", "_")
        figures[f"confusion_{slug}"] = plot_confusion_matrix(
            r["test"]["confusion_matrix"], f"{name} (test set)", args.reports_dir / f"confusion_matrix_{slug}.png")
        (args.reports_dir / f"classification_report_{slug}.txt").write_text(r["classification_report"], encoding="utf-8")
    figures["model_comparison"] = plot_model_comparison(comparison, args.reports_dir / "model_comparison.png")
    figures["roc_pr"] = plot_roc_pr(curves, args.reports_dir / "roc_pr_curves.png")
    if "Logistic Regression" in fitted:
        top = plot_top_features(fitted["Logistic Regression"], args.reports_dir / "top_features_logreg.png")
        if top:
            figures["top_features"] = top

    pipeline_path = args.models_dir / config.PIPELINE_PATH.name
    joblib.dump(fitted[selected], pipeline_path)

    metadata = {
        "model_name": selected,
        "selection_rule": "highest 5-fold cross-validated F1 (fraudulent class) on the training split",
        "decision_threshold": config.DECISION_THRESHOLD,
        "probability_note": "Uncalibrated model score; treat as a relative indicator, not a true probability.",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": args.seed,
        "test_size": args.test_size,
        "dataset_path": str(args.data),
        "dataset": clean_report,
        "text_fields": config.TEXT_FIELDS,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__},
        "results": {n: {k: v for k, v in r.items() if k != "classification_report"} for n, r in results.items()},
    }
    (args.models_dir / config.METADATA_PATH.name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.reports_dir / config.METRICS_PATH.name).write_text(
        json.dumps({**metadata, "eda": {k: v for k, v in eda.items() if k != "figures"},
                    "figures": {k: Path(v).name for k, v in figures.items()},
                    "classification_reports": {n: r["classification_report"] for n, r in results.items()}}, indent=2),
        encoding="utf-8",
    )

    print(f"[6/6] Saved pipeline -> {pipeline_path}")
    print(f"      Saved reports  -> {args.reports_dir}")
    print("\n" + comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for name, r in results.items():
        print(f"\n=== {name} (test set) ===\n{r['classification_report']}\nConfusion matrix [[TN, FP], [FN, TP]]: {r['test']['confusion_matrix']}")
    return metadata


if __name__ == "__main__":
    main()
