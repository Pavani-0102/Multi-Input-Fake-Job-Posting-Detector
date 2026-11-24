import json

import pytest

import train_model
from evaluation import compute_metrics


def test_compute_metrics_matches_hand_calculation():
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 0, 1, 0]
    m = compute_metrics(y_true, y_pred, [0.9, 0.8, 0.4, 0.1, 0.2, 0.1, 0.6, 0.3])
    assert m["confusion_matrix"] == [[4, 1], [1, 2]]
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert 0.5 < m["roc_auc"] <= 1.0


def test_metadata_records_selection_and_no_test_leak(trained_model):
    _, out = trained_model
    meta = json.loads((out / "models" / "model_metadata.json").read_text())
    assert meta["model_name"] in meta["results"]
    best_cv = max(r["cv_f1"] for r in meta["results"].values())
    assert meta["results"][meta["model_name"]]["cv_f1"] == best_cv  # selected on CV, not test
    assert set(meta["results"]) == {"Logistic Regression", "Multinomial Naive Bayes"}


def test_training_is_reproducible(synthetic_csv, tmp_path):
    runs = []
    for i in range(2):
        meta = train_model.main(["--data", str(synthetic_csv), "--models-dir", str(tmp_path / f"m{i}"),
                                 "--reports-dir", str(tmp_path / f"r{i}"), "--cv-folds", "3", "--seed", "7"])
        runs.append({n: r["test"]["f1"] for n, r in meta["results"].items()})
    assert runs[0] == runs[1]


def test_missing_dataset_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as exc:
        train_model.main(["--data", str(tmp_path / "absent.csv"), "--models-dir", str(tmp_path / "m"), "--reports-dir", str(tmp_path / "r")])
    assert exc.value.code == 1
