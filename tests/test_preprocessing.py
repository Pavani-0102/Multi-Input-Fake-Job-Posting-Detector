import pandas as pd
import pytest

from preprocessing import DataValidationError, build_text, clean_dataframe, clean_text, load_raw_dataset
from tests.synthetic_data import make_synthetic_jobs


def test_clean_text_handles_none_and_nan():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("   ") == ""


def test_clean_text_normalises_tokens():
    out = clean_text("Visit https://x.com or mail HR@corp.com! Salary 5,000 <b>now</b> #URL_ab12cd#")
    assert "urltoken" in out and "emailtoken" in out and "numtoken" in out
    assert "<" not in out and "http" not in out and "@" not in out
    assert out == out.lower()


def test_build_text_skips_missing_and_blank():
    text = build_text({"title": "Engineer", "location": None, "description": "  Build things ", "benefits": float("nan")})
    assert text == "Engineer\nBuild things"


def test_clean_dataframe_removes_duplicates_and_conflicts():
    base = make_synthetic_jobs(50, seed=1)
    dup = base.iloc[[0]].copy()
    conflict_a = base.iloc[[1]].copy()
    conflict_b = base.iloc[[1]].copy()
    conflict_b["fraudulent"] = 1 - conflict_b["fraudulent"]
    df = pd.concat([base, dup, conflict_b], ignore_index=True)
    cleaned, report = clean_dataframe(df)
    assert report["rows_dropped_duplicates"] >= 1
    assert report["rows_dropped_conflicting_duplicates"] == 2
    assert cleaned["clean_text"].is_unique
    assert report["rows_clean"] == len(cleaned)


def test_clean_dataframe_drops_invalid_labels_and_empty_rows():
    df = make_synthetic_jobs(20, seed=2)
    df.loc[0, "fraudulent"] = None
    df.loc[1, ["title", "location", "salary_range", "employment_type", "company_profile", "description", "requirements", "benefits"]] = ""
    cleaned, report = clean_dataframe(df)
    assert report["rows_dropped_invalid_label"] == 1
    assert report["rows_dropped_empty_text"] == 1


def test_load_raw_dataset_validates(tmp_path):
    with pytest.raises(DataValidationError):
        load_raw_dataset(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(bad, index=False)
    with pytest.raises(DataValidationError, match="missing required columns"):
        load_raw_dataset(bad)
