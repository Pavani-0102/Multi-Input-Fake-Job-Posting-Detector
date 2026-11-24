"""Shared data-cleaning and text-preprocessing utilities.

The same functions are used by training and by inference, so the feature space is
identical in both places.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd

from config import FRAUD_LABEL, LABEL_COLUMN, LEGIT_LABEL, REQUIRED_COLUMNS, TEXT_FIELDS

# EMSCAD replaces URLs / e-mails / phone numbers with placeholders such as #URL_ab12cd#.
_PLACEHOLDER_RE = re.compile(r"#(url|email|phone)_[a-z0-9]+#", re.IGNORECASE)
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_HTML_RE = re.compile(r"<[^>]+>")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_NON_LETTER_RE = re.compile(r"[^a-z\s]")
_SPACE_RE = re.compile(r"\s+")


class DataValidationError(ValueError):
    """Raised when the training CSV does not look like the EMSCAD dataset."""


def clean_text(text: Any) -> str:
    """Normalise raw text for TF-IDF.

    Lower-cases, strips HTML, maps URLs/e-mails/phones/numbers to stable tokens and
    removes remaining punctuation. Returns an empty string for None/NaN.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).lower()
    text = _HTML_RE.sub(" ", text)
    text = _PLACEHOLDER_RE.sub(lambda m: f" {m.group(1).lower()}token ", text)
    text = _URL_RE.sub(" urltoken ", text)
    text = _EMAIL_RE.sub(" emailtoken ", text)
    text = _NUMBER_RE.sub(" numtoken ", text)
    text = _NON_LETTER_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def build_text(fields: Mapping[str, Any]) -> str:
    """Concatenate the configured job fields into a single raw text string."""
    parts: list[str] = []
    for name in TEXT_FIELDS:
        value = fields.get(name)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        value = str(value).strip()
        if value:
            parts.append(value)
    return "\n".join(parts)


def load_raw_dataset(path) -> pd.DataFrame:
    """Read the EMSCAD CSV and verify the expected columns exist."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataValidationError(f"Dataset not found at '{path}'.") from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Could not parse '{path}' as CSV: {exc}") from exc
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Dataset is missing required columns: {missing}. "
            "Expected the EMSCAD 'fake_job_postings.csv' file."
        )
    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Handle missing values, build the text column and remove ambiguous duplicates.

    Duplicates are removed BEFORE the train/test split so identical postings cannot
    appear on both sides (a common source of leakage in EMSCAD). Postings whose
    identical text carries conflicting labels are dropped entirely.
    """
    report: dict[str, Any] = {"rows_raw": int(len(df))}
    out = df.copy()

    out[LABEL_COLUMN] = pd.to_numeric(out[LABEL_COLUMN], errors="coerce")
    bad_label = out[LABEL_COLUMN].isna() | ~out[LABEL_COLUMN].isin([LEGIT_LABEL, FRAUD_LABEL])
    report["rows_dropped_invalid_label"] = int(bad_label.sum())
    out = out.loc[~bad_label].copy()
    out[LABEL_COLUMN] = out[LABEL_COLUMN].astype(int)

    for col in TEXT_FIELDS:
        out[col] = out[col].fillna("").astype(str).str.strip()

    out["text"] = out.apply(lambda r: build_text(r), axis=1)
    out["clean_text"] = out["text"].map(clean_text)

    empty = out["clean_text"].str.len() == 0
    report["rows_dropped_empty_text"] = int(empty.sum())
    out = out.loc[~empty].copy()

    nunique_labels = out.groupby("clean_text")[LABEL_COLUMN].transform("nunique")
    conflicting = nunique_labels > 1
    report["rows_dropped_conflicting_duplicates"] = int(conflicting.sum())
    out = out.loc[~conflicting].copy()

    before = len(out)
    out = out.drop_duplicates(subset="clean_text", keep="first")
    report["rows_dropped_duplicates"] = int(before - len(out))

    out = out.reset_index(drop=True)
    report["rows_clean"] = int(len(out))
    report["fraud_count"] = int((out[LABEL_COLUMN] == FRAUD_LABEL).sum())
    report["legit_count"] = int((out[LABEL_COLUMN] == LEGIT_LABEL).sum())
    report["fraud_ratio"] = float(report["fraud_count"] / max(len(out), 1))
    return out, report
