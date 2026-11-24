"""Streamlit app: multi-input fake job posting detection (text, e-mail screenshot, portal screenshot)."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

import config
from ocr_utils import OCRError, extract_text_from_image, tesseract_status
from prediction import (
    InputValidationError,
    LoadedModel,
    ModelNotFoundError,
    PredictionResult,
    load_model,
    predict_fields,
    predict_text,
)

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🛡️", layout="wide")

EMPLOYMENT_TYPES = ["", "Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"]
PSM_OPTIONS = {
    "Automatic page layout (recommended)": 3,
    "Single column of text": 4,
    "Single uniform block": 6,
}
SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading trained model…")
def get_model() -> LoadedModel:
    return load_model()


def try_load_model() -> LoadedModel | None:
    try:
        return get_model()
    except ModelNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:  # corrupted file, incompatible scikit-learn version, ...
        st.error(
            "The saved model could not be loaded (it may have been trained with a different "
            f"scikit-learn version). Re-run `python train_model.py`. Details: {exc}"
        )
    return None


def render_result(result: PredictionResult) -> None:
    st.subheader("Result")
    col_label, col_prob, col_ind = st.columns([2, 2, 1])
    with col_label:
        if result.is_fraud_predicted:
            st.error(f"⚠️ **{result.label}**")
        else:
            st.success(f"✅ **{result.label}**")
        st.caption(f"Model: {result.model_name} · {result.word_count} words analysed")
    with col_prob:
        if result.fraud_probability is not None:
            st.metric("Model-estimated fraud score", f"{result.fraud_probability:.1%}")
            st.progress(min(max(result.fraud_probability, 0.0), 1.0))
            st.caption(f"Decision threshold {result.threshold:.0%}. Uncalibrated score, not a guaranteed probability.")
        else:
            st.info("The selected model does not provide probability estimates.")
    with col_ind:
        st.metric("Rule-based flags", len(result.indicators))

    st.markdown("#### Suspicious indicators (transparent rule checks)")
    if result.indicators:
        for ind in result.indicators:
            st.warning(f"{SEVERITY_ICON[ind.severity]} **{ind.title}** ({ind.severity})  \n{ind.detail}")
    else:
        st.info("No rule-based warning patterns were found. This does not guarantee the posting is genuine.")
    st.caption("Indicators come from fixed keyword/pattern rules and are independent of the ML model.")
    st.info(config.DISCLAIMER)


def run_prediction(model: LoadedModel, text: str, rule_context: str = "") -> None:
    try:
        with st.spinner("Analysing…"):
            result = predict_text(model, text, rule_context=rule_context)
    except InputValidationError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error(f"Prediction failed unexpectedly: {exc}")
        return
    render_result(result)


# ----------------------------------------------------------------------------- Mode A
def mode_text(model: LoadedModel | None) -> None:
    st.markdown("Enter the details of the job posting. Leave unknown fields blank.")
    with st.form("text_form"):
        c1, c2 = st.columns(2)
        title = c1.text_input("Job title")
        location = c2.text_input("Location")
        c3, c4 = st.columns(2)
        salary = c3.text_input("Salary information", placeholder="e.g. 30000-45000 per month")
        employment = c4.selectbox("Employment type", EMPLOYMENT_TYPES)
        company_profile = st.text_area("Company profile", height=110)
        description = st.text_area("Job description", height=170)
        requirements = st.text_area("Job requirements", height=120)
        benefits = st.text_area("Benefits (optional)", height=80)
        submitted = st.form_submit_button("Analyse posting", type="primary")
    if not submitted:
        return
    if model is None:
        st.error("Train the model first: `python train_model.py`.")
        return
    fields = {
        "title": title, "location": location, "salary_range": salary, "employment_type": employment,
        "company_profile": company_profile, "description": description,
        "requirements": requirements, "benefits": benefits,
    }
    if not any(v.strip() for v in fields.values()):
        st.warning("Please fill in at least the job description.")
        return
    try:
        with st.spinner("Analysing…"):
            result = predict_fields(model, fields)
    except InputValidationError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error(f"Prediction failed unexpectedly: {exc}")
        return
    render_result(result)


# ----------------------------------------------------------------------------- Modes B / C
def mode_screenshot(model: LoadedModel | None, kind: str) -> None:
    is_email = kind == "email"
    if is_email:
        st.markdown("Upload a screenshot of a recruitment e-mail or job-offer message (JPG, JPEG or PNG).")
    else:
        st.markdown("Upload a screenshot of a job advertisement from a portal such as LinkedIn or Naukri (JPG, JPEG or PNG). "
                    "Crop out menus and unrelated listings for best results.")

    ok, message = tesseract_status()
    if not ok:
        st.error(message)

    upload = st.file_uploader("Screenshot", type=["jpg", "jpeg", "png"], key=f"upload_{kind}")
    sender = ""
    if is_email:
        sender = st.text_input(
            "Sender address (optional)", key="sender_email",
            help="Used only for the rule-based checks (e.g. free e-mail domains). It is not given to the ML model.",
        )

    text_key, meta_key, id_key = f"edit_{kind}", f"meta_{kind}", f"fileid_{kind}"

    if upload is None:
        for key in (text_key, meta_key, id_key):
            st.session_state.pop(key, None)
        return

    data = upload.getvalue()
    file_id = hashlib.sha256(data).hexdigest()
    if st.session_state.get(id_key) != file_id:  # a different image: discard the previous OCR text
        st.session_state.pop(text_key, None)
        st.session_state.pop(meta_key, None)
        st.session_state[id_key] = file_id

    left, right = st.columns([1, 1])
    with left:
        st.image(data, caption=upload.name, width="stretch")
    with right:
        with st.expander("OCR options"):
            psm_label = st.selectbox("Layout mode", list(PSM_OPTIONS), key=f"psm_{kind}")
        if st.button("Extract text with OCR", type="primary", disabled=not ok, key=f"ocr_btn_{kind}"):
            try:
                with st.spinner("Reading text from the image…"):
                    ocr = extract_text_from_image(data, psm=PSM_OPTIONS[psm_label])
                st.session_state[text_key] = ocr.text
                st.session_state[meta_key] = ocr
            except OCRError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected OCR error: {exc}")

        if text_key in st.session_state:
            ocr = st.session_state.get(meta_key)
            if ocr is not None:
                conf = f"{ocr.mean_confidence:.0f}/100" if ocr.mean_confidence is not None else "n/a"
                st.caption(f"Extracted {ocr.word_count} words · mean OCR confidence {conf}")
                for warning in ocr.warnings:
                    st.warning(warning)
            st.text_area(
                "Extracted text — review and correct before analysing",
                key=text_key, height=320,
            )
            if st.button("Analyse text", type="primary", key=f"analyse_{kind}"):
                if model is None:
                    st.error("Train the model first: `python train_model.py`.")
                else:
                    context = f"From: {sender}" if sender.strip() else ""
                    st.session_state[f"run_{kind}"] = (st.session_state[text_key], context)
        else:
            st.caption("Click **Extract text with OCR** to read the image.")

    pending = st.session_state.pop(f"run_{kind}", None)
    if pending and model is not None:
        run_prediction(model, pending[0], pending[1])


# ----------------------------------------------------------------------------- evaluation tab
def evaluation_tab() -> None:
    if not config.METRICS_PATH.exists():
        st.info("No evaluation results yet. Run `python train_model.py` to train the models and generate them.")
        return
    try:
        metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.error("The metrics file is unreadable. Re-run `python train_model.py`.")
        return

    st.markdown(f"**Selected model:** {metrics['model_name']}  \n"
                f"**Selection rule:** {metrics['selection_rule']}  \n"
                f"**Trained (UTC):** {metrics['trained_at_utc']} · seed {metrics['random_seed']}")
    ds = metrics["dataset"]
    st.caption(f"Cleaned dataset: {ds['rows_clean']:,} postings ({ds['fraud_count']:,} fraudulent, "
               f"{ds['fraud_ratio']:.1%}) from {ds['rows_raw']:,} raw rows. All figures below were computed on a held-out test split.")

    rows = []
    for name, r in metrics["results"].items():
        t = r["test"]
        rows.append({"Model": name, "CV F1 (train)": r["cv_f1"], "Accuracy": t["accuracy"], "Precision": t["precision"],
                     "Recall": t["recall"], "F1": t["f1"], "ROC-AUC": t.get("roc_auc"), "PR-AUC": t.get("pr_auc"),
                     "Best params": json.dumps(r["best_params"])})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.4f}" for c in
                 ["CV F1 (train)", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]}, na_rep="n/a"),
                 width="stretch", hide_index=True)
    st.caption("Precision, recall and F1 treat 'fraudulent' as the positive class. Accuracy alone is misleading on imbalanced data.")

    figs = metrics.get("figures", {})

    def show(key: str, caption: str) -> None:
        path = config.REPORTS_DIR / figs.get(key, "")
        if key in figs and path.exists():
            st.image(str(path), caption=caption, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        show("confusion_logistic_regression", "Logistic Regression")
    with c2:
        show("confusion_multinomial_naive_bayes", "Multinomial Naive Bayes")
    show("model_comparison", "Model comparison")
    show("roc_pr", "ROC and precision-recall curves")

    with st.expander("Classification reports"):
        for name, report in metrics.get("classification_reports", {}).items():
            st.markdown(f"**{name}**")
            st.code(report)
    with st.expander("Exploratory data analysis"):
        e1, e2, e3 = st.columns(3)
        with e1:
            show("class_distribution", "Class distribution")
        with e2:
            show("missing_values", "Missing values")
        with e3:
            show("text_length", "Posting length")
    with st.expander("Most influential terms (Logistic Regression)"):
        show("top_features", "Terms pushing predictions toward fraudulent / legitimate")


# ----------------------------------------------------------------------------- main
def main() -> None:
    st.title("🛡️ Multi-Input Fake Job Posting Detector")
    st.caption("Machine Learning + NLP + OCR · runs fully offline on this computer")

    with st.sidebar:
        st.header("Status")
        model = try_load_model()
        if model is not None:
            st.success(f"Model: {model.metadata.get('model_name', 'loaded')}")
            res = model.metadata.get("results", {}).get(model.metadata.get("model_name", ""), {})
            if res:
                st.caption(f"Test F1 (fraud class): {res['test']['f1']:.3f}")
        ok, msg = tesseract_status()
        (st.success if ok else st.warning)(f"OCR: {msg}")
        if st.button("Reload model"):
            get_model.clear()
            st.rerun()
        st.divider()
        st.caption(config.DISCLAIMER)

    tab_analyse, tab_eval = st.tabs(["Analyse a posting", "Model evaluation"])
    with tab_analyse:
        mode = st.radio(
            "Input mode",
            ["A · Enter text", "B · Email screenshot", "C · Job portal screenshot"],
            horizontal=True,
        )
        if mode.startswith("A"):
            mode_text(model)
        elif mode.startswith("B"):
            mode_screenshot(model, "email")
        else:
            mode_screenshot(model, "portal")
    with tab_eval:
        evaluation_tab()


main()
