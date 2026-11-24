import pytest

from prediction import (
    InputValidationError,
    ModelNotFoundError,
    PredictionResult,
    load_model,
    predict_fields,
    predict_text,
    validate_text,
)

FRAUD_TEXT = ("No experience needed, earn 5000 per week working from home with guaranteed income. Pay a small "
              "registration fee to receive your training kit. Contact our manager on WhatsApp, urgent hiring, limited seats!")
LEGIT_TEXT = ("You will collaborate with cross functional teams to design and maintain scalable systems. The ideal "
              "candidate holds a bachelor degree and has three years of relevant professional experience. We offer "
              "health insurance and structured career development.")


def test_validate_text_errors():
    with pytest.raises(InputValidationError):
        validate_text("")
    with pytest.raises(InputValidationError):
        validate_text(None)
    with pytest.raises(InputValidationError, match="too short"):
        validate_text("just a few words")
    with pytest.raises(InputValidationError, match="readable letters"):
        validate_text(" ".join(["1234567 ###"] * 30))
    with pytest.raises(InputValidationError, match="too long"):
        validate_text("word " * 20_000)


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotFoundError, match="train_model.py"):
        load_model(tmp_path / "nope.joblib")


def test_predict_text_output_structure(trained_model):
    model, _ = trained_model
    result = predict_text(model, FRAUD_TEXT)
    assert isinstance(result, PredictionResult)
    assert result.label in {"Potentially fraudulent", "Potentially legitimate"}
    assert 0.0 <= result.fraud_probability <= 1.0
    assert result.is_fraud_predicted == (result.fraud_probability >= result.threshold)
    assert result.indicators  # rule checks fire on the scam-style text


def test_separates_scam_and_legit_on_synthetic_data(trained_model):
    # Synthetic data is trivially separable; this only proves the plumbing is wired correctly.
    model, _ = trained_model
    assert predict_text(model, FRAUD_TEXT).fraud_probability > predict_text(model, LEGIT_TEXT).fraud_probability


def test_predict_fields_and_rule_context(trained_model):
    model, _ = trained_model
    fields = {"title": "Typist", "company_profile": "", "description": FRAUD_TEXT, "requirements": ""}
    result = predict_fields(model, fields)
    codes = {i.code for i in result.indicators}
    assert {"missing_company_profile", "upfront_payment"} <= codes

    with_context = predict_text(model, LEGIT_TEXT, rule_context="From: recruiter@gmail.com")
    assert "free_email_contact" in {i.code for i in with_context.indicators}


def test_rule_context_does_not_change_model_output(trained_model):
    model, _ = trained_model
    a = predict_text(model, LEGIT_TEXT)
    b = predict_text(model, LEGIT_TEXT, rule_context="From: x@gmail.com WhatsApp")
    assert a.fraud_probability == b.fraud_probability


def test_artifacts_written(trained_model):
    _, out = trained_model
    for name in ("models/fake_job_pipeline.joblib", "models/model_metadata.json", "reports/metrics.json",
                 "reports/model_comparison.csv", "reports/confusion_matrix_logistic_regression.png",
                 "reports/confusion_matrix_multinomial_naive_bayes.png", "reports/eda_class_distribution.png"):
        assert (out / name).exists(), name
