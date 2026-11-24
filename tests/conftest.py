import pytest

from tests.synthetic_data import make_synthetic_jobs


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "fake_job_postings.csv"
    make_synthetic_jobs().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained_model(synthetic_csv, tmp_path_factory):
    """Run the real training script on synthetic data and load its output."""
    import train_model
    from prediction import load_model

    out = tmp_path_factory.mktemp("artifacts")
    train_model.main([
        "--data", str(synthetic_csv),
        "--models-dir", str(out / "models"),
        "--reports-dir", str(out / "reports"),
        "--cv-folds", "3",
    ])
    return load_model(out / "models" / "fake_job_pipeline.joblib", out / "models" / "model_metadata.json"), out
