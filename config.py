"""Central configuration shared by training, prediction and the Streamlit app."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATASET_PATH = DATA_DIR / "fake_job_postings.csv"
PIPELINE_PATH = MODELS_DIR / "fake_job_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5

LABEL_COLUMN = "fraudulent"
LEGIT_LABEL = 0
FRAUD_LABEL = 1
CLASS_NAMES = ["Legitimate", "Fraudulent"]

# Fields used to build the single text feature. Order matters only for readability.
# Every field here can be provided in the text form (Mode A) and is also present
# in the EMSCAD CSV, so training and inference use the same feature space.
TEXT_FIELDS = [
    "title",
    "location",
    "salary_range",
    "employment_type",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]
REQUIRED_COLUMNS = TEXT_FIELDS + [LABEL_COLUMN]

# Input validation limits
MIN_WORDS = 15
MAX_CHARS = 50_000
MIN_LETTER_RATIO = 0.40

# OCR limits
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}

DISCLAIMER = (
    "This result is a statistical estimate produced by a machine-learning model "
    "trained on a public dataset. It is NOT proof that a job posting is fraudulent "
    "or legitimate, and it does not verify the identity of any company or recruiter. "
    "Always verify employers through official channels, and never pay fees or share "
    "sensitive documents before confirming an offer independently."
)
