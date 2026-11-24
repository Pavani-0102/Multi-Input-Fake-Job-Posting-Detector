# Multi-Input Fake Job Posting Detection (ML + NLP + OCR)
Detects *potentially* fraudulent job postings from (A) typed job details, (B) e-mail screenshots and
(C) job-portal screenshots. Everything runs locally: scikit-learn, Tesseract OCR and Streamlit. No API keys or cloud services.

> **Not proof.** Output is a statistical estimate plus transparent rule-based warnings. It does not verify any company.

## Folder structure

```
fake_job_detector/
├── app.py                 # Streamlit UI (3 input modes + evaluation tab)
├── train_model.py         # EDA, cleaning, split, CV, model comparison, saving
├── preprocessing.py       # cleaning + text normalisation shared by training and inference
├── prediction.py          # model loading, validation, prediction
├── rules.py               # transparent suspicious-indicator checks
├── ocr_utils.py           # image validation, preprocessing, Tesseract OCR
├── evaluation.py          # metrics, EDA and plotting helpers
├── config.py              # paths, constants, disclaimer
├── requirements.txt
├── pytest.ini
├── .gitignore
├── README.md
├── data/                  # put fake_job_postings.csv here
├── models/                # created by train_model.py
├── reports/               # created by train_model.py
└── tests/                 # pytest suite (synthetic data used for tests only)
```

## 1. Dataset (EMSCAD)

The Employment Scam Aegean Dataset (EMSCAD, University of the Aegean) is mirrored on Kaggle as
**"Real / Fake Job Posting Prediction"** (~17,880 postings, ~866 fraudulent):
<https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction>

Place the file `fake_job_postings.csv` at `data/fake_job_postings.csv`.

- **Manual:** open the link above (free Kaggle account), click *Download*, unzip, copy the CSV into `data/`.
- **Kaggle CLI:** create an API token (Kaggle → Settings → API), put `kaggle.json` in `~/.kaggle/` (Windows: `%USERPROFILE%\.kaggle\`), then:
  ```bash
  pip install kaggle
  kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction -p data --unzip
  ```

The training script stops with a clear error if the file is missing; it never falls back to synthetic data.

## 2. Setup

**macOS / Linux**
```bash
cd fake_job_detector
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd fake_job_detector
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
(If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. In Command Prompt use `.venv\Scripts\activate.bat`.)

### Tesseract OCR (system program, required for Modes B and C)

| OS | Command |
|----|---------|
| macOS | `brew install tesseract` |
| Ubuntu/Debian | `sudo apt update && sudo apt install tesseract-ocr` |
| Windows | Install from <https://github.com/UB-Mannheim/tesseract/wiki> (default path `C:\Program Files\Tesseract-OCR`) |

Verify with `tesseract --version`. The app auto-detects the default Windows install paths. If Tesseract is elsewhere, set:

```powershell
# Windows PowerShell (current session)
$env:TESSERACT_CMD = "C:\Path\To\tesseract.exe"
```
```bash
# macOS / Linux
export TESSERACT_CMD=/path/to/tesseract
```

## 3. Train and evaluate

```bash
python train_model.py
```
Optional: `--seed 42 --test-size 0.2 --cv-folds 5 --n-jobs 2 --data path/to/fake_job_postings.csv`

What it does: loads and validates the CSV → handles missing values → removes duplicate and label-conflicting postings
**before** splitting → stratified 80/20 split → TF-IDF (1–2-grams, fitted only inside training folds) → grid-searches
Logistic Regression (`class_weight="balanced"`) and Multinomial Naive Bayes (balanced sample weights) with 5-fold
stratified CV on the training split → picks the model with the best CV F1 (fraudulent class) → reports accuracy,
precision, recall, F1, ROC-AUC, PR-AUC, classification report and confusion matrix on the untouched test split →
saves artifacts.

Outputs:
- `models/fake_job_pipeline.joblib` – selected pipeline (TF-IDF + classifier)
- `models/model_metadata.json` – selection rule, parameters, versions, metrics
- `reports/` – `metrics.json`, `model_comparison.csv`, classification reports, confusion matrices, ROC/PR curves, EDA plots, top terms

All reported numbers are computed during your run; none are hardcoded. Training takes a few minutes on a typical laptop.

## 4. Run the app

```bash
streamlit run app.py
```
Open <http://localhost:8501>. Modes: **A** form, **B** e-mail screenshot, **C** job-portal screenshot. In B and C, run OCR, then
review/correct the extracted text before clicking *Analyse text*.

## 5. Tests

```bash
python -m pytest -q
```
Tests use a small synthetic dataset (`tests/synthetic_data.py`) purely to check that the code runs; they say nothing about
real model quality. OCR tests are skipped automatically if Tesseract is not installed. Expected: all tests pass, e.g. `31 passed`.

## Design notes and limitations

- **Single text feature.** Form fields are concatenated into one text string, so typed input and OCR text go through exactly the same features. Structured EMSCAD columns such as `has_company_logo` are deliberately not used, since screenshots cannot supply them.
- **Score is not a calibrated probability.** It is shown as a "model-estimated fraud score" with the decision threshold (0.5).
- **Rules are separate from the model** and never change its prediction. The optional e-mail sender address feeds only the rules.
- **Dataset limits.** EMSCAD is from 2012–2014 and mostly English; scam styles have evolved, and Indian portal formats (e.g. Naukri) may differ from its distribution. Treat results as a research prototype.
- OCR quality depends on screenshot resolution; always check the extracted text.

## Common errors and fixes

| Problem | Fix |
|---------|-----|
| `ERROR: dataset not found` | Put `fake_job_postings.csv` in `data/` (section 1). |
| `Dataset is missing required columns` | You downloaded a different file; use the EMSCAD CSV above. |
| `No trained model found` in the app | Run `python train_model.py`, then click *Reload model* in the sidebar. |
| `Tesseract OCR is not installed or not found` | Install Tesseract (section 2) and restart the terminal, or set `TESSERACT_CMD`. |
| Windows: `tesseract is not recognized` | Add `C:\Program Files\Tesseract-OCR` to PATH or set `TESSERACT_CMD`. |
| Model fails to load after upgrading packages | Re-run `python train_model.py` (pickles depend on the scikit-learn version). |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt`. |
| `ModuleNotFoundError: preprocessing` when loading the model elsewhere | Run commands from the project root. |
| Garbled OCR text / "Low OCR confidence" | Use a sharper, larger screenshot, crop to the posting, try another *Layout mode*, and edit the text. |
| `width="stretch"` TypeError in the app | Upgrade Streamlit: `pip install -U "streamlit>=1.50"`. |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
