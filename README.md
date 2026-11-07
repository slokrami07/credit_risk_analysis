# ◈ CreditLens — AI-Powered Loan Default Risk Analyzer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Calibrated-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

**A production-grade credit risk scoring system — from raw data to a live prediction API with a sleek dark-mode UI.**

[Live Demo](#) · [API Docs](#api-reference) · [Model Details](#ml-pipeline)

</div>

---

## What is this?

CreditLens takes a loan applicant's profile and instantly predicts whether they are likely to **default** on their loan — returning a calibrated probability score, risk band, and decision threshold.

Built end-to-end: data cleaning → feature engineering → model training → hyperparameter tuning → probability calibration → REST API → polished web interface → one-click Render deployment.

---

## Features

- **Four models benchmarked** — Logistic Regression, XGBoost, Random Forest, AdaBoost evaluated with 5-fold stratified cross-validation
- **Hyperparameter tuning** — `RandomizedSearchCV` with 150 iterations on XGBoost and Random Forest
- **Probability calibration** — Isotonic regression calibration on XGBoost for well-calibrated confidence scores
- **Bootstrap stability analysis** — 1000-iteration bootstrap CI comparison across calibration methods
- **Threshold optimization** — Best F1 threshold selected on a held-out validation set
- **SHAP explainability** — Global feature importance + local waterfall explanations
- **FastAPI backend** — `/predict` endpoint with Pydantic validation, risk banding, and confidence percentage
- **Dark-mode UI** — Animated gauge, probability bar, risk chip, stats grid — no frameworks, pure HTML/CSS/JS
- **Render-ready deployment** — `render.yaml` included, zero config changes needed between local and production

---

## ML Pipeline

```
Raw CSV (32,581 rows)
       │
       ▼
┌─────────────────────────────┐
│  EDA & Visualization        │  → 15+ plots auto-saved to plots_and_graphs/
│  · Distribution plots       │
│  · Correlation heatmap      │
│  · Boxplots per feature     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Data Cleaning              │
│  · Remove duplicates        │
│  · Filter invalid ages      │
│  · Filter emp_length > age  │
│  · Remove zero loan amounts │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Preprocessing Pipelines    │
│  · Numeric: Median impute   │
│    + StandardScaler (LR)    │
│  · Categorical: Mode impute │
│    + OneHotEncoder          │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Class Imbalance Handling   │
│  · class_weight="balanced"  │
│    for LR & RF              │
│  · scale_pos_weight for XGB │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  5-Fold Stratified CV       │
│  Logistic Regression        │  ROC-AUC: 0.871
│  XGBoost                    │  ROC-AUC: 0.947 ★
│  Random Forest              │  ROC-AUC: 0.925
│  AdaBoost                   │  ROC-AUC: 0.871
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Hyperparameter Tuning      │
│  RandomizedSearchCV 150iter │
│  XGBoost best → 0.9511 AUC  │
│  RF best     → 0.9422 AUC   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Isotonic Calibration       │
│  Brier Score: 0.047         │
│  Bootstrap CI validated     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Threshold Optimization     │
│  Best F1 threshold on val   │
│  SHAP global + local explns │
└────────────┬────────────────┘
             │
             ▼
   credit_risk_model.pkl
   best_threshold.pkl
```

### Model Comparison

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.871 | 0.812 | 0.547 | 0.778 | 0.643 |
| AdaBoost | 0.871 | 0.844 | 0.895 | 0.312 | 0.463 |
| Random Forest (tuned) | 0.942 | 0.930 | **0.980** | 0.710 | 0.820 |
| **XGBoost (tuned + calibrated)** | **0.951** | **0.920** | 0.840 | **0.790** | **0.820** |

> XGBoost selected as final model — best balance of AUC, recall, and calibrated probabilities.

---

## Project Structure

```
credit_risk_score/
│
├── main.py                    # FastAPI app — API + serves frontend
├── ml_model.py                # Full ML pipeline — EDA → training → SHAP
├── requirements.txt           # Python dependencies
├── render.yaml                # Render deployment config
├── .gitignore
│
├── static/                    # Frontend (served by FastAPI)
│   ├── index.html             # Markup only
│   ├── style.css              # Dark-mode design system
│   └── app.js                 # Validation, fetch, gauge, toast
│
├── plots_and_graphs/          # Auto-generated training visualizations
│   ├── correlation_heatmap.png
│   ├── shap_summary_plot.png
│   ├── xgboost_hyper_parameter_tuning_confusion_matrix.png
│   └── ...
│
├── credit_risk_model.pkl      # Trained + calibrated XGBoost model
├── best_threshold.pkl         # Optimal F1 classification threshold
└── credit_risk_dataset.csv    # Source data (not committed)
```

---

## API Reference

### `POST /predict`

Predict loan default probability for an applicant.

**Request body:**

```json
{
  "person_age": 29,
  "person_income": 55000,
  "person_home_ownership": "RENT",
  "person_emp_length": 4.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.18,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

**Response:**

```json
{
  "default_probability": 0.1823,
  "default_prediction": 0,
  "threshold": 0.3741,
  "result": "low risk",
  "risk_band": "Low Risk",
  "risk_color": "green",
  "confidence_pct": 18.23
}
```

**Risk bands:**

| Band | Probability | Color |
|---|---|---|
| Low Risk | < 30% | 🟢 Green |
| Moderate Risk | 30% – 60% | 🟡 Amber |
| High Risk | > 60% | 🔴 Red |

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /docs`

Auto-generated Swagger UI — interactive API playground.

---

## Field Reference

| Field | Type | Description |
|---|---|---|
| `person_age` | int | Applicant age (18–100) |
| `person_income` | float | Annual gross income in USD |
| `person_home_ownership` | str | `RENT` `OWN` `MORTGAGE` `OTHER` |
| `person_emp_length` | float | Employment length in years (0–60) |
| `loan_intent` | str | `PERSONAL` `EDUCATION` `MEDICAL` `VENTURE` `HOMEIMPROVEMENT` `DEBTCONSOLIDATION` |
| `loan_grade` | str | `A` to `G` (A = best, G = riskiest) |
| `loan_amnt` | float | Loan amount in USD |
| `loan_int_rate` | float | Interest rate % (typically 5–23) |
| `loan_percent_income` | float | Loan ÷ income, e.g. `0.18` for 18% |
| `cb_person_default_on_file` | str | `Y` or `N` — prior default record |
| `cb_person_cred_hist_length` | int | Years of credit history |

---

## Run Locally

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/credit-risk-score.git
cd credit-risk-score

# 2. Install dependencies
uv sync

# 3. (Optional) Re-train the model
uv run python ml_model.py

# 4. Start the server
uv run uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — the UI loads instantly.

API docs available at **http://localhost:8000/docs**

---

## Deploy to Render

> The repo includes `render.yaml` — Render picks it up automatically.

**1. Push to GitHub** (make sure `.pkl` files are included):

```bash
git add main.py requirements.txt render.yaml static/ credit_risk_model.pkl best_threshold.pkl
git commit -m "deploy: credit risk app"
git push origin main
```

**2. Create a Render Web Service:**

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` and sets:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Hit **Deploy**

**3. Done.** Your app goes live at `https://credit-risk-score.onrender.com`

> No URL changes needed in the frontend — the JS uses relative `/predict` which works on any domain automatically.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML Training** | scikit-learn, XGBoost, SHAP, scipy |
| **Data** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Backend API** | FastAPI, Pydantic, uvicorn |
| **Model Persistence** | joblib |
| **Frontend** | Vanilla HTML5 / CSS3 / JavaScript (no frameworks) |
| **Deployment** | Render |
| **Package Manager** | uv |

---

## Dataset

**Credit Risk Dataset** — 32,581 loan records with 11 features and a binary `loan_status` target (0 = no default, 1 = default).

Class distribution: ~78% no default / ~22% default — handled via class weights and `scale_pos_weight`.

---

## License

MIT — do whatever you want with it.

---

<div align="center">

Built with 🖤 · XGBoost + Isotonic Calibration · FastAPI · Pure CSS/JS

</div>

