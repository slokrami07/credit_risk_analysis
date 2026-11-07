from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import os

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model['model'] = joblib.load('credit_risk_model.pkl')
    ml_model['threshold'] = joblib.load('best_threshold.pkl')
    yield
    ml_model.clear()


app = FastAPI(lifespan=lifespan, title="Credit Risk Score API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Applicant age (18–100)")
    person_income: float = Field(..., gt=0, description="Annual income in USD")
    person_home_ownership: str = Field(..., description="RENT | OWN | MORTGAGE | OTHER")
    person_emp_length: float = Field(..., ge=0, le=60, description="Employment length in years")
    loan_intent: str = Field(..., description="PERSONAL | EDUCATION | MEDICAL | VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION")
    loan_grade: str = Field(..., description="A | B | C | D | E | F | G")
    loan_amnt: float = Field(..., gt=0, description="Loan amount in USD")
    loan_int_rate: float = Field(..., ge=0, le=100, description="Interest rate (%)")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan amount as fraction of income")
    cb_person_default_on_file: str = Field(..., description="Y | N — historical default on file")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Credit history length in years")


@app.get("/")
def serve_ui():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Credit Risk Score API — visit /docs for the API docs."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml_model}


@app.post("/predict")
def predict(data: LoanApplication):
    input_df = pd.DataFrame([data.dict()])

    probability = float(ml_model['model'].predict_proba(input_df)[:, 1][0])
    threshold = float(ml_model['threshold'])
    prediction = int(probability >= threshold)

    # Risk band
    if probability < 0.3:
        risk_band = "Low Risk"
        risk_color = "green"
    elif probability < 0.6:
        risk_band = "Moderate Risk"
        risk_color = "amber"
    else:
        risk_band = "High Risk"
        risk_color = "red"

    return {
        "default_probability": round(probability, 4),
        "default_prediction": prediction,
        "threshold": round(threshold, 4),
        "result": "high risk" if prediction == 1 else "low risk",
        "risk_band": risk_band,
        "risk_color": risk_color,
        "confidence_pct": round(probability * 100, 2),
    }
