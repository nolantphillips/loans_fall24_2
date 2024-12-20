from typing import Union

from fastapi import FastAPI
import dill
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()

with open('./app/xgb_model_v2.pkl', 'rb') as f:
    reloaded_model = dill.load(f)

class Payload(BaseModel):
    Age: float
    Gender: str
    Education: str
    HomeOwnership: str
    Intent: str
    Income: float
    YearsExperience: float
    LoanAmount: float
    InterestRate: float
    PercentIncome: float
    CreditScore: float
    YearsCreditHistory: float
    PreviousLoanDefault: str

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "Name": "Nolan Phillips",
        "Project": "Predicting Loan Acceptance/Denial",
        "Model": "XGBoost Classifier"
    }

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
def predict(payload: Payload):
    df = pd.DataFrame([payload.model_dump().values()], columns=payload.model_dump().keys())
    y_hat = reloaded_model.predict(df)
    return {"prediction": y_hat[0].item()}