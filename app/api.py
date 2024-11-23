from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "LuisFLopezA"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "37d1c615f665c61af97d8a85683704cd5ca42315"




mlflow.set_tracking_uri("https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow")

app = FastAPI()

MODEL_URI = "models:/luis-lopez_champion_naive_bayes/3" 
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"Error loading model from MLflow: {e}")

class PredictionRequest(BaseModel):
    cleaned_complaint: str
    ticket_classification: str

class PredictionResponse(BaseModel):
    predicted_topic: int

@app.get("/")
def read_root():
    return {"message": "API is running. Use /predict to get predictions."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    combined_text = f"{request.cleaned_complaint} {request.ticket_classification}"
    input_df = pd.DataFrame({"text": [combined_text]})

    try:
        
        prediction = model.predict(input_df)
        return PredictionResponse(predicted_topic=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")