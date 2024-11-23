
import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient
import dagshub
from pydantic import BaseModel



dagshub_repo = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn"
MLFLOW_TRACKING_URI = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow"
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

run_uri = 'runs:/349a456f84c04ccb96759c2d69458666/logistic_regression_model'


dv = mlflow.pyfunc.load_model(run_uri)


def preprocess(input_complaint):
    return input_complaint.complaint_what_happened

def predict(input_data):
    X_pred = preprocess(input_data)
    return dv.predict([X_pred])


app = FastAPI()

class InputData(BaseModel):
    complaint_what_happened: str


@app.post("/predict")
def predict_endpoint(input_data: InputData):
    result = predict(input_data)[0]
    return {
        "ypred": [result]  # Cambia la clave a "ypred"
    }
