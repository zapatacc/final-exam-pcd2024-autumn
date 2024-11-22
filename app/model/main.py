import pickle
import mlflow
import dagshub
import pandas as pd
import mlflow.artifacts
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow.tracking import MlflowClient


app = FastAPI()

mlflow.set_tracking_uri("https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow")

class InputData(BaseModel):
    complaint_what_happened: str

client = MlflowClient()
alias = client.get_model_version_by_alias(name="dafne-model", alias='champion')
runID = alias.run_id
modelURI = f"runs:/{runID}/best_model_Logistic Regression"

model = mlflow.pyfunc.load_model(model_uri=modelURI)

with open(mlflow.artifacts.download_artifacts(run_id=runID, artifact_path='label_encoder.pkl'), "rb") as file:
    label_encoder = pickle.load(file)

with open(mlflow.artifacts.download_artifacts(run_id=runID, artifact_path='tfidf.pkl'), "rb") as file:
    tfidf_vectorizer = pickle.load(file)

def predict(data: dict):
    df = pd.DataFrame([data])

    X_transformed = tfidf_vectorizer.transform(df["complaint_what_happened"])

    y_pred = model.predict(X_transformed)

    return label_encoder.inverse_transform(y_pred)

@app.post("/predict")
def send_predictions(input_data: InputData):
    to_predict = input_data.dict()

    y_pred = predict(to_predict)

    return {"prediction": y_pred.tolist()}