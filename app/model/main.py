from pydantic import BaseModel
import mlflow
import pandas as pd
import pickle
from fastapi import FastAPI
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow')

app = FastAPI()

class InputData(BaseModel):
    complaint_what_happened: str

def get_run_id_from_alias(model_name: str, alias: str) -> str:
    client = MlflowClient()
    alias_info = client.get_model_version_by_alias(name=model_name, alias=alias)
    return alias_info.run_id

def predict(data: dict):
    model_name = "patricio-model"
    alias = "champion"

    run_id = get_run_id_from_alias(model_name=model_name, alias=alias)

    model_uri = f"runs:/{run_id}/pipeline_model"

    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

    label_encoder_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="label_encoder.pkl")
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    df = pd.DataFrame([data])
    prediction_encoded = loaded_model.predict(df["complaint_what_happened"])

    prediction_decoded = label_encoder.inverse_transform(prediction_encoded)

    return prediction_decoded

@app.post("/predict")
def predict_endpoint(input_data: InputData):
    input_dict = input_data.dict()

    result = predict(input_dict)

    return {
        "prediction": result[0]
    }
