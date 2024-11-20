from pydantic import BaseModel
import mlflow
import pandas as pd
import pickle
from fastapi import FastAPI

mlflow.set_tracking_uri("file:///tmp/mlruns") 


app = FastAPI()

class InputData(BaseModel):
    complaint_what_happened: str

def predict(data: dict):
    run_id = "1d229578028d4d7ab3e76a36a0bb3f79"  
    logged_model = f"runs:/{run_id}/pipeline_model"

    loaded_model = mlflow.pyfunc.load_model(logged_model)

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
