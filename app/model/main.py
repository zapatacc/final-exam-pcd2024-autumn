from pydantic import BaseModel
import mlflow
import pandas as pd
import pickle
from fastapi import FastAPI
from mlflow.tracking import MlflowClient

# Configura la URI de MLflow
mlflow.set_tracking_uri('https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow')

app = FastAPI()

class InputData(BaseModel):
    complaint_what_happened: str

# Función para obtener el run_id desde un alias específico
def get_run_id_from_alias(model_name: str, alias: str) -> str:
    client = MlflowClient()
    # Obtén la información del alias asociado al modelo
    alias_info = client.get_model_version_by_alias(name=model_name, alias=alias)
    return alias_info.run_id

def predict(data: dict):
    model_name = "patricio-model"
    alias = "champion"

    # Obtén el run_id asociado al alias
    run_id = get_run_id_from_alias(model_name=model_name, alias=alias)

    # Construye la URI del modelo
    model_uri = f"runs:/{run_id}/pipeline_model"

    # Carga el modelo desde MLflow
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

    # Descarga y carga el LabelEncoder asociado
    label_encoder_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="label_encoder.pkl")
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Preprocesa los datos de entrada
    df = pd.DataFrame([data])
    prediction_encoded = loaded_model.predict(df["complaint_what_happened"])

    # Decodifica la predicción
    prediction_decoded = label_encoder.inverse_transform(prediction_encoded)

    return prediction_decoded

@app.post("/predict")
def predict_endpoint(input_data: InputData):
    input_dict = input_data.dict()

    result = predict(input_dict)

    return {
        "prediction": result[0]
    }
