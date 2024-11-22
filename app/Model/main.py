from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pickle
from mlflow.artifacts import download_artifacts
from mlflow import MlflowClient


mlflow.set_tracking_uri('https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow')

model_name = "arturo-prefect-model"
model_version_or_stage = "champion"

# Usamos la función de MLflow para cargar el modelo por alias
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version_or_stage}")

# Descargamos los artefactos asociados al modelo
label_encoder_path = download_artifacts(
    artifact_path="LabelEncoder/labelencoder.pkl",
    artifact_uri=f"models:/{model_name}/{model_version_or_stage}"
)
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

vectorizer_path = download_artifacts(
    artifact_path="Vectorizer/vectorizer.pkl",
    artifact_uri=f"models:/{model_name}/{model_version_or_stage}"
)
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Inicializamos FastAPI
app = FastAPI()


# Modelo de datos para el input
class InputData(BaseModel):
    text: str


# Endpoint para realizar predicciones
@app.post("/predict")
def predict(input_data: InputData):
    try:
        input_vector = vectorizer.transform([input_data.text])

        prediction = model.predict(input_vector)

        resultado = label_encoder.inverse_transform(prediction)[0]

        return {"prediction": resultado}
    except Exception as e:
        # Retorna un error bien definido
        return {"error": f"Ocurrio un error: {str(e)}"}
