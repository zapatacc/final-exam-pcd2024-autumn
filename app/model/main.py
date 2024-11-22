import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient

# Configuración de la URI de seguimiento de MLflow
MLFLOW_TRACKING_URI = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow"

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Obtener el mejor experimento Champion
run_ = mlflow.search_runs(
    order_by=["metrics.accuracy DESC"],
    output_format="list",
    experiment_names=["Tinoco-random_forest-modelo-prefect"]
)[0]
run_id = run_.info.run_id

# Descargar el preprocesador
client.download_artifacts(
    run_id=run_id,
    path="preprocessor",
    dst_path="."
)

# Cargar el preprocesador
with open("preprocessor/preprocessor.b", "rb") as f_in:
    dv = pickle.load(f_in)

# Cargar el modelo Champion
model_name = "Tinoco-random_forest-modelo-prefect"
alias = "champion"
model_uri = f"models:/{model_name}@{alias}"

champion_model = mlflow.pyfunc.load_model(model_uri=model_uri)

# Función de preprocesamiento
def preprocess(input_data):
    """
    Convierte los datos de entrada a una representación adecuada para el modelo.
    """
    input_dict = {
        "complaint_what_happend": input_data.complaint_what_happend
    }

    return dv.transform(input_dict)

# Función de predicción
def predict(input_data):
    """
    Realiza la predicción usando el modelo Champion.
    """
    X_pred = preprocess(input_data)
    return champion_model.predict(X_pred)

# Definición de la API
app = FastAPI()

# Clase para los datos de entrada
class InputData(BaseModel):
    complaint_what_happend: str

@app.get("/")
def greet():
    """
    Endpoint de prueba para verificar que la API está activa.
    """
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(input_data: InputData):
    """
    Endpoint para realizar predicciones.
    """
    result = predict(input_data)[0]
    return {"ticket_classification": str(result)}


