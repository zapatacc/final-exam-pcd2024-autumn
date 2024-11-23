import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient
import numpy as np

# MLflow settings

MLFLOW_TRACKING_URI = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow"

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


model_name = "juan-ley-random-forest-prefect"
alias = "champion"

model_uri = f"models:/{model_name}@{alias}"

model = mlflow.pyfunc.load_model(
    model_uri=model_uri
)


# Cargar el vectorizador TF-IDF
tfidf_path = r"C:\Users\jplv0\PycharmProjects\final-exam-pcd2024-autumn\training_pipeline\models\tfidf_vectorizer.pkl"
with open(tfidf_path, "rb") as file:
    tfidf_vectorizer = pickle.load(file)

# Cargar el label encoder
label_encoder_path = r"C:\Users\jplv0\PycharmProjects\final-exam-pcd2024-autumn\training_pipeline\models\label_encoder2.pkl"
with open(label_encoder_path, "rb") as file:
    label_encoder = pickle.load(file)


def preprocess(input_data):
    input_text = input_data.complaint_what_happened
    X_pred = tfidf_vectorizer.transform([input_text])
    return X_pred

# Definir la función de predicción
def predict(input_data):
    X_pred = preprocess(input_data)
    y_pred = model.predict(X_pred)
    y_pred_label =label_encoder.fit_transform(y_pred)
    return y_pred_label

# Definir la aplicación FastAPI
app = FastAPI(title="API para Servir el Modelo Champion", description="Una API simple para servir el modelo Champion para realizar inferencias.")

# Definir el esquema de datos de entrada usando Pydantic
class ModelInput(BaseModel):
    complaint_what_happened: str

# Definir la ruta principal para la inferencia
@app.post("/predict/")
async def predict_endpoint(input_data: ModelInput):
    # Realizar la predicción usando el modelo Champion
    result = predict(input_data)[0]
    # Retornar el resultado de la predicción
    return {"prediction": str(result)}




