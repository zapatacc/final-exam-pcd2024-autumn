from fastapi import FastAPI
import mlflow
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    data: list

# Cargar el modelo Champion
model = mlflow.sklearn.load_model('models:/carlos-moreno-modelo/Production')

@app.post("/predict")
def predict(request: PredictionRequest):
    # Convertir los datos de entrada en un DataFrame
    input_data = pd.DataFrame(request.data)
    # Realizar predicción
    predictions = model.predict(input_data)
    # Devolver resultados
    return {"predictions": predictions.tolist()}
