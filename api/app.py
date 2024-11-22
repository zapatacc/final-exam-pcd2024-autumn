from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pickle
import os

# Configuración de conexión a DagsHub y MLflow
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow"

# Configuración del modelo
MODEL_NAME = "Francisco-LogisticRegression"

# Crear la aplicación FastAPI
app = FastAPI()

# Intentar cargar el modelo y el vectorizador desde DagsHub
try:
    print("Intentando cargar el modelo y el vectorizador desde DagsHub...")
    # Especificar la versión del modelo
    model_uri = f"models:/{MODEL_NAME}/2"
    model = mlflow.sklearn.load_model(model_uri)
    print("Modelo Champion cargado exitosamente desde DagsHub.")

    # Cargar el vectorizador desde los artefactos del modelo
    vectorizer_uri = f"models:/Francisco-LogisticRegression/2/artifacts/preprocessing/vectorizer.pkl"
    vectorizer = pickle.loads(mlflow.artifacts.download_artifacts(vectorizer_uri))
    print("Vectorizador cargado exitosamente desde DagsHub.")
except Exception as e:
    print(f"Error al cargar desde DagsHub, intentando cargar localmente: {e}")
    with open("artifacts/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("Vectorizador cargado exitosamente desde el sistema local.")

# Modelo de entrada para la API
class PredictionRequest(BaseModel):
    complaint_what_happened: str  # Entrada de texto para el modelo

# Ruta principal para verificar el estado
@app.get("/")
def read_root():
    """Verifica que la API esté funcionando correctamente."""
    return {"status": "API is running", "model_status": "Loaded"}

# Ruta para realizar predicciones
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Validar entrada
        if not request.complaint_what_happened.strip():
            raise HTTPException(status_code=400, detail="El texto proporcionado está vacío.")

        # Preprocesar la entrada con el vectorizador
        input_text = [request.complaint_what_happened]
        input_vectorized = vectorizer.transform(input_text)

        # Realizar predicción
        prediction = model.predict(input_vectorized)
        probability = model.predict_proba(input_vectorized).max()

        return {
            "prediction": prediction[0],
            "probability": probability
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")