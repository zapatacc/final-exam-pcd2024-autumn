from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pickle
from mlflow.artifacts import download_artifacts


mlflow.set_tracking_uri('https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow')

# Cargamos el modelo
logged_model = 'runs:/92798365c52745fb8a52d4d6b60b7d5d/model-lr'
model = mlflow.pyfunc.load_model(logged_model)
# Descargamos los artefactos
label_encoder_path = download_artifacts(run_id="92798365c52745fb8a52d4d6b60b7d5d",
                                        artifact_path="LabelEncoder/labelencoder.pkl")
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

vectorizer_path = download_artifacts(run_id="92798365c52745fb8a52d4d6b60b7d5d",
                                     artifact_path="Vectorizer/vectorizer.pkl")
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
