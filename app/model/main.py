from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import pickle

# Inicializa FastAPI
app = FastAPI()

mlflow.set_tracking_uri('https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow')


# URI del modelo en MLflow
logged_model = 'runs:/5029198c9d1140fe9584c6352a0c58f2/SVC-maripau'

# Carga el modelo como un PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Descarga y carga los artefactos adicionales
with open(mlflow.artifacts.download_artifacts(run_id="5029198c9d1140fe9584c6352a0c58f2", artifact_path="TfidfVectorizer.pkl"), "rb") as file:
    tfidf_vectorizer = pickle.load(file)

with open(mlflow.artifacts.download_artifacts(run_id="5029198c9d1140fe9584c6352a0c58f2", artifact_path="LabelEncoder.pkl"), "rb") as file:
    label_encoder = pickle.load(file)

# Clase para los datos de entrada
class InputData(BaseModel):
    complaint_what_happened: str

# Función de predicción
def predict(data: dict):
    # Convierte los datos de entrada en un DataFrame
    df = pd.DataFrame([data])
    
    # Aplica el vectorizador TF-IDF al texto de entrada
    X_transformed = tfidf_vectorizer.transform(df["complaint_what_happened"])
    
    # Realiza la predicción con el modelo cargado
    y_pred = loaded_model.predict(X_transformed)
    
    # Convierte las etiquetas predichas a su representación original
    return label_encoder.inverse_transform(y_pred)

# Endpoint de predicción
@app.post("/predict")
def send_predictions(input_data: InputData):
    # Convierte los datos de entrada a un diccionario
    to_predict = input_data.dict()
    
    # Realiza la predicción
    y_pred = predict(to_predict)

    # Devuelve el resultado en formato JSON
    return {"ypred": y_pred.tolist()}