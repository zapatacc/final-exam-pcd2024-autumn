import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# Definir la ruta base del contenedor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Configurar el registro de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar MLflow Tracking URI
mlflow.set_tracking_uri("https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow")

# Descargar recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Variables globales para modelos y artefactos
vectorizer = None
model = None
label_encoder = None


# Función para cargar modelos y artefactos
def load_models():
    global vectorizer, model, label_encoder
    error_messages = []

    # Cargar el vectorizador
    if vectorizer is None:
        try:
            vectorizer = joblib.load(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'))
            logger.info("TF-IDF Vectorizer cargado exitosamente.")
        except Exception as e:
            error_messages.append(f"Error al cargar el vectorizador: {e}")

    # Cargar el modelo desde el Model Registry
    if model is None:
        try:
            model_name = "colome-prefect-modelos"
            model_alias = "Champion"  # Usar el alias Champion
            model_uri = f"models:/{model_name}@{model_alias}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Modelo Champion cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            model = None

    # Cargar el label encoder
    if label_encoder is None:
        try:
            label_encoder = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))
            logger.info("Label Encoder cargado exitosamente.")
        except Exception as e:
            error_messages.append(f"Error al cargar el label encoder: {e}")

    return error_messages


# Función de preprocesamiento de texto
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()

    # Eliminar URLs y correos electrónicos
    text = re.sub(r'http\S+|www.\S+|@\S+', '', text)
    
    # Eliminar números y caracteres especiales
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenización
    tokens = text.split()
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lematización
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Reconstruir el texto
    text = ' '.join(tokens)
    return text


# Crear la aplicación FastAPI
app = FastAPI()


# Definir el esquema para las solicitudes de predicción
class Complaint(BaseModel):
    text: str


@app.post("/predict")
def predict(complaint: Complaint):
    """
    Endpoint para realizar una predicción utilizando el modelo Champion.
    """
    # Cargar modelos y artefactos si no están ya cargados
    error_messages = load_models()
    if error_messages:
        logger.error(f"Errores al cargar los modelos: {error_messages}")
        raise HTTPException(status_code=500, detail={"error": "Error al cargar modelos", "details": error_messages})

    try:
        # Preprocesar el texto de entrada
        clean_text = preprocess_text(complaint.text)
        # Transformar el texto con el vectorizador
        X = vectorizer.transform([clean_text])
        # Realizar la predicción
        prediction = model.predict(X)
        # Decodificar la etiqueta predicha
        predicted_label = label_encoder.inverse_transform(prediction)
        return {"prediction": predicted_label[0]}
    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=500, detail={"error": "Error durante la predicción", "details": str(e)})



# uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug