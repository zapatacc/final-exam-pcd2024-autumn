# import os
# import sys
# from fastapi import FastAPI
# from pydantic import BaseModel
# import mlflow
# import joblib
# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import logging

# # Configurar el registro de logs
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Descargar recursos necesarios de NLTK
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# app = FastAPI()

# class Complaint(BaseModel):
#     text: str

# # Variables globales para modelos
# vectorizer = None
# model = None
# label_encoder = None

# # Función para cargar modelos
# def load_models():
#     global vectorizer, model, label_encoder
#     error_messages = []

#     if vectorizer is None:
#         try:
#             vectorizer = joblib.load('models/vectorizer.pkl')
#             logger.info("Vectorizador cargado exitosamente.")
#         except Exception as e:
#             error_messages.append(f"Error al cargar el vectorizador: {e}")

#     if model is None:
#         try:
#             model = mlflow.sklearn.load_model('models:/diego-mercado-modelo-2/versions/5')
#             logger.info("Modelo cargado exitosamente.")
#         except Exception as e:
#             error_messages.append(f"Error al cargar el modelo: {e}")

#     if label_encoder is None:
#         try:
#             label_encoder = joblib.load('models/label_encoder.pkl')
#             logger.info("Label encoder cargado exitosamente.")
#         except Exception as e:
#             error_messages.append(f"Error al cargar el label encoder: {e}")

#     return error_messages

# # Función de preprocesamiento
# def preprocess_text(text):
#     # Implementa tu lógica de preprocesamiento aquí
#     return text

# @app.post("/predict")
# def predict(complaint: Complaint):
#     error_messages = load_models()
#     if error_messages:
#         logger.error(f"Errores al cargar los modelos: {error_messages}")
#         return {"error": "Error al cargar modelos.", "details": error_messages}

#     try:
#         clean_text = preprocess_text(complaint.text)
#         X = vectorizer.transform([clean_text])
#         prediction = model.predict(X)
#         predicted_label = label_encoder.inverse_transform(prediction)
#         return {"prediction": predicted_label[0]}
#     except Exception as e:
#         logger.error(f"Error durante la predicción: {e}")
#         return {"error": "Error durante la predicción.", "details": str(e)}

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
import sys
import joblib

# Configurar el registro de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar MLflow Tracking URI y credenciales
mlflow.set_tracking_uri("https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow")
# os.environ["MLFLOW_TRACKING_USERNAME"] = "tu_usuario"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "tu_token_o_contraseña"

# Descargar recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Variables globales para modelos
vectorizer = None
model = None
label_encoder = None


# Función para cargar modelos
def load_models():
    global vectorizer, model, label_encoder
    error_messages = []

    if vectorizer is None:
        try:
            vectorizer = joblib.load('models/vectorizer.pkl')
            logger.info("Vectorizador cargado exitosamente.")
        except Exception as e:
            error_messages.append(f"Error al cargar el vectorizador: {e}")

    if model is None:
        # Cargar el modelo
        try:
            model_name = "diego-mercado-modelo-2"
            model_version = 5  # Actualiza según corresponda
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Modelo cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            model = None

    if label_encoder is None:
        try:
            label_encoder = joblib.load('models/label_encoder.pkl')
            logger.info("Label encoder cargado exitosamente.")
        except Exception as e:
            error_messages.append(f"Error al cargar el label encoder: {e}")

    return error_messages




# Función de preprocesamiento
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

app = FastAPI()

class Complaint(BaseModel):
    text: str

@app.post("/predict")
def predict(complaint: Complaint):
    error_messages = load_models()
    if error_messages:
        logger.error(f"Errores al cargar los modelos: {error_messages}")
        return {"error": "Error al cargar modelos.", "details": error_messages}

    try:
        clean_text = preprocess_text(complaint.text)
        X = vectorizer.transform([clean_text])
        prediction = model.predict(X)
        predicted_label = label_encoder.inverse_transform(prediction)
        return {"prediction": predicted_label[0]}
    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        return {"error": "Error durante la predicción.", "details": str(e)}

