from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = FastAPI()

# Definir el modelo de datos para la solicitud
class Complaint(BaseModel):
    text: str

# Cargar el vectorizador y el modelo
vectorizer = mlflow.sklearn.load_model('models:/tfidf_vectorizer/production')
model = mlflow.sklearn.load_model('models:/diego-mercado-modelo-2/production')

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

@app.post("/predict")
def predict(complaint: Complaint):
    # Preprocesar el texto
    clean_text = preprocess_text(complaint.text)
    # Vectorizar el texto
    X = vectorizer.transform([clean_text])
    # Realizar la predicción
    prediction = model.predict(X)
    # Obtener la etiqueta correspondiente
    label_encoder = mlflow.sklearn.load_model('models:/label_encoder/production')
    predicted_label = label_encoder.inverse_transform(prediction)
    return {"prediction": predicted_label[0]}

    # Usar Uvicorn para ejecutar la API

    #uvicorn api:app --reload --host 0.0.0.0 --port 8000