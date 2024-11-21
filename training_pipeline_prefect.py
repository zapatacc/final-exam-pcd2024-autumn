# Importar librerías estándar
import pandas as pd
import numpy as np
import os
import re

# Importar librerías de aprendizaje automático y procesamiento de texto
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# Importar librerías para manejo del desbalance de clases
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Importar librerías para procesamiento de texto
import nltk

# Importar Prefect para la orquestación
from prefect import task, flow
from prefect.tasks import task_input_hash
from datetime import timedelta

# Importar MLflow para el tracking y registro de modelos
import mlflow
import mlflow.sklearn


# Funcion de preprocesamiento de texto
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar URLs y direcciones de correo electrónico
    text = re.sub(r'http\\S+|www.\\S+|@\\S+', '', text)
    
    # Eliminar números y caracteres especiales
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    
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

# Task para cargar y preprocesar los datos
# from prefect import task

@task
def load_and_preprocess_data():
    # Cargar los datos limpios
    df = pd.read_csv('./data/clean_data/cleaned_tickets.csv')
    
    # Preprocesamiento de texto
    df['clean_complaint'] = df['complaint_what_happened'].apply(preprocess_text)
    
    # Codificación de etiquetas
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['ticket_classification'])
    
    return df, label_encoder

# Task para vectorizar y dividir los datos en train y test
@task
def vectorize_and_split_data(df):
    # Separar características y etiquetas
    X = df['clean_complaint']
    y = df['label_encoded']

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=11
    )

    # Vectorización con TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# Task para balancear los datos usando SMOTE y RandomUnderSampler
@task
def balance_data(X_train_tfidf, y_train):
    # Filtrar clases con suficientes muestras para SMOTE
    min_samples = 3
    class_counts = np.bincount(y_train)
    valid_classes = np.where(class_counts >= min_samples)[0]

    # Filtrar los datos
    filtered_indices = np.isin(y_train, valid_classes)
    X_train_filtered = X_train_tfidf[filtered_indices]
    y_train_filtered = y_train[filtered_indices]

    # Pipeline de resampling
    smote = SMOTE(random_state=11, k_neighbors=2)
    undersample = RandomUnderSampler(random_state=11)

    resample_pipeline = Pipeline(steps=[
        ('smote', smote),
        ('undersample', undersample)
    ])

    X_train_resampled, y_train_resampled = resample_pipeline.fit_resample(
        X_train_filtered, y_train_filtered
    )

    return X_train_resampled, y_train_resampled

# Task para entrenar el modelo y registrar los resultados en MLflow
@task
def train_and_log_model(
    X_train_resampled,
    y_train_resampled,
    X_test_tfidf,
    y_test,
    label_encoder,
    experiment_name
):
    # Configurar MLflow
    mlflow.set_experiment(experiment_name)

    # Definir el modelo y los hiperparámetros
    logreg = LogisticRegression(max_iter=1000)
    params = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs'
    }

    with mlflow.start_run(run_name='Logistic Regression with Prefect'):
        # Entrenamiento del modelo
        logreg.set_params(**params)
        logreg.fit(X_train_resampled, y_train_resampled)

        # Predicciones
        y_pred = logreg.predict(X_test_tfidf)

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Obtener todas las etiquetas posibles
        all_labels = label_encoder.transform(label_encoder.classes_)

        # Reporte de clasificación
        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            labels=all_labels,
            zero_division=0
        )

        # Logging de parámetros y métricas
        mlflow.log_params(params)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('f1_macro', f1_macro)
        mlflow.log_metric('f1_weighted', f1_weighted)

        # Logging del modelo
        mlflow.sklearn.log_model(logreg, 'logistic_regression_model')

        # Imprimir resultados
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score (Macro): {f1_macro}")
        print(f"F1 Score (Weighted): {f1_weighted}")
        print("\nReporte de Clasificación:")
        print(report)

# Correr en la terminal el comando 'prefect server start' para iniciar el servidor de Prefect

@flow(name="Training Pipeline")
def training_pipeline():
    # Cargar y preprocesar los datos
    df, label_classes = load_and_preprocess_data()

    # Vectorizar y dividir los datos
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = vectorize_and_split_data(df)

    # Balancear los datos
    X_train_resampled, y_train_resampled = balance_data(X_train_tfidf, y_train)

    # Entrenar y registrar el modelo
    train_and_log_model(
        X_train_resampled,
        y_train_resampled,
        X_test_tfidf,
        y_test,
        label_encoder,
        experiment_name='Logistic Regression Prefect'
    )

# Ejecutar el flow
if __name__ == "__main__":
    training_pipeline()