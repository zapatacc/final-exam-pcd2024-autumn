import pathlib
import dagshub
from prefect import task, flow
import pandas as pd
import numpy as np
import os
import json
import pickle
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import nltk
import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

import warnings

warnings.filterwarnings('ignore')

# Iniciamos dagshub
dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)


# Función para expandir contracciones
def expand_contractions(text):
    return contractions.fix(text)


# Función para limpiar el texto
def clean_text(text):
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r'xx+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Función para preprocesar el texto
def preprocess_text(text):
    text = clean_text(text)
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)
    return text


# Función para filtrar categorías con pocas ocurrencias
def filter_categories(df, min_frequency=80):
    value_counts = df['ticket_classification'].value_counts()
    categories_to_keep = value_counts[value_counts >= min_frequency].index
    filtered_df = df[df['ticket_classification'].isin(categories_to_keep)]
    return filtered_df


# Tarea para cargar y preparar los datos
@task(task_run_name='Data Cleaning and loading')
def load_and_prepare_data():
    file_path = '../data/raw_data/tickets_classification_eng.json'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe")
    else:
        with open(file_path, "r") as file:
            datos = json.load(file)

    # Convertimos los datos JSON a un DataFrame de pandas
    df = pd.json_normalize(datos)

    # Seleccionamos y renombramos las columnas
    df_clean = df[['_source.complaint_what_happened', '_source.product', '_source.sub_product']]
    df_clean.rename(columns={
        '_source.complaint_what_happened': 'complaint_what_happened',
        '_source.product': 'category',
        '_source.sub_product': 'sub_product'
    }, inplace=True)

    # Creamos una nueva columna combinando 'category' y 'sub_product'
    df_clean['ticket_classification'] = df_clean['category'] + ' + ' + df_clean['sub_product']

    # Eliminamos las columnas que ya no necesitamos
    df_clean.drop(['category', 'sub_product'], axis=1, inplace=True)

    # Reemplazamos cadenas vacías por NaN
    df_clean['complaint_what_happened'].replace('', pd.NA, inplace=True)

    # Eliminamos filas con datos faltantes
    df_clean.dropna(subset=['complaint_what_happened', 'ticket_classification'], inplace=True)

    # Reiniciamos el índice del DataFrame
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean


# Tarea para preprocesar los datos
@task(task_run_name='Data Preprocessing')
def preprocess_data(df):
    df['complaint_what_happened'] = df['complaint_what_happened'].apply(preprocess_text)
    return df


# Tarea para filtrar los datos según la frecuencia de las categorías
@task(task_run_name='Data Filtering')
def filter_data(df, min_frequency=80):
    df_filtered = filter_categories(df, min_frequency)
    return df_filtered


# Tarea para dividir el train test
@task(task_run_name='Split Data')
def split_data(df):
    X = df['complaint_what_happened']
    y = df['ticket_classification']

    label_encoder = LabelEncoder()

    # Codificamos las etiquetas (texto a números)
    y_encoded = label_encoder.fit_transform(y)

    # Guardamos el LabelEncoder para usarlo luego en la predicción
    pathlib.Path("models").mkdir(exist_ok=True)
    with open("models/label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)

    # Dividimos los datos en train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test


# Tarea para entrenar y evaluar los modelos
@task(name='Train and Test')
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    # Inicializamos el vectorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    # Ajustamos y transformamos los datos de entrenamiento
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Transformamos los datos de prueba
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Diccionario para almacenar resultados
    models_results = {}

    # 1. Regresión Logística Básica
    lr_baseline = LogisticRegression(max_iter=1000)
    lr_baseline.fit(X_train_tfidf, y_train)
    y_pred_lr_baseline = lr_baseline.predict(X_test_tfidf)
    accuracy_lr_baseline = accuracy_score(y_test, y_pred_lr_baseline)
    models_results['LogisticRegression_Baseline'] = {
        'model': lr_baseline,
        'accuracy': accuracy_lr_baseline,
        'params': lr_baseline.get_params()
    }

    # 2. Regresión Logística con Ajuste de Hiperparámetros
    param_grid_lr = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    grid_search_lr = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid_lr,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_lr.fit(X_train_tfidf, y_train)
    best_lr_model = grid_search_lr.best_estimator_
    y_pred_lr_tuned = best_lr_model.predict(X_test_tfidf)
    accuracy_lr_tuned = accuracy_score(y_test, y_pred_lr_tuned)
    models_results['LogisticRegression_Tuned'] = {
        'model': best_lr_model,
        'accuracy': accuracy_lr_tuned,
        'params': grid_search_lr.best_params_
    }

    # 3. SVC Básico
    svc_baseline = SVC()
    svc_baseline.fit(X_train_tfidf, y_train)
    y_pred_svc_baseline = svc_baseline.predict(X_test_tfidf)
    accuracy_svc_baseline = accuracy_score(y_test, y_pred_svc_baseline)
    models_results['SVC_Baseline'] = {
        'model': svc_baseline,
        'accuracy': accuracy_svc_baseline,
        'params': svc_baseline.get_params()
    }

    # 4. SVC con Ajuste de Hiperparámetros
    param_grid_svc = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid_search_svc = GridSearchCV(
        SVC(),
        param_grid_svc,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_svc.fit(X_train_tfidf, y_train)
    best_svc_model = grid_search_svc.best_estimator_
    y_pred_svc_tuned = best_svc_model.predict(X_test_tfidf)
    accuracy_svc_tuned = accuracy_score(y_test, y_pred_svc_tuned)
    models_results['SVC_Tuned'] = {
        'model': best_svc_model,
        'accuracy': accuracy_svc_tuned,
        'params': grid_search_svc.best_params_
    }
    # Guardamos el vectorizador
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    return models_results


EXPERIMENT_NAME = 'blanco-all-models-prefect'


# Tarea para registrar los modelos en MLflow
@task(name="log")
def log_models(models_results):
    # Configuramos MLflow
    mlflow.set_tracking_uri('https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow')
    client = MlflowClient()

    # Establecemos el experimento
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Lista para almacenar información de los runs
    run_infos = []

    # Registramos cada modelo y sus métricas
    for model_name, result in models_results.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(result['params'])
            mlflow.log_metric('accuracy', result['accuracy'])
            mlflow.sklearn.log_model(
                sk_model=result['model'],
                artifact_path='model'
            )
            pathlib.Path("models").mkdir(exist_ok=True)

            # Guardamos el LabelEncoder y el vectorizador como artefactos
            mlflow.log_artifact("models/label_encoder.pkl", artifact_path="LabelEncoder")
            mlflow.log_artifact("models/tfidf_vectorizer.pkl", artifact_path="Vectorizer")
            # Obtenemos el run_id y guardamos la info
            run_id = mlflow.active_run().info.run_id
            run_infos.append({
                'model_name': model_name,
                'run_id': run_id,
                'accuracy': result['accuracy']
            })

    # Ordenamos los runs por accuracy para encontrar nuestro champion
    sorted_runs = sorted(run_infos, key=lambda x: x['accuracy'], reverse=True)

    # Nombre del modelo
    registered_model_name = 'blanco-models-prefect'

    # Registramos el Champion
    champion_run = sorted_runs[0]
    champion_model_uri = f"runs:/{champion_run['run_id']}/model"

    champion_model_version = mlflow.register_model(
        model_uri=champion_model_uri,
        name=registered_model_name
    )

    client.set_registered_model_alias(
        name=registered_model_name,
        alias='champion',
        version=champion_model_version.version
    )

    # Si hay más de un modelo, registramos de los que quedan el siguiente mejor (Challenger)
    if len(sorted_runs) > 1:
        challenger_run = sorted_runs[1]
        challenger_model_uri = f"runs:/{challenger_run['run_id']}/model"

        challenger_model_version = mlflow.register_model(
            model_uri=challenger_model_uri,
            name=registered_model_name
        )

        client.set_registered_model_alias(
            name=registered_model_name,
            alias='challenger',
            version=challenger_model_version.version
        )

    # Imprimimos los resultados
    print(
        f"Champion Model: {champion_run['model_name']}, Run ID: {champion_run['run_id']}, Version: {champion_model_version.version}")
    if len(sorted_runs) > 1:
        print(
            f"Challenger Model: {challenger_run['model_name']}, Run ID: {challenger_run['run_id']}, Version: {challenger_model_version.version}")


# Definimos el flujo principal
@flow()
def main():
    df_clean = load_and_prepare_data()
    df_preprocessed = preprocess_data(df_clean)
    df_filtered = filter_data(df_preprocessed, min_frequency=80)
    X_train, X_test, y_train, y_test = split_data(df_filtered)
    models_results = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    log_models(models_results)


# Ejecutamos el flujo principal
if __name__ == '__main__':
    main()
