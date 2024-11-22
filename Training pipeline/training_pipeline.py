import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import dagshub
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import pickle
import pathlib
from mlflow.tracking import MlflowClient
from prefect import task,flow
import json
import contractions
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

@task(name = "data wrangling")
def data_wrangling(file_path):
    """Carga el archivo JSON y lo convierte en un DataFrame limpio para modelar."""
    with open(file_path, "r") as file:
        data = json.load(file)
    df = pd.json_normalize(data)

    # Seleccionamos columnas necesarias
    df = df[['_source.complaint_what_happened', '_source.product', '_source.sub_product']]

    # Renombramos columnas
    df.rename(columns={
        '_source.complaint_what_happened': 'complaint_what_happened',
        '_source.product': 'category',
        '_source.sub_product': 'sub_product'
    }, inplace=True)

    # Creamos columna 'ticket_classification'
    df['ticket_classification'] = df['category'] + ' + ' + df['sub_product']

    # Eliminamos las columnas innecesarias
    df.drop(columns=['category', 'sub_product'], inplace=True)

    # Reemplazamos valores vacíos con valore nulos
    df['complaint_what_happened'].replace('', pd.NA, inplace=True)

    # Eliminamos filas con datos faltantes
    df.dropna(subset=['complaint_what_happened', 'ticket_classification'], inplace=True)

    # Reiniciamos el  índice
    df.reset_index(drop=True, inplace=True)

    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        # Expandir contracciones
        text = contractions.fix(text)

        # Eliminar 'X' mayúsculas
        text = re.sub(r'X+', '', text)

        # Convertir a minúsculas
        text = text.lower()

        # Tokenizar texto
        words = nltk.word_tokenize(text)

        # Eliminar stopwords
        words = [word for word in words if word not in stop_words]

        # Unir palabras
        return ' '.join(words)

    df['complaint_what_happened'] = df['complaint_what_happened'].apply(clean_text)

    counts = df['ticket_classification'].value_counts()
    valid_categories = counts[counts >= 50].index
    df = df[df['ticket_classification'].isin(valid_categories)]
    df = df.drop_duplicates()

    return df
@task(name = "split data")
def split_data(df):
    # Separamos nuestras variables
    X = df['complaint_what_happened']
    y = df['ticket_classification']
    label_encoder = LabelEncoder()
    pathlib.Path("models").mkdir(exist_ok=True)
    with open ("models/labelencoder.pkl","wb") as file:
        pickle.dump(label_encoder, file)

    y = label_encoder.fit_transform(y)

    # Dividimos los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorizamos los datos
    vectorizer = TfidfVectorizer(max_features=5000)
    pathlib.Path("models").mkdir(exist_ok=True)
    with open ("models/vectorizer.pkl","wb") as file:
        pickle.dump(vectorizer, file)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test

@task(name = "Hyper-Parameter Tunning")
def hyper_parameter_tunning(X_train, X_test, y_train, y_test):
    def objective_lr(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "LogisticRegression-prefect")
            mlflow.log_params(params)

            model = LogisticRegression(**params, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model-lr")
            mlflow.log_artifact("models/labelencoder.pkl", artifact_path="LabelEncoder")
            mlflow.log_artifact("models/vectorizer.pkl", artifact_path="Vectorizer")

        return {'loss': accuracy, 'status': STATUS_OK}

    with mlflow.start_run(run_name="LogisticRegression Hyper-parameter Optimization"):
        search_space_lr = {
            'C': hp.loguniform('C', -4, 2),
            'solver': hp.choice('solver', ['liblinear', 'lbfgs'])
        }

        best_params_lr = fmin(
            fn=objective_lr,
            space=search_space_lr,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )

        # Convertimos parámetros al formato adecuado
        best_params_lr['solver'] = ['liblinear', 'lbfgs'][best_params_lr['solver']]
        mlflow.log_params(best_params_lr)

    def objective_rf(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "RandomForest-prefect")
            mlflow.log_params(params)

            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, artifact_path="model-rf")
            mlflow.log_artifact("models/labelencoder.pkl", artifact_path="LabelEncoder")
            mlflow.log_artifact("models/vectorizer.pkl", artifact_path="Vectorizer")

        return {'loss': -accuracy, 'status': STATUS_OK}

    with mlflow.start_run(run_name="RandomForest Hyper-parameter Optimization"):
        search_space_rf = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 100, 500, 1)),
            'max_depth': scope.int(hp.quniform('max_depth', 5, 50, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
            'bootstrap': hp.choice('bootstrap', [True, False])
        }

        best_params_rf = fmin(
            fn=objective_rf,
            space=search_space_rf,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )

        # Convertir parámetros al formato adecuado
        best_params_rf['n_estimators'] = int(best_params_rf['n_estimators'])
        best_params_rf['max_depth'] = int(best_params_rf['max_depth'])
        best_params_rf['min_samples_split'] = int(best_params_rf['min_samples_split'])
        best_params_rf['min_samples_leaf'] = int(best_params_rf['min_samples_leaf'])
        best_params_rf['bootstrap'] = bool(best_params_rf['bootstrap'])
        mlflow.log_params(best_params_rf)

    return best_params_lr, best_params_rf

@task(name = "Train Best Models")
def train_best_model(X_train, X_test, y_train, y_test, best_params_lr, best_params_rf) -> None:
    with mlflow.start_run(run_name="Best lr model ever"):
        best_model_lr = LogisticRegression(**best_params_lr, random_state=42)
        best_model_lr.fit(X_train, y_train)

        y_pred_lr = best_model_lr.predict(X_test)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        mlflow.log_metric("accuracy", accuracy_lr)

    with mlflow.start_run(run_name="Best rf model ever"):
        best_model_rf = RandomForestClassifier(**best_params_rf, random_state=42)
        best_model_rf.fit(X_train, y_train)

        y_pred_rf = best_model_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        mlflow.log_metric("accuracy", accuracy_rf)

    pathlib.Path("models").mkdir(exist_ok=True)
    mlflow.log_artifact("models/labelencoder.pkl", artifact_path="LabelEncoder")
    mlflow.log_artifact("models/vectorizer.pkl", artifact_path="Vectorizer")

    return None

@task(name="Register Best Model")
def register_best_model() -> None:
    client = MlflowClient()

    # Declaramos el experimento en el que estamos trabajando
    experiment_name = "arturo-prefect-experiment"

    experiment = client.get_experiment_by_name(experiment_name)

    # Buscamos las dos mejores ejecuciones en base al accuracy
    top_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],  # Cambia a ASC si buscas minimizar
        max_results=2  # Recuperar las dos mejores
    )

    # Obtenemos los IDs de las mejores ejecuciones
    champion_run = top_runs.iloc[0]
    challenger_run = top_runs.iloc[1]

    # Obtenemos los IDs de las ejecuciones
    champion_run_id = champion_run.run_id
    challenger_run_id = challenger_run.run_id

    champion_model_uri = f"runs:/{champion_run_id}/model"
    challenger_model_uri = f"runs:/{challenger_run_id}/model"

    # Declaramos el nombre del modelo registrado
    model_name = "arturo-prefect-model"

    # Registramos el Champion
    champion_model_version = mlflow.register_model(champion_model_uri, model_name)
    client.set_registered_model_alias(model_name, "champion", champion_model_version.version)

    # Registramos el Challenger
    challenger_model_version = mlflow.register_model(challenger_model_uri, model_name)
    client.set_registered_model_alias(model_name, "challenger", challenger_model_version.version)

@flow(name="Main flow")
def main_flow(year: str, month_train: str, month_val: str) -> None:
    file_path = "../Data/raw_data/tickets_classification_eng.json"

    # Inicializar DagsHub y MLflow
    dagshub.init(url="https://dagshub.com/zapatacc/final-exam-pcd2024-autumn", mlflow=True)
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="arturo-prefect-experiment")

    print("MLflow tracking URI:", MLFLOW_TRACKING_URI)

    # Ejecutar las tareas del flujo
    print("Ejecutando tarea: data wrangling")
    df = data_wrangling(file_path)

    print("Ejecutando tarea: split data")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Ejecutando tarea: hyper-parameter tuning")
    best_params_lr, best_params_rf = hyper_parameter_tunning(X_train, X_test, y_train, y_test)

    print("Ejecutando tarea: train best models")
    train_best_model(X_train, X_test, y_train, y_test, best_params_lr, best_params_rf)

    print("Ejecutando tarea: register best model")
    register_best_model()

    print("Flujo completado con éxito.")


main_flow(year="2024", month_train="11", month_val="21")
