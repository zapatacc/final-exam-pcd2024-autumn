import pickle
import mlflow
import pathlib
import dagshub
import pandas as pd
import json
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from mlflow import MlflowClient
from hyperopt.pyll import scope
from sklearn.metrics import  root_mean_squared_error, classification_report, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import flow, task
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Configura la URL del experimento en DagsHub
DAGSHUB_URL = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn"

@task(name="Read and clean Data", retries=4, retry_delay_seconds=[1, 4, 8, 16])
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    file_path = "../data/tickets_classification_eng.json"

    with open(file_path, "r", encoding="utf-8") as file:
        datos = json.load(file)

    df = pd.json_normalize(datos)

    columnas = ['_source.complaint_what_happened', '_source.product', '_source.sub_product']
    df = df[columnas]

    # renombramos
    df.columns = ['complaint_what_happened', 'category', 'sub_product']

    # creacion de nueva columna
    df['ticket_classification'] = df['category'] + ' + ' + df['sub_product']

    #dropeamos columnas redundantes
    df = df.drop(columns=['sub_product', 'category'])

    #limpieza de datos
    df['complaint_what_happened'] = df['complaint_what_happened'].replace('', np.nan)

    #eliminacion de datos faltantes
    df = df.dropna(subset=['complaint_what_happened', 'ticket_classification'])

    # Reiniciar Indice
    df = df.reset_index(drop=   True)

    return df


@task(name="Train test split")
def train_test(df: pd.DataFrame):
    # Dividir características y etiquetas
    X = df['complaint_what_happened']
    y = df['ticket_classification']

    # Vectorización de texto usando TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Convertir las etiquetas en valores numéricos
    label_encoder = LabelEncoder()
    pathlib.Path("models").mkdir(exist_ok=True)
    with open("models/label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)
    y_train = label_encoder.fit_transform(y_train)
    y_test = y_test.map(lambda x: x if x in label_encoder.classes_ else 'unknown')
    label_encoder.classes_ = np.append(label_encoder.classes_, 'unknown')
    y_test = label_encoder.transform(y_test)

    return X_train, X_test, y_train, y_test


# Función de ajuste de hiperparámetros de rf
@task(name="Hyperparameter tuning random forest")
def hyper_parameter_tuning_random_forest(X_train, X_test, y_train, y_test):
    mlflow.sklearn.autolog()

    # Definición de la función objetivo
    def objective_rf(params):
        with mlflow.start_run(nested=True):
            # Set model tag
            mlflow.set_tag("model_family", "random_forest")

            # Crear modelo de RandomForest con parámetros específicos
            rf_model = RandomForestClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                min_samples_split=int(params['min_samples_split']),
                min_samples_leaf=int(params['min_samples_leaf']),
                random_state=42
            )

            # Entrenar el modelo
            rf_model.fit(X_train, y_train)

            # Predecir en el conjunto de validación
            y_pred = rf_model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Log accuracy metric
            mlflow.log_metric("accuracy", accuracy)

            return {'loss': -accuracy, 'status': STATUS_OK}

    # Espacio de búsqueda de hiperparámetros para Random Forest
    search_space_rf = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)),
        'max_depth': scope.int(hp.quniform('max_depth', 5, 30, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
    }

    # Ejecución de la optimización de hiperparámetros
    with mlflow.start_run(run_name="Optimización de Hiperparámetros Random Forest", nested=True):
        best_params_rf = fmin(
            fn=objective_rf,
            space=search_space_rf,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )

        # Ajustar los tipos de parámetros según Random Forest
        best_params_rf["n_estimators"] = int(best_params_rf["n_estimators"])
        best_params_rf["max_depth"] = int(best_params_rf["max_depth"])
        best_params_rf["min_samples_split"] = int(best_params_rf["min_samples_split"])
        best_params_rf["min_samples_leaf"] = int(best_params_rf["min_samples_leaf"])

        # Loggear los mejores parámetros en mlflow
        mlflow.log_params(best_params_rf)

    return best_params_rf

# Función de ajuste de hiperparámetros lr
@task(name="Hyperparameter tuning logistic regression")
def hyper_parameter_tuning_logistic_regression(X_train, X_test, y_train, y_test):
    mlflow.sklearn.autolog()

    # Definición de la función objetivo
    def objective_lr(params):
        with mlflow.start_run(nested=True):
            # Set model tag
            mlflow.set_tag("model_family", "logistic_regression")

            # Crear modelo de Logistic Regression con parámetros específicos
            lr_model = LogisticRegression(
                C=params['C'],
                max_iter=int(params['max_iter']),
                solver=params['solver'],
                random_state=42
            )

            # Entrenar el modelo
            lr_model.fit(X_train, y_train)

            # Predecir en el conjunto de validación
            y_pred = lr_model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Log accuracy metric
            mlflow.log_metric("accuracy", accuracy)

            return {'loss': -accuracy, 'status': STATUS_OK}

    # Espacio de búsqueda de hiperparámetros para Logistic Regression
    search_space_lr = {
        'C': hp.loguniform('C', np.log(0.01), np.log(10)),
        'max_iter': hp.quniform('max_iter', 100, 500, 1),
        'solver': hp.choice('solver', ['liblinear', 'lbfgs'])
    }

    # Ejecución de la optimización de hiperparámetros
    with mlflow.start_run(run_name="Optimización de Hiperparámetros Logistic Regression", nested=True):
        best_params_lr = fmin(
            fn=objective_lr,
            space=search_space_lr,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )

        # Ajustar los tipos de parámetros según Logistic Regression
        best_params_lr["max_iter"] = int(best_params_lr["max_iter"])
        best_params_lr["solver"] = ['liblinear', 'lbfgs'][best_params_lr['solver']]

        # Loggear los mejores parámetros en mlflow
        mlflow.log_params(best_params_lr)

    return best_params_lr

# Función para entrenar el mejor modelo

@task(name="Train Best Model")
def train_best_model(X_train, X_test, y_train, y_test, best_params_rf, best_params_lr, model_type: str) -> None:
    if model_type == "random_forest":
        with mlflow.start_run(run_name="Mejor modelo Random Forest"):
            # Registrar los mejores parámetros en mlflow si existen
            if best_params_rf is not None:
                mlflow.log_params(best_params_rf)

            # Crear y entrenar el modelo RandomForest con los mejores parámetros
            rf_model = RandomForestClassifier(
                n_estimators=int(best_params_rf['n_estimators']),
                max_depth=int(best_params_rf['max_depth']),
                min_samples_split=int(best_params_rf['min_samples_split']),
                min_samples_leaf=int(best_params_rf['min_samples_leaf']),
                random_state=42
            )

            rf_model.fit(X_train, y_train)

            # Realizar predicciones en el conjunto de validación
            y_pred = rf_model.predict(X_test)

            # Calcular la métrica de precisión
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            # Guardar el modelo RandomForest entrenado
            with open("models/tfidf_vectorizer.pkl", "wb") as f_model:
                pickle.dump(rf_model, f_model)

    elif model_type == "logistic_regression":
        with mlflow.start_run(run_name="Mejor modelo Logistic Regression"):
            # Registrar los mejores parámetros en mlflow si existen
            if best_params_lr is not None:
                mlflow.log_params(best_params_lr)

            # Crear y entrenar el modelo Logistic Regression con los mejores parámetros
            lr_model = LogisticRegression(
                C=best_params_lr['C'],
                max_iter=int(best_params_lr['max_iter']),
                solver=best_params_lr['solver'],
                random_state=42
            )

            lr_model.fit(X_train, y_train)

            # Realizar predicciones en el conjunto de validación
            y_pred = lr_model.predict(X_test)

            # Calcular la métrica de precisión
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            # Guardar el modelo Logistic Regression entrenado
            with open("models/logistic_regression_model.pkl", "wb") as f_model:
                pickle.dump(lr_model, f_model)
    else:
        raise ValueError("El modelo no es el adecuado. Use mejor 'random_forest' o 'logistic_regression'.")

    return None

@task(name='Register Models')
def register_models() -> None:
    """
    Aquí se registra el modelo "Champion" y "Challenger" para mis modelos de Random-Forest y Logistic-Regression.
    """
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Buscar las mejores ejecuciones para cada métrica ósea el accuracy
    rf_runs = mlflow.search_runs(filter_string="tags.model_family = 'random_forest'", order_by=['metrics.accuracy DESC'])
    lr_runs = mlflow.search_runs(filter_string="tags.model_family = 'logistic_regression'", order_by=['metrics.accuracy DESC'])

    # Registrar el modelo Champion para Random Forest (mayor Accuracy)
    if not rf_runs.empty:
        best_rf_run_id = rf_runs.loc[0]['run_id']
        rf_run_uri = f"runs:/{best_rf_run_id}/model"
        rf_result = mlflow.register_model(model_uri=rf_run_uri, name="Tinoco-RandomForest-prefect")
        client.set_registered_model_alias(name="Tinoco-RandomForest-prefect", alias="champion", version=rf_result.version)

    # Registrar el modelo Challenger para Random Forest (segundo mejor Accuracy)
    if len(rf_runs) > 1:
        challenger_rf_run_id = rf_runs.loc[1]['run_id']
        challenger_rf_run_uri = f"runs:/{challenger_rf_run_id}/model"
        challenger_rf_result = mlflow.register_model(model_uri=challenger_rf_run_uri, name="Tinoco-RandomForest-prefect")
        client.set_registered_model_alias(name="Tinoco-RandomForest-prefect", alias="challenger", version=challenger_rf_result.version)

    # Registrar el modelo Champion para Logistic Regression (mayor Accuracy)
    if not lr_runs.empty:
        best_lr_run_id = lr_runs.loc[0]['run_id']
        lr_run_uri = f"runs:/{best_lr_run_id}/model"
        lr_result = mlflow.register_model(model_uri=lr_run_uri, name="Tinoco-modelo-prefect")
        client.set_registered_model_alias(name="Tinoco-modelo-prefect", alias="champion", version=lr_result.version)

    # Registrar el modelo Challenger para Logistic Regression (segundo mejor Accuracy)
    if len(lr_runs) > 1:
        challenger_lr_run_id = lr_runs.loc[1]['run_id']
        challenger_lr_run_uri = f"runs:/{challenger_lr_run_id}/model"
        challenger_lr_result = mlflow.register_model(model_uri=challenger_lr_run_uri, name="Tinoco-modelo-prefect")
        client.set_registered_model_alias(name="Tinoco-modelo-prefect", alias="challenger", version=challenger_lr_result.version)

    return None


# Definir el flujo principal con Prefect
@flow(name="Main Flow")
def main_flow(model_type: str = "random_forest") -> None:
    """Pipeline de entrenamiento principal para los modelos de Random Forest y Logistic Regression"""

    # Inicializar MLflow en DagsHub
    DAGSHUB_URL = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn"
    dagshub.init(url=DAGSHUB_URL, mlflow=True)
    mlflow.set_experiment(experiment_name=f"Tinoco-{model_type}-model-experiment-prefect")

    # Paso 1: Lectura de datos
    CSV_PATH = "../data/tickets_classification_eng.json"
    df = read_data(CSV_PATH)

    # Paso 2: Agregar características
    X_train, X_test, y_train, y_test = train_test(df)


    # Paso 3: Ajuste de hiperparámetros
    if model_type == "random_forest":
        best_params_rf = hyper_parameter_tuning_random_forest(X_train, X_test, y_train, y_test)
        best_params_lr = None
    elif model_type == "logistic_regression":
        best_params_lr = hyper_parameter_tuning_logistic_regression(X_train, X_test, y_train, y_test)
        best_params_rf = None
    else:
        raise ValueError("El tipo de modelo especificado no es soportado. Use 'random_forest' o 'logistic_regression'.")

    # Paso 4: Entrenar el mejor modelo
    train_best_model(X_train, X_test, y_train, y_test, best_params_rf, best_params_lr, model_type)

    # Paso 5: Registrar modelos Champion y Challenger
    register_models()

# Ejecutar el flujo principal
if __name__ == "__main__":
    main_flow()

