import prefect
from prefect import flow, task
import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
import pickle
import dagshub

# funcion utilitaria para eliminar la letra "X" de un texto
def removeX(text: str) -> str:
    '''
    entrada de la funcion
    text: texto en el cual se busca eliminar la letra "X"
    
    que hace la funcion
    reemplaza las ocurrencias de "X" con una cadena vacia
    
    salida de la funcion
    texto sin la letra "X"
    '''
    return text.replace("X", "")

# tarea para leer un archivo json desde la ruta especificada
@task(name="readData")
def read_data(path: str) -> json:
    '''
    entrada de la funcion
    path: ruta del archivo json
    
    que hace la funcion
    lee un archivo json desde la ruta proporcionada y devuelve el contenido
    
    salida de la funcion
    contenido del archivo json como objeto python
    '''
    with open(path, "r") as f:
        data = json.load(f)
    return data

# tarea para normalizar datos json a un dataframe
@task(name="normalize")
def normalize_data(data: json) -> pd.DataFrame:
    '''
    entrada de la funcion
    data: datos en formato json
    
    que hace la funcion
    convierte los datos json en un dataframe de pandas
    
    salida de la funcion
    dataframe con los datos normalizados
    '''
    return pd.json_normalize(data)

# tarea para preprocesar datos, renombrar columnas y filtrar informacion relevante
@task(name="preprocess")
def preprocess_data(df: pd.DataFrame) -> None:
    '''
    entrada de la funcion
    df: dataframe con los datos normalizados
    
    que hace la funcion
    filtra columnas relevantes, renombra columnas, genera una nueva columna de clasificacion y guarda los datos preprocesados
    
    salida de la funcion
    no devuelve nada, pero guarda los datos preprocesados en un archivo csv
    '''
    df = df[["_source.complaint_what_happened", "_source.product", "_source.sub_product"]]

    rename = {
        "_source.complaint_what_happened": "complaint_what_happened",
        "_source.product": "category",
        "_source.sub_product": "sub_product",
    }

    df = df.rename(rename, axis=1)
    df["ticket_classification"] = df["category"] + " + " + df["sub_product"]
    df = df.drop(columns=["category", "sub_product"], axis=1)
    df["complaint_what_happened"] = df["complaint_what_happened"].replace("", pd.NA)
    df = df.dropna()
    df.to_csv("../data/preprocessed_data/preprocessed.csv", index=False)
    return None

# tarea para limpiar datos eliminando clases con menos de 100 registros
@task(name="clean")
def clean_data() -> pd.DataFrame:
    '''
    entrada de la funcion
    no requiere argumentos
    
    que hace la funcion
    carga los datos preprocesados, limpia el texto y filtra clases con al menos 100 registros
    
    salida de la funcion
    dataframe con los datos limpios
    '''
    df = pd.read_csv("../data/preprocessed_data/preprocessed.csv")
    df["complaint_what_happened"] = df["complaint_what_happened"].apply(removeX)
    class_counts = df['ticket_classification'].value_counts()
    valid_classes = class_counts[class_counts >= 100].index
    df = df[df['ticket_classification'].isin(valid_classes)]
    df.to_csv("../data/clean_data/cleaned.csv", index=False)
    return df

# tarea para entrenar un modelo con gridsearch y registrar los resultados en mlflow
@task(name="training")
def training_pipeline(model_name: str, model, param_grid: dict, df: pd.DataFrame):
    '''
    entrada de la funcion
    model_name: nombre del modelo
    model: modelo sklearn
    param_grid: diccionario con los parametros para gridsearch
    df: dataframe con los datos limpios
    
    que hace la funcion
    entrena un modelo con pipeline y gridsearch, calcula metricas y las registra en mlflow
    
    salida de la funcion
    no devuelve nada, pero registra el modelo y metricas en mlflow
    '''
    X = df['complaint_what_happened']
    y = df['ticket_classification']

    # codifica las etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # divide los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.4, random_state=17, stratify=y_encoded
    )

    # inicializa dagshub y mlflow
    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)
    mlflow.set_experiment("patricio-villanueva-experiments")

    # define el pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", model)
    ])

    # entrena el modelo con gridsearch
    grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=3, n_jobs=-1)

    with mlflow.start_run(run_name=f"{model_name} Pipeline"):
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # calcula metricas
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # registra parametros, metricas y el modelo
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])
        mlflow.sklearn.log_model(best_model, artifact_path="pipeline_model")

        # guarda y registra el labelencoder
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
        mlflow.log_artifact("label_encoder.pkl")

# tarea para seleccionar el mejor modelo basado en las metricas de precision
@task(name="selectBestModel")
def select_best_model():
    '''
    entrada de la funcion
    no requiere argumentos
    
    que hace la funcion
    busca las mejores corridas en mlflow, selecciona el mejor modelo y registra alias para las versiones champion y challenger
    
    salida de la funcion
    no devuelve nada, pero actualiza los alias en mlflow
    '''
    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)
    all_runs = mlflow.search_runs(
        experiment_names=["patricio-villanueva-experiments"],
        order_by=["metrics.accuracy DESC"],
    )

    bestsRun = all_runs.drop_duplicates(subset="metrics.accuracy").head(100).reset_index()

    client = MlflowClient()

    aliases = {
        "champion": bestsRun.run_id[0],
        "challenger": bestsRun.run_id[1]
    }

    model_name = "patricio-model"

    for alias, run_id in aliases.items():
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        model_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version

        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=model_version
        )

# flujo principal que coordina todas las tareas
@flow(name="mainFlow")
def main_flow(path: str):
    '''
    entrada de la funcion
    path: ruta del archivo json con los datos sin procesar
    
    que hace la funcion
    coordina la lectura, normalizacion, preprocesamiento, limpieza, entrenamiento y seleccion del mejor modelo
    
    salida de la funcion
    no devuelve nada, pero ejecuta todo el flujo de trabajo
    '''
    data = read_data(path)
    df = normalize_data(data)
    preprocess_data(df)
    cleaned_data = clean_data()

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=200), {
            "model__C": [0.1, 1, 10]
        }),
        ("Random Forest", RandomForestClassifier(), {
            "model__n_estimators": [10, 50, 100],
            "model__max_depth": [None, 10, 20]
        }),
        ("SVM", SVC(), {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        })
    ]

    for model_name, model, param_grid in models:
        training_pipeline(model_name, model, param_grid, cleaned_data)

    select_best_model()

# ejecuta el flujo principal
main_flow("../data/raw_data/tickets_classification_eng.json")
