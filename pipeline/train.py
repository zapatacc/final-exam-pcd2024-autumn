import prefect 
from prefect import flow, task
import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline


def removeX(text:str)-> str: 
    return text.replace("X","")


@task(name="readData")
def readData(path:str)-> json:
    with open(path, "r") as f:
        data = json.load(f)

    return data

@task(name="normalize")
def normalize(Json:json) -> pd.DataFrame:
    return pd.json_normalize(Json)


@task(name="preprocess")
def preprocessData(df:pd.DataFrame) -> None:
    df = df[["_source.complaint_what_happened","_source.product","_source.sub_product"]]

    rename = {"_source.complaint_what_happened":"complaint_what_happened",
    "_source.product":"category",
    "_source.sub_product":"sub_product"}

    df = df.rename(rename, axis=1)

    df["ticket_classification"] = df["category"] + " + " + df["sub_product"]
    df = df.drop(columns=["category","sub_product"],axis=1)
    df["complaint_what_happened"] = df["complaint_what_happened"].replace("",pd.NA)
    df = df.dropna()


    df.to_csv("../data/preprocessed_data/preprocessed.csv",index=False)

    return None

@task(name="clean")
def cleanData() -> pd.DataFrame:
    df = pd.read_csv("../data/preprocessed_data/preprocessed.csv")
    df["complaint_what_happened"] = df["complaint_what_happened"].apply(removeX)
    class_counts = df['ticket_classification'].value_counts()
    valid_classes = class_counts[class_counts >= 100].index
    df = df[df['ticket_classification'].isin(valid_classes)]
    df.to_csv("../data/clean_data/cleaned.csv",index=False)

    return df


@task(name="parametersProcess")
def processParameters(df:pd.DataFrame):
    # Define X e y
    X = df['complaint_what_happened']
    y = df['ticket_classification']

    # Codifica las clases de `y` como valores enteros
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.4, random_state=17, stratify=y_encoded
    )

    train_classes = set(y_train)
    valid_indices = [i for i, label in enumerate(y_test) if label in train_classes]

    if len(valid_indices) < len(y_test):
        print(f"Filtrando {len(y_test) - len(valid_indices)} instancias de prueba con clases desconocidas.")
    X_test = X_test.iloc[valid_indices]
    y_test = y_test[valid_indices]

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    return X,y,y_train, y_test, X_train_tfidf,X_test_tfidf

import pickle


@task(name="training")
def trainingModel(df: pd.DataFrame) -> None:
    # Define X e y
    X = df['complaint_what_happened']
    y = df['ticket_classification']

    # Codifica las clases de `y`
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.4, random_state=17, stratify=y_encoded
    )

    # Configura el tracking de MLflow local
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("patricio-villanueva-experiments")

    # Define el pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),  # Paso 1: Transformación con TF-IDF
        ("model", LogisticRegression(max_iter=1000))  # Paso 2: Modelo
    ])

    # Define parámetros para GridSearch
    param_grid = {
        "model__C": [0.1, 1, 10],  # Hiperparámetros del modelo
        "model__penalty": ["l2"]
    }

    # Entrena con GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=3, n_jobs=-1)

    with mlflow.start_run(run_name="Logistic Regression Pipeline"):
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Calcula métricas
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Loggea parámetros y métricas del mejor modelo
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        # Loggea el pipeline completo
        mlflow.sklearn.log_model(best_model, artifact_path="pipeline_model")

        # Guarda el LabelEncoder
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
        mlflow.log_artifact("label_encoder.pkl")


@task(name="selectBestModel")
def bestmodel():
    mlflow.set_tracking_uri("file:///tmp/mlruns") 

    # Nombre del experimento
    experiment_name = "patricio-villanueva-experiments"
    client = MlflowClient() 

    # Id del experimento
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No se encontró el experimento con nombre: {experiment_name}")
    experiment_id = experiment.experiment_id

    # Todos los runs (bueno los ultimos 1000)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=1000
    )

    # Encuentra el run con la mejor accuracy se que hay un order by pero neta nunca me sale asi que a la antigua
    best_run = None
    best_accuracy = -float("inf")

    for run in runs:
        metrics = run.data.metrics
        if "accuracy" in metrics and metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_run = run

    if best_run is None:
        raise ValueError("No se encontraron runs con la métrica 'accuracy'.")

    # Log del mejor run
    print(f"El mejor run es: {best_run.info.run_id} con accuracy: {best_accuracy}")

    # Registra el modelo del mejor run
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "patricio-model"

    # Registra el modelo en MLflow Model Registry
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Agrega un alias "champion" al modelo registrado
    model_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version
    )

    # Transiciona el modelo a la etapa "Production"
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production",
        archive_existing_versions=True  
    )

    print(f"El modelo ha sido registrado como '{model_name}' con alias 'champion' y etapa 'Production'.")


@flow(name="mainFlow")
def mainFlow(path:str)->None:
    # data almacena un tipo de dato json que contiene la informacion actualizada del rawData
    data = readData(path)
    # df almacena un tipo de dato DataFrame que contiene el json normalizado
    df = normalize(data)
    # preprocessData es una funcion sin tipo de dato de retorno que actualiza el csv contenido en data/proprocessed_data/ con los datos limpios y listos para procesarse
    preprocessData(df)
    # cleanData no necesita parametros ya que esta pensado para funcionar remplazando el preprocessed.csv en cada ejecucion de este training pipeline por lo que el path es fijo
    cleanedData = cleanData()
    # trainingModel recibe como parametros un DataFrame con el que se encarga de realizar el preprocesado interno de las variables para predecir 
    trainingModel(cleanedData)

    # Best model es una funcion independiente que se encarga de actualizar el modelo champion
    bestmodel()





mainFlow("../data/raw_data/tickets_classification_eng.json")
