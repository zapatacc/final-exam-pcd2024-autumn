import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from prefect import task, flow
import dagshub
import mlflow
from mlflow.tracking import MlflowClient

# Descargar stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


# Función para preprocesamiento
def preprocess_text(text):
    text = re.sub(r"XX|xx", "", text)  # Eliminar "XX"
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Mantener solo letras
    text = text.lower()  # Convertir a minúsculas
    tokens = text.split()  # Tokenizar
    tokens = [word for word in tokens if word not in stop_words]  # Eliminar stopwords
    return " ".join(tokens)


@task(name="read_data")
def read_csv():
    df = pd.read_csv("/home/melanie/PycharmProjects/final-exam-pcd2024-autumn/raw_data/data.csv")
    return df


@task(name="preprocess")
def apply_preprocess(df):
    # Aplicar preprocesamiento
    df['complaint_what_happened'] = df['complaint_what_happened'].apply(preprocess_text)
    df['ticket_classification'] = df['ticket_classification'].apply(preprocess_text)
    X = df['complaint_what_happened']
    y = df['ticket_classification']
    return X, y


@task(name="Vectorization and Splitting")
def vectorizer(X, y):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, tfidf, label_encoder


@task(name="Run Models")
def run_models(X_train, X_test, y_train, y_test):
    models_gridSearch = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
        },
        "SVC": {
            "model": SVC(probability=True, random_state=42),
            "params": {"C": [1, 10], "kernel": ["linear", "rbf"]}
        }
    }

    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)
    mlflow.set_experiment("melanie-michel")  # Cambié el nombre del experimento
    results = {}

    for model_name, model_info in models_gridSearch.items():
        with mlflow.start_run(run_name=model_name):
            grid = GridSearchCV(model_info["model"], model_info["params"], scoring="accuracy", cv=3)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(best_model, artifact_path=f"{model_name.lower()}_model")

            results[model_name] = {"accuracy": accuracy, "run_id": mlflow.active_run().info.run_id}

    return results


@task(name="Champion/Challenger")
def best_model_selection(experiment_name: str, model_name: str = "best-text-model"):
    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)

    # Obtener todas las ejecuciones de MLflow para el experimento dado
    all_runs = mlflow.search_runs(experiment_names=[experiment_name])
    best_accuracy = -float("inf")
    second_best_accuracy = -float("inf")
    best_run_id = None
    second_best_run_id = None

    # Seleccionar el modelo campeón y el modelo challenger
    for _, run in all_runs.iterrows():
        accuracy = run["metrics.accuracy"]
        run_id = run["run_id"]
        if accuracy > best_accuracy:
            second_best_accuracy = best_accuracy
            second_best_run_id = best_run_id
            best_accuracy = accuracy
            best_run_id = run_id
        elif accuracy > second_best_accuracy:
            second_best_accuracy = accuracy
            second_best_run_id = run_id

    client = MlflowClient()

    # Registrar el modelo campeón
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        best_model_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version
        client.set_registered_model_alias(name=model_name, alias="champion", version=best_model_version)

    # Registrar el modelo challenger
    if second_best_run_id:
        model_uri = f"runs:/{second_best_run_id}/model"
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        second_best_model_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version
        client.set_registered_model_alias(name=model_name, alias="challenger", version=second_best_model_version)

    return True


@flow
def main_flow():
    df = read_csv()
    X, y = apply_preprocess(df)
    X_train, X_test, y_train, y_test, _, _ = vectorizer(X, y)
    results = run_models(X_train, X_test, y_train, y_test)
    best_model_selection("melanie-michel", "final-model")  # Cambié el nombre del experimento aquí


if __name__ == "__main__":
    main_flow()

