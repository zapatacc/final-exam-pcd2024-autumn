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

# Utility function to remove "X" from text
def removeX(text: str) -> str:
    return text.replace("X", "")


@task(name="readData")
def read_data(path: str) -> json:
    with open(path, "r") as f:
        data = json.load(f)
    return data


@task(name="normalize")
def normalize_data(data: json) -> pd.DataFrame:
    return pd.json_normalize(data)


@task(name="preprocess")
def preprocess_data(df: pd.DataFrame) -> None:
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


@task(name="clean")
def clean_data() -> pd.DataFrame:
    df = pd.read_csv("../data/preprocessed_data/preprocessed.csv")
    df["complaint_what_happened"] = df["complaint_what_happened"].apply(removeX)
    class_counts = df['ticket_classification'].value_counts()
    valid_classes = class_counts[class_counts >= 100].index
    df = df[df['ticket_classification'].isin(valid_classes)]
    df.to_csv("../data/clean_data/cleaned.csv", index=False)
    return df


@task(name="training")
def training_pipeline(model_name: str, model, param_grid: dict, df: pd.DataFrame):
    X = df['complaint_what_happened']
    y = df['ticket_classification']

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.4, random_state=17, stratify=y_encoded
    )

    # Initialize DAGsHub and MLflow
    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)
    mlflow.set_experiment("patricio-villanueva-experiments")

    # Define pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", model)
    ])

    # Train model with GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=3, n_jobs=-1)

    with mlflow.start_run(run_name=f"{model_name} Pipeline"):
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Log parameters, metrics, and model
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])
        mlflow.sklearn.log_model(best_model, artifact_path="pipeline_model")

        # Save and log the LabelEncoder
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
        mlflow.log_artifact("label_encoder.pkl")


@task(name="selectBestModel")
def select_best_model():
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


@flow(name="mainFlow")
def main_flow(path: str):
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


main_flow("../data/raw_data/tickets_classification_eng.json")
