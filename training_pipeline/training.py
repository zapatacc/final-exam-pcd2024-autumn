from prefect import task, flow
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from mlflow.tracking import MlflowClient
import dagshub
import mlflow
from sklearn.metrics import  root_mean_squared_error, accuracy_score,classification_report
import pickle
from sklearn.preprocessing import LabelEncoder


@task(name="readData")
def readData(path:str):
    df = pd.read_csv(path)
    return df

@task(name="prepareData")
def prepareData(df:DataFrame):
    X = df["complaint_what_happened"]
    y = df["ticket_classification"]

    return X,y 

@task(name="splitData")
def splitData(X,y):
    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    tfidf = TfidfVectorizer()
    X_vec = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec,y_encoded, test_size=0.2, random_state=24, stratify=y_encoded) 

    return X_train, X_test, y_train, y_test


@task(name="trainModels")
def trainModels(X_train, X_test, y_train, y_test):

    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("dafne-tamayo-experiments")


    label_encoder = LabelEncoder()


    models = {
        "Logistic Regression": (LogisticRegression, {"penalty": "l2", "C": 1.0, "solver": "lbfgs"}),
        "Support Vector Machine": (SVC, {"C": 1.0, "kernel": "rbf", "gamma": "scale"}),
        "K-Nearest Neighbors": (KNeighborsClassifier, {"n_neighbors": 5, "weights": "uniform"}),
        "K-Nearest Neighbors2": (KNeighborsClassifier, {"n_neighbors": 5, "weights": "distance"})
    }


    for model_name, (model, params) in models.items():
        with mlflow.start_run(run_name=f"dafne-tamayo-{model_name}"):
            instance = model(**params)
            
            instance.fit(X_train, y_train)
            y_pred = instance.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            with open ("label_encoder.pkl", 'wb') as file:
                pickle.dump(label_encoder, file)
            mlflow.log_artifact("label_encoder.pkl")
            
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(instance, artifact_path=f"best_model_{model_name}")


@task(name="selectModels")
def selectModels():

    client = MlflowClient()

    all_runs = mlflow.search_runs(
        experiment_names=["dafne-tamayo-experiments"],
        order_by=["metrics.accuracy DESC"],
    )

    bestsRun = all_runs.drop_duplicates(subset="metrics.accuracy").head(2).reset_index()

    model_uri = f"runs:/{bestsRun.run_id[1]}/model"
    model_name = "dafne-model"

    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Agrega un alias "champion" al modelo registrado
    model_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version
    client.set_registered_model_alias(
        name=model_name,
        alias="challenger",
        version=model_version
    )

    ###### 

    model_uri = f"runs:/{bestsRun.run_id[0]}/model"
    model_name = "dafne-model"

    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Agrega un alias "champion" al modelo registrado
    model_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version
    )

@flow(name="mainFlow")
def mainFlow(path:str):
    df = readData(path)
    X,y = prepareData(df)
    X_train, X_test, y_train, y_test = splitData(X,y)
    trainModels(X_train, X_test, y_train, y_test)
    selectModels()

# "../data/clean_data/datatransformed.csv"

mainFlow("../data/clean_data/datatransformed.csv")
