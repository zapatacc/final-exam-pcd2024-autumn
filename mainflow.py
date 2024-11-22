from prefect import task, flow
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import dagshub
from nltk.corpus import stopwords
from mlflow.tracking import MlflowClient

# Initialize DagsHub and MLflow
@task(name="Initialize DagsHub and MLflow")
def init_dagshub():
    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)
    mlflow.set_experiment("renata-orozco-experiments")

# Load data
@task(name="Load Data")
def load_data():
    df = pd.read_csv(r"../Renata/datos/cleaned.csv")
    return df

# Split data into train and test sets
@task(name="Split Data")
def split_data(df):
    X = df['complaint_what_happened']
    y = df['ticket_classification']
    text_train, text_test, label_train, label_test = train_test_split(X, y, test_size=0.30, random_state=7)
    return text_train, text_test, label_train, label_test

# Encode labels
@task(name="Encode Labels")
def encode_labels(label_train, label_test):
    encoded_labels_train = pd.factorize(label_train)[0]
    encoded_labels_test = pd.factorize(label_test)[0]
    return encoded_labels_train, encoded_labels_test

# Create and train the model
@task(name="Train Model")
def train_model(text_train, encoded_labels_train):
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    logreg_pipeline = Pipeline([
        ("vectorizer", tfidf_vectorizer),
        ("logreg", LogisticRegression(max_iter=500))
    ])

    param_grid = {
        'logreg__C': [0.5, 0.15, 0.8],
        'logreg__penalty': ['l2'],
        'logreg__solver': ['lbfgs'],
    }

    grid_search = GridSearchCV(logreg_pipeline, param_grid, scoring='accuracy', cv=5, n_jobs=1, verbose=1)
    grid_search.fit(text_train, encoded_labels_train)
    
    return grid_search

# Evaluate the model
@task(name="Evaluate Model")
def evaluate_model(grid_search, text_test, encoded_labels_test):
    best_logreg_model = grid_search.best_estimator_
    predictions = best_logreg_model.predict(text_test)
    
    accuracy = accuracy_score(encoded_labels_test, predictions)
    report = classification_report(encoded_labels_test, predictions, output_dict=True)
    
    return accuracy, report, best_logreg_model

# Log metrics and model
@task(name="Log Metrics and Model")
def log_metrics_and_model(grid_search, accuracy, report, best_logreg_model):
    with mlflow.start_run(run_name="Logreg Pipeline"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        mlflow.sklearn.log_model(best_logreg_model, artifact_path="logreg_pipeline_model")

# Save and log label mapping
@task(name="Save and Log Label Mapping")
def save_label_mapping(label_train):
    label_mapping = dict(enumerate(pd.unique(label_train)))
    with open("label_mapping.pkl", "wb") as f:
        pickle.dump(label_mapping, f)
    mlflow.log_artifact("label_mapping.pkl")

# Select the best model and set aliases
@task(name="Select Best Model")
def select_best_model():
    """
    Selects the best runs based on precision from MLflow, registers their versions, 
    and assigns aliases "champion" and "challenger".
    """
    # Initialize DagsHub and MLflow
    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)

    # Retrieve all runs from the experiment
    experiment_name = "renata-orozco-experiments" 
    all_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.precision DESC"]  # Order by precision in descending order
    )

    # Ensure at least two runs exist
    if all_runs.shape[0] < 2:
        raise ValueError("Not enough runs available to select champion and challenger.")

    top_runs = all_runs.head(2).reset_index()  # Get the top 2 runs

    # Initialize MLflow client
    client = MlflowClient()

    # Assign aliases for the top 2 runs
    aliases = {
        "champion": top_runs.loc[0, "run_id"],  # Best model
        "challenger": top_runs.loc[1, "run_id"]  # Second best model
    }

    # Name of the registered model
    model_name = "renata-model"

    # Register aliases in MLflow
    for alias, run_id in aliases.items():
        model_uri = f"runs:/{run_id}/model"
        
        # Register the model if not already registered
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Get the latest version of the registered model
        latest_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version

        # Set alias for the registered model version
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=latest_version
        )

# Define the flow
@flow(name="mainflow")
def mainFlow():
    init_dagshub_task = init_dagshub()
    df = load_data()
    text_train, text_test, label_train, label_test = split_data(df)
    encoded_labels_train, encoded_labels_test = encode_labels(label_train, label_test)
    grid_search = train_model(text_train, encoded_labels_train)
    accuracy, report, best_logreg_model = evaluate_model(grid_search, text_test, encoded_labels_test)
    log_metrics_and_model(grid_search, accuracy, report, best_logreg_model)
    save_label_mapping(label_train)
    select_best_model()


if _name_ == "_main_":
    mainFlow()