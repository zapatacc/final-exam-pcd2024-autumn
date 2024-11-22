from prefect import flow, task
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
from dagshub import init
import pickle

# Inicializar dagshub y configurar mlflow
init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)

# Variables globales
best_rf_model = None
best_rf_accuracy = 0
best_lr_model = None
best_lr_accuracy = 0
vectorizer = None

# Cargar datos
def load_data(file_path):
    global vectorizer
    df = pd.read_csv(file_path)
    X = df['complaint_what_happened']
    y = df['ticket_classification']
    vectorizer = CountVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)
    return train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Función objetivo para Random Forest
def objective_rf(params):
    global best_rf_model, best_rf_accuracy
    rf_model = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_rf_accuracy:
        best_rf_accuracy = accuracy
        best_rf_model = rf_model

    return {'loss': -accuracy, 'status': STATUS_OK}

# Espacio de búsqueda para Random Forest
search_space_rf = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
}

# Función objetivo para Logistic Regression
def objective_lr(params):
    global best_lr_model, best_lr_accuracy
    lr_model = LogisticRegression(
        C=params['C'], 
        solver='liblinear', 
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_lr_accuracy:
        best_lr_accuracy = accuracy
        best_lr_model = lr_model

    return {'loss': -accuracy, 'status': STATUS_OK}

# Espacio de búsqueda para Logistic Regression
search_space_lr = {
    'C': hp.loguniform('C', -4, 2)
}

@task
def register_models():
    client = MlflowClient()

    # Comparar resultados para determinar Champion y Challenger
    if best_rf_accuracy > best_lr_accuracy:
        champion_model, challenger_model = best_rf_model, best_lr_model
        champion_accuracy, challenger_accuracy = best_rf_accuracy, best_lr_accuracy
        champion_name = "Francisco-RandomForest"
        challenger_name = "Francisco-LogisticRegression"
    else:
        champion_model, challenger_model = best_lr_model, best_rf_model
        champion_accuracy, challenger_accuracy = best_lr_accuracy, best_rf_accuracy
        champion_name = "Francisco-LogisticRegression"
        challenger_name = "Francisco-RandomForest"

    # Registrar Champion
    with mlflow.start_run(run_name="Francisco-Model"):
        mlflow.log_metric("accuracy", champion_accuracy)
        mlflow.sklearn.log_model(champion_model, "model")
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact("vectorizer.pkl", artifact_path="preprocessing")
        registered_model_champion = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            champion_name
        )
        print(f"Champion model registered: {registered_model_champion.name}")

    # Registrar Challenger
    with mlflow.start_run(run_name="Francisco-Model"):
        mlflow.log_metric("accuracy", challenger_accuracy)
        mlflow.sklearn.log_model(challenger_model, "model")
        registered_model_challenger = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            challenger_name
        )
        print(f"Challenger model registered: {registered_model_challenger.name}")

    print(f"Champion Accuracy: {champion_accuracy}")
    print(f"Challenger Accuracy: {challenger_accuracy}")

@flow(name="Entrenamiento y Optimización")
def main_flow(file_path):
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = load_data(file_path)

    # Tracking en MLflow
    mlflow.set_experiment("Francisco-RandomForest")
    with mlflow.start_run(run_name="Random Forest Training"):
        fmin(fn=objective_rf, space=search_space_rf, algo=tpe.suggest, max_evals=10, trials=Trials())
        mlflow.log_metric("best_rf_accuracy", best_rf_accuracy)

    mlflow.set_experiment("Francisco-LogisticRegression")
    with mlflow.start_run(run_name="Logistic Regression Training"):
        fmin(fn=objective_lr, space=search_space_lr, algo=tpe.suggest, max_evals=10, trials=Trials())
        mlflow.log_metric("best_lr_accuracy", best_lr_accuracy)

    # Registrar modelos en el Model Registry
    register_models()

# Ejecutar flujo principal
if __name__ == "__main__":
    file_path = "Data/processed/cleaned_tickets.csv"
    main_flow(file_path)
