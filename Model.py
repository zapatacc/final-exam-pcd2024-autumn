from prefect import flow, task
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Configuración de MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Francisco-Gonzalez-logistic-randomforest")

# Cargar y preprocesar datos
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df['complaint_what_happened']
    y = df['ticket_classification']
    vectorizer = CountVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)
    return train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Definir tareas
@task
def train_logistic_regression(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="Francisco-Gonzalez-logistic-regression") as run:
        params = {'C': [0.1, 1, 10]}
        model = GridSearchCV(LogisticRegression(), params, cv=3)
        model.fit(X_train, y_train)
        
        best_model = model.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_param("best_C", model.best_params_['C'])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(best_model, "model")
        
        return run.info.run_id, accuracy


@task
def train_random_forest(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="Francisco-Gonzalez-random-forest") as run:
        params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
        model = GridSearchCV(RandomForestClassifier(), params, cv=3)
        model.fit(X_train, y_train)
        
        best_model = model.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_params(model.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(best_model, "model")
        
        return run.info.run_id, accuracy

@task
def register_models(logistic_run_id, rf_run_id, logistic_acc, rf_acc):
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Nombres de los modelos
    champion_model_name = "Francisco-Gonzalez-model-champion"
    challenger_model_name = "Francisco-Gonzalez-model-challenger"

    # Función para verificar si un modelo ya está registrado
    def model_exists(model_name):
        try:
            client.get_registered_model(model_name)
            return True
        except mlflow.exceptions.RestException:
            return False

    # Registrar modelos como Champion y Challenger
    if logistic_acc > rf_acc:
        # Registrar el modelo Champion
        if not model_exists(champion_model_name):
            client.create_registered_model(champion_model_name)
        client.create_model_version(
            name=champion_model_name,
            source=f"runs:/{logistic_run_id}/model",
            run_id=logistic_run_id,
        )

        # **Guardar el modelo Champion**
        best_model = client.download_artifacts(run_id=logistic_run_id, path="model")
        import joblib
        joblib.dump(best_model, "champion_model.pkl")

        # Registrar el modelo Challenger
        if not model_exists(challenger_model_name):
            client.create_registered_model(challenger_model_name)
        client.create_model_version(
            name=challenger_model_name,
            source=f"runs:/{rf_run_id}/model",
            run_id=rf_run_id,
        )
    else:
        # Registrar el modelo Champion
        if not model_exists(champion_model_name):
            client.create_registered_model(champion_model_name)
        client.create_model_version(
            name=champion_model_name,
            source=f"runs:/{rf_run_id}/model",
            run_id=rf_run_id,
        )

        # **Guardar el modelo Champion**
        best_model = client.download_artifacts(run_id=rf_run_id, path="model")
        import joblib
        joblib.dump(best_model, "champion_model.pkl")

        # Registrar el modelo Challenger
        if not model_exists(challenger_model_name):
            client.create_registered_model(challenger_model_name)
        client.create_model_version(
            name=challenger_model_name,
            source=f"runs:/{logistic_run_id}/model",
            run_id=logistic_run_id,
        )


@flow(name="Entrenamiento y Registro")
def main_flow(file_path):
    X_train, X_test, y_train, y_test = load_data(file_path)
    logistic_run_id, logistic_acc = train_logistic_regression(X_train, y_train, X_test, y_test)
    rf_run_id, rf_acc = train_random_forest(X_train, y_train, X_test, y_test)
    register_models(logistic_run_id, rf_run_id, logistic_acc, rf_acc)

# Ejecutar el flujo
if __name__ == "__main__":
    file_path = "Data/processed/cleaned_tickets.csv"
    main_flow(file_path)
