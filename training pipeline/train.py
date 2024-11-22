import pandas as pd
import numpy as np
import pickle
import pathlib
import nltk
import re
import dagshub
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import dagshub
import mlflow.sklearn
from hyperopt.pyll import scope
from sklearn.linear_model import LogisticRegression
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import flow, task
import mlflow
import json

@task(name="Read Data", retries=4, retry_delay_seconds=[1, 4, 8, 16])
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    with open(file_path, "r") as file:
        datos = json.load(file)
    # Pass it as pandas dataframe
    df = pd.json_normalize(datos)

    # Selección de columnas
    cols = ['_source.complaint_what_happened', '_source.product', '_source.sub_product']

    for col in df.columns:
        if col not in cols:
            df.drop(col, axis=1, inplace=True)

    # Renombrar las columnas
    df.rename(columns={"_source.complaint_what_happened": "complaint_what_happened", "_source.product": "category",
                       "_source.sub_product": "sub_product"}, inplace=True)

    # Creación de nueva columna
    df['ticket_classification'] = df['category'] + '+' + df['sub_product']
    # Eliminar columnas redundantes
    df.drop(columns=['category', 'sub_product'], inplace=True)
    # Reemplazar los valores vacíos con nan
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    # Eliminación de filas con datos faltantes
    df.dropna(inplace=True)
    # Reiniciar índice
    df.reset_index(inplace=True, drop=True)

    return df

@task(name="Process data")
def process_data(df):
    # Función para limpiar texto
    def remove_sensitive_info(text):
        if isinstance(text, str):  # Ensure the input is a string
            return re.sub(r'X{2,}', '', text)  # Remove XX, XXX, XXXX
        return text
    df['complaint_what_happened'] = df['complaint_what_happened'].apply(remove_sensitive_info)

    # Threshold para categorías
    category_counts = df['ticket_classification'].value_counts()
    # Filter categories that meet the threshold
    filters = category_counts[category_counts >= 100].index
    df = df[df['ticket_classification'].isin(filters)]

    # NLTK processing
    stop_words = set(stopwords.words('english'))
    def remove_stopwords(text):
        # Tokenizar el texto
        tokens = word_tokenize(text)

        # Filtrar palabras vacías y caracteres especiales
        filtered_text = [word for word in tokens if word.lower() not in stop_words and not re.match(r'[^\w\s]', word)]

        # Unir las palabras filtradas en una cadena
        return ' '.join(filtered_text)

    df['complaint_what_happened'] = df['complaint_what_happened'].apply(remove_stopwords)

    X = df.complaint_what_happened
    y = df.ticket_classification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

@task(name="SVM Hyperparameter Tuning")
def svm_hyperparameter_tuning(X_train, X_test, y_train, y_test):
    def objective(params):
        # Extract parameters from the search space
        alpha = params['alpha']
        max_iter = int(params['max_iter'])

        # Build and train the SVM pipeline
        sgd = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=alpha,
                random_state=42,
                max_iter=max_iter,
                tol=None,
                class_weight='balanced'))
        ])
        sgd.fit(X_train, y_train)

        # Make predictions and calculate the objective metric (negative accuracy for minimization)
        y_pred = sgd.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return -accuracy  # Return negative because fmin minimizes by default

    # Set up the search space for hyperparameters
    search_space = {
        'alpha': hp.loguniform('alpha', -5, 1),  # Alpha (learning rate) on a log scale
        'max_iter': scope.int(hp.quniform('max_iter', 5, 100, 5))  # Iterations
    }

    with mlflow.start_run(run_name="SVM Hyper-parameter Optimization", nested=True):
        # Optimize SVM parameters using hyperopt
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=30,  # Adjust for more evaluations
            trials=trials
        )

        # Convert parameters to usable types
        best_params['alpha'] = float(best_params['alpha'])
        best_params['max_iter'] = int(best_params['max_iter'])

        # Log the best parameters to MLflow
        mlflow.log_params(best_params)

    return best_params

@task(name="LogReg Hyperparameter Tuning")
def logistic_regression_hyperparameter_tuning(X_train, X_test, y_train, y_test):
    def objective(params):
        # Extract parameters from the search space
        C = params['C']
        max_iter = int(params['max_iter'])

        # Build and train the Logistic Regression pipeline
        logreg = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(
                n_jobs=1,
                C=C,
                max_iter=max_iter,
                class_weight='balanced',
                random_state=42))
        ])
        logreg.fit(X_train, y_train)

        # Make predictions and calculate the objective metric (e.g., negative accuracy for minimization)
        y_pred = logreg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return -accuracy  # Return negative because fmin minimizes by default

    # Set up the search space for hyperparameters
    search_space = {
        'C': hp.loguniform('C', -3, 3),  # Regularization strength on a log scale
        'max_iter': scope.int(hp.quniform('max_iter', 100, 1000, 100))  # Number of iterations
    }

    with mlflow.start_run(run_name="LogReg Hyper-parameter Optimization", nested=True):
        # Optimize Logistic Regression parameters using hyperopt
        trials = Trials()
        best_log_reg_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=30,  # Adjust for more evaluations
            trials=trials
        )

        # Convert parameters to usable types
        best_log_reg_params['C'] = float(best_log_reg_params['C'])
        best_log_reg_params['max_iter'] = int(best_log_reg_params['max_iter'])

        # Log the best parameters to MLflow
        mlflow.log_params(best_log_reg_params)

    return best_log_reg_params

@task(name = "Train Best SVM Model")
def train_best_svm_model(X_train, X_test, y_train, y_test, best_params) -> None:
    with mlflow.start_run(run_name="Best SVM Model"):
        # Set experiment tags
        mlflow.set_tags({
            "project": "Text Classification with SVM",
            "optimizer_engine": "hyper-opt",
            "model_family": "Linear SVM",
            "feature_set_version": 1,
        })

        # Train the SVM model using the best parameters
        sgd = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=best_params['alpha'],
                random_state=42,
                max_iter=best_params['max_iter'],
                tol=None,
                class_weight='balanced'))
        ])
        sgd.fit(X_train, y_train)

        # Make predictions and calculate metrics
        y_pred = sgd.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)

        # Save the trained SVM pipeline to MLflow
        mlflow.sklearn.log_model(sgd, artifact_path="model")

    return None

@task(name = "Train Best LogReg Model")
def train_best_logreg_model(X_train, X_test, y_train, y_test, best_log_reg_params) -> None:

    with mlflow.start_run(run_name="Best Logistic Regression Model"):
        # Set experiment tags
        mlflow.set_tags({
            "project": "Text Classification with Logistic Regression",
            "optimizer_engine": "hyper-opt",
            "model_family": "Logistic Regression",
            "feature_set_version": 1,
        })

        # Train the Logistic Regression model using the best parameters
        logreg = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(
                n_jobs=1,
                C=best_log_reg_params['C'],
                max_iter=best_log_reg_params['max_iter'],
                class_weight='balanced',
                random_state=42))
        ])
        logreg.fit(X_train, y_train)

        # Make predictions and calculate metrics
        y_pred = logreg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)

        # Save the trained Logistic Regression pipeline to MLflow
        mlflow.sklearn.log_model(logreg, artifact_path="model")

    return None

@task(name='Register Model')
def register_model():
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    df = mlflow.search_runs(order_by=['metrics.f1'])

    best_run_id = df.loc[df['metrics.f1'].idxmin()]['run_id']
    best_run_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(
        model_uri=best_run_uri,
        name="erick-model-perfect"
    )

    # Register the second-best model as the challenger
    second_best_run_id = df.iloc[df['metrics.f1'].nsmallest(2).index[-1]]['run_id']
    second_best_run_uri = f"runs:/{second_best_run_id}/model"
    mlflow.register_model(
        model_uri=second_best_run_uri,
        name="erick-model-perfect"
    )

    # Set aliases
    latest_versions = client.get_latest_versions("erick-model-perfect")

    # Assign "champion" to the best model
    for version in latest_versions:
        if version.run_id == best_run_id:
            client.set_registered_model_alias(
                name="erick-model-perfect",
                alias="champion",
                version=version.version
            )

    # Assign "challenger" to the second-best model
    for version in latest_versions:
        if version.run_id == second_best_run_id:
            client.set_registered_model_alias(
                name="erick-model-perfect",
                alias="challenger",
                version=version.version
            )


@flow(name="Main Flow")
def main_flow() -> None:
    """The main training pipeline"""

    data = f"../data/tickets_classification_eng.json"

    # MLflow settings
    dagshub.init(url="https://dagshub.com/zapatacc/final-exam-pcd2024-autumn", mlflow=True)

    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="erick-model-prefect")

    #Load
    df = read_data(data)

    # Process
    X_train, X_test, y_train, y_test = process_data(df)

    # SVM Hyperparameter tuning
    best_params = svm_hyperparameter_tuning(X_train, X_test, y_train, y_test)

    # LogReg Hyperparameter tuning
    best_log_reg_params = logistic_regression_hyperparameter_tuning(X_train, X_test, y_train, y_test)

    # Train best SVM model
    train_best_svm_model(X_train, X_test, y_train, y_test, best_params)

    # Train best logreg model
    train_best_logreg_model(X_train, X_test, y_train, y_test, best_log_reg_params)

    # Register model
    register_model()

main_flow()