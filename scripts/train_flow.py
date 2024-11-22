# train_flow.py

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from prefect import flow, task, get_run_logger
import mlflow
import mlflow.sklearn
import logging
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)

# Configuración de MLflow
#mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ajusta la URI si es necesario
mlflow.set_experiment("carlos-moreno-experiment")

## Serie de Tasks

@task
def load_data():
    logger = get_run_logger()
    start_time = time.time()
    logger.info("Cargando datos y etiquetas...")
    
    # Cargar embeddings
    embeddings = np.load('data/processed_data/embeddings.npy')
    logger.info(f"Embeddings cargados con forma: {embeddings.shape}")
    
    # Cargar LabelEncoder
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Cargar dataset
    df = pd.read_csv('data/processed_data/cld_dataset.csv')
    
    # Asegurar que df tiene el mismo número de filas que embeddings
    if len(df) != embeddings.shape[0]:
        logger.warning("El número de filas en df no coincide con el número de embeddings. Ajustando df...")
        df = df.iloc[:embeddings.shape[0]].reset_index(drop=True)
    
    y = le.transform(df['ticket_classification'])
    logger.info(f"Etiquetas ajustadas con longitud: {len(y)}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Datos cargados en {elapsed_time:.2f} segundos.")
    return embeddings, y, le

@task
def adjust_classes(y, le, min_samples=20):
    logger = get_run_logger()
    class_counts = np.bincount(y)
    rare_classes = np.where(class_counts < min_samples)[0]
    logger.info(f"Clases raras (menos de {min_samples} muestras): {len(rare_classes)}")

    # Crear un mapa de índice de etiqueta a nombre de clase
    label_to_class = {i: class_name for i, class_name in enumerate(le.classes_)}

    y_adjusted_labels = []
    for label in y:
        if label in rare_classes:
            y_adjusted_labels.append('Rare')
        else:
            y_adjusted_labels.append(label_to_class[label])

    le_adjusted = LabelEncoder()
    y_adjusted = le_adjusted.fit_transform(y_adjusted_labels)
    logger.info(f"Nuevas clases después del ajuste: {le_adjusted.classes_}")

    # Crear un diccionario de mapeo de etiquetas
    adjusted_label_mapping = {label: le_adjusted.transform([label_to_class[label]])[0]
                              if label not in rare_classes else le_adjusted.transform(['Rare'])[0]
                              for label in np.unique(y)}

    return y_adjusted, le_adjusted, adjusted_label_mapping


@task
def split_data(X, y, test_size=0.2, random_state=42):
    logger = get_run_logger()
    logger.info("Dividiendo los datos en entrenamiento y prueba...")
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    logger.info(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]}")
    logger.info(f"Tamaño del conjunto de prueba: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

@task
def verify_labels(y_train, y_test, le_adjusted):
    logger = get_run_logger()
    for name, y in [("y_train", y_train), ("y_test", y_test)]:
        unique_labels = np.unique(y)
        class_names = le_adjusted.inverse_transform(unique_labels)
        logger.info(f"Etiquetas únicas en {name}: {class_names}")


@task
def scale_features(X_train, X_test):
    logger = get_run_logger()
    logger.info("Escalando las características...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

@task
def train_mlp(X_train, y_train):
    logger = get_run_logger()
    start_time = time.time()
    logger.info("Entrenando el modelo MLP...")

    param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [300, 500],
        'early_stopping': [True]
    }
    mlp = MLPClassifier(random_state=42)
    grid = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Modelo MLP entrenado en {elapsed_time:.2f} segundos.")
    logger.info(f"Mejores hiperparámetros para MLP: {best_params}")
    return best_model, best_params


@task
def train_random_forest(X_train, y_train):
    logger = get_run_logger()
    start_time = time.time()
    logger.info("Entrenando el modelo Random Forest...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Modelo Random Forest entrenado en {elapsed_time:.2f} segundos.")
    logger.info(f"Mejores hiperparámetros para Random Forest: {best_params}")
    return best_model, best_params


@task
def reduce_dimensionality(X_train, X_test, n_components=100):
    logger = get_run_logger()
    logger.info(f"Reduciendo dimensionalidad a {n_components} componentes...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    logger.info(f"Dimensionalidad reducida: {X_train_pca.shape[1]} componentes.")
    return X_train_pca, X_test_pca

@task
def train_svm(X_train, y_train):
    logger = get_run_logger()
    start_time = time.time()
    logger.info("Entrenando el modelo SVM...")

    param_grid = {
        'C': [1, 10],
        'kernel': ['linear'],  # El kernel lineal es más rápido
    }
    svm = SVC(probability=True, random_state=42, class_weight='balanced')
    grid = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Modelo SVM entrenado en {elapsed_time:.2f} segundos.")
    logger.info(f"Mejores hiperparámetros para SVM: {best_params}")
    return best_model, best_params


@task
def evaluate_model(model, X_test, y_test, le, model_name):
    logger = get_run_logger()
    logger.info(f"Evaluando el modelo {model_name}...")
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generar el reporte de clasificación
    report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        zero_division=0
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Modelo {model_name} evaluado en {elapsed_time:.2f} segundos.")
    logger.info(f"Accuracy de {model_name}: {accuracy:.4f}")
    logger.info(f"Reporte de clasificación para {model_name}:\n{report}")
    return accuracy, report

## Main Flow

@flow(name="main-flow")
def main_flow():
    logger = get_run_logger()
    total_start_time = time.time()
    logger.info("Iniciando el flujo principal de entrenamiento...")
    
    embeddings, y, le = load_data()
    y, le_adjusted, adjusted_label_mapping = adjust_classes(y, le, min_samples=20)
    X_train, X_test, y_train, y_test = split_data(embeddings, y)
    verify_labels(y_train, y_test, le_adjusted)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Guardar el LabelEncoder ajustado
    with open('models/label_encoder_adjusted.pkl', 'wb') as f:
        pickle.dump(le_adjusted, f)
    
    # Entrenar MLP
    mlp_model, mlp_params = train_mlp(X_train_scaled, y_train)
    mlp_accuracy, mlp_report = evaluate_model(mlp_model, X_test_scaled, y_test, le_adjusted, "MLP")
    
    # Registrar MLP con MLflow
    with mlflow.start_run(run_name='MLP'):
        mlflow.log_params(mlp_params)
        mlflow.log_metric('accuracy', mlp_accuracy)
        mlflow.sklearn.log_model(mlp_model, 'model')
        mlflow.log_artifact('models/label_encoder.pkl')
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "carlos-moreno-mlp"
        )
    
    # Entrenar Random Forest
    rf_model, rf_params = train_random_forest(X_train, y_train)
    rf_accuracy, rf_report = evaluate_model(rf_model, X_test, y_test, le_adjusted, "Random Forest")
    
    # Registrar Random Forest con MLflow
    with mlflow.start_run(run_name='RandomForest'):
        mlflow.log_params(rf_params)
        mlflow.log_metric('accuracy', rf_accuracy)
        mlflow.sklearn.log_model(rf_model, 'model')
        mlflow.log_artifact('models/label_encoder.pkl')
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "carlos-moreno-random-forest"
        )
    
    # Reducir dimensionalidad
    X_train_pca, X_test_pca = reduce_dimensionality(X_train_scaled, X_test_scaled)

    # Entrenar SVM con los datos reducidos
    svm_model, svm_params = train_svm(X_train_pca, y_train)
    svm_accuracy, svm_report = evaluate_model(svm_model, X_test_pca, y_test, le_adjusted, "SVM")

    
    # Registrar SVM con MLflow
    with mlflow.start_run(run_name='SVM'):
        mlflow.log_params(svm_params)
        mlflow.log_metric('accuracy', svm_accuracy)
        mlflow.sklearn.log_model(svm_model, 'model')
        mlflow.log_artifact('models/label_encoder.pkl')
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "carlos-moreno-svm"
        )
    
    # Comparar los modelos y seleccionar Champion y Challenger
    accuracies = {
        'MLP': mlp_accuracy,
        'Random Forest': rf_accuracy,
        'SVM': svm_accuracy
    }
    
    sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    champion_model_name = sorted_models[0][0]
    challenger_model_name = sorted_models[1][0]
    
    logger.info(f"Modelo Champion: {champion_model_name} con accuracy: {sorted_models[0][1]:.4f}")
    logger.info(f"Modelo Challenger: {challenger_model_name} con accuracy: {sorted_models[1][1]:.4f}")
    
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    total_elapsed_minutes = total_elapsed_time // 60
    total_elapsed_seconds = total_elapsed_time % 60
    logger.info(f"Flujo de entrenamiento completado en {int(total_elapsed_minutes)} minutos y {total_elapsed_seconds:.2f} segundos.")

if __name__ == '__main__':
    main_flow()
