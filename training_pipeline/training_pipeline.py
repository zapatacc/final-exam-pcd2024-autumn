import pathlib
import pickle
from prefect import task, flow
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import dagshub

dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)

# Configurar el tracking URI de MLflow
MLFLOW_TRACKING_URI = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow"  
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Nombre del experimento
EXPERIMENT_NAME = "colome-prefect-experiment"
MODEL_REGISTRY_NAME = "colome-prefect-modelos"

try:
    client.get_registered_model(MODEL_REGISTRY_NAME)
    print(f"El registro de modelos '{MODEL_REGISTRY_NAME}' ya existe.")
except:
    client.create_registered_model(MODEL_REGISTRY_NAME)
    print(f"Registro de modelos '{MODEL_REGISTRY_NAME}' creado.")

mlflow.set_experiment(EXPERIMENT_NAME)

# cargar datos
@task
def load_json(filepath: str):
    with open(filepath, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)
    df = pd.json_normalize(datos)
    return df


# Seleccionar y renombrar columnas
@task
def preprocess_columns(df: pd.DataFrame):
    # Seleccionar columnas
    df = df[['_source.complaint_what_happened', '_source.product', '_source.sub_product']]
    
    # Renombrar columnas
    df.rename(columns={
        '_source.complaint_what_happened': 'complaint_what_happened',
        '_source.product': 'category',
        '_source.sub_product': 'sub_product'
    }, inplace=True)
    return df


# crear la columna 'ticket_classification'
@task
def create_ticket_classification(df: pd.DataFrame):
    df['ticket_classification'] = df['category'] + ' + ' + df['sub_product']
    return df


# eliminar columnas redundantes y limpiar datos
@task
def clean_data(df: pd.DataFrame):
    # Eliminar columnas redundantes
    df.drop(columns=['category', 'sub_product'], inplace=True)
    
    # Reemplazar campos vacíos o nulos con NaN
    df['complaint_what_happened'] = df['complaint_what_happened'].replace('', pd.NA)
    
    # Eliminar filas con datos faltantes
    df.dropna(subset=['complaint_what_happened', 'ticket_classification'], inplace=True)
    return df


# reiniciar el índice
@task
def reset_index(df: pd.DataFrame):
    df.reset_index(drop=True, inplace=True)
    return df


# dividir datos
@task
def split_data(df: pd.DataFrame):
    X = df['complaint_what_happened']
    y = df['ticket_classification']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Guardar el label encoder
    pathlib.Path("models").mkdir(exist_ok=True)
    with open("models/label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# guardar tfidf
@task
def train_and_save_tfidf(X_train):
    # Entrenar el TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Guardar el vectorizador
    pathlib.Path("models").mkdir(exist_ok=True, parents=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(tfidf_vectorizer, file)

    return X_train_tfidf


# Entrenar modelo
@task
def train_model(X_train, y_train, model_type="random_forest"):
    # Cargar o inicializar el TF-IDF
    tfidf_path = "models/tfidf_vectorizer.pkl"
    if pathlib.Path(tfidf_path).exists():
        with open(tfidf_path, "rb") as file:
            tfidf_vectorizer = pickle.load(file)
    else:
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train = tfidf_vectorizer.fit_transform(X_train)

        # Guardar el vectorizador
        with open(tfidf_path, "wb") as file:
            pickle.dump(tfidf_vectorizer, file)

    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
    elif model_type == "logistic_regression":
        model = LogisticRegression(C=1.0, solver="liblinear")
    else:
        raise ValueError("Modelo no soportado")

    model.fit(X_train, y_train)
    return model


# Evaluar modelo
@task
def evaluate_model(model, X_test, y_test):
    # Cargar el TF-IDF
    with open("models/tfidf_vectorizer.pkl", "rb") as file:
        tfidf_vectorizer = pickle.load(file)

    X_test = tfidf_vectorizer.transform(X_test)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


# random forest hyperparameter tuning
@task
def grid_search_random_forest(X_train, y_train, X_test, y_test):
    try:
        # Cargar el TF-IDF Vectorizer
        with open("models/tfidf_vectorizer.pkl", "rb") as file:
            tfidf_vectorizer = pickle.load(file)
        
        # Transformar los datos
        X_train_tfidf = tfidf_vectorizer.transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Definir los parámetros para GridSearch
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20]
        }

        # Crear el modelo y realizar la búsqueda de hiperparámetros
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_tfidf, y_train)

        # Obtener el mejor modelo, los mejores parámetros y evaluar
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        accuracy = accuracy_score(y_test, best_model.predict(X_test_tfidf))

        return best_model, best_params, accuracy
    except Exception as e:
        print(f"Error en Grid Search para Random Forest: {e}")
        raise



# logistic regression hyper parameter tuning
@task
def grid_search_logistic_regression(X_train, y_train, X_test, y_test):
    try:
        # Cargar el TF-IDF Vectorizer
        with open("models/tfidf_vectorizer.pkl", "rb") as file:
            tfidf_vectorizer = pickle.load(file)
        
        # Transformar los datos
        X_train_tfidf = tfidf_vectorizer.transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Definir los parámetros para GridSearch
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }

        # Crear el modelo y realizar la búsqueda de hiperparámetros
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_tfidf, y_train)

        # Obtener el mejor modelo, los mejores parámetros y evaluar
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        accuracy = accuracy_score(y_test, best_model.predict(X_test_tfidf))

        return best_model, best_params, accuracy
    except Exception as e:
        print(f"Error en Grid Search para Regresión Logística: {e}")
        raise



# guardar artefactos
@task
def save_artifact(path, obj):
    pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)
    if not pathlib.Path(path).exists():
        with open(path, "wb") as file:
            pickle.dump(obj, file)


# Log y registro de modelos en MLflow
@task
def log_and_register_model(model, accuracy, best_params, run_name, experiment_name, model_registry_name, artifact_path):
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        # Loggear la métrica
        mlflow.log_metric("accuracy", accuracy)
        
        # Loggear hiperparámetros si existen
        if best_params:
            for param, value in best_params.items():
                mlflow.log_param(param, value)
        
        # Registrar el modelo
        mlflow.sklearn.log_model(model, artifact_path)
        
        # Loggear artefactos adicionales
        label_encoder_path = "C:\\Users\\colom\\OneDrive - ITESO\\iteso\\5to semestre\\cienciadatos\\final-exam-pcd2024-autumn\\models\\label_encoder.pkl"
        tfidf_vectorizer_path = "C:\\Users\\colom\\OneDrive - ITESO\\iteso\\5to semestre\\cienciadatos\\final-exam-pcd2024-autumn\\models\\tfidf_vectorizer.pkl"
        
        if pathlib.Path(label_encoder_path).exists():
            mlflow.log_artifact(label_encoder_path, artifact_path="artifacts")
        else:
            print(f"Advertencia: {label_encoder_path} no encontrado.")
        
        if pathlib.Path(tfidf_vectorizer_path).exists():
            mlflow.log_artifact(tfidf_vectorizer_path, artifact_path="artifacts")
        else:
            print(f"Advertencia: {tfidf_vectorizer_path} no encontrado.")
        
        # Crear una versión del modelo en el Model Registry
        model_version = MlflowClient().create_model_version(
            name=model_registry_name,
            source=f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}",
            run_id=mlflow.active_run().info.run_id
        )
    
    return model_version



# Asignar Champion y Challenger
@task
def assign_champion_and_challenger(model_registry_name, models_with_accuracy):
    models_with_accuracy = sorted(models_with_accuracy, key=lambda x: x[1], reverse=True)
    champion = models_with_accuracy[0]
    challenger = models_with_accuracy[1]
    
    client.transition_model_version_stage(
        name=model_registry_name,
        version=champion[0].version,
        stage="Production"
    )
    client.set_registered_model_alias(model_registry_name, "Champion", champion[0].version)
    
    client.transition_model_version_stage(
        name=model_registry_name,
        version=challenger[0].version,
        stage="Staging"
    )
    client.set_registered_model_alias(model_registry_name, "Challenger", challenger[0].version)


# Orquestación las tareas
@flow
def main_flow(raw_data_path):
    # 1. Cargar datos crudos
    raw_df = load_json(raw_data_path)
    
    # 2. Preprocesar datos
    df = preprocess_columns(raw_df)
    df = create_ticket_classification(df)
    df = clean_data(df)
    df = reset_index(df)
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 4. Entrenar y guardar el TF-IDF
    X_train_tfidf = train_and_save_tfidf(X_train)

    # 5. Grid Search para Random Forest
    rf_model, rf_params, rf_accuracy = grid_search_random_forest(X_train, y_train, X_test, y_test)
    
    # 6. Grid Search para Logistic Regression
    lr_model, lr_params, lr_accuracy = grid_search_logistic_regression(X_train, y_train, X_test, y_test)
    
    # 7. Registrar modelos en MLflow
    rf_version = log_and_register_model(
        rf_model, rf_accuracy, rf_params, "RandomForest-GridSearch", 
        EXPERIMENT_NAME, MODEL_REGISTRY_NAME, "gridsearch-random-forest"
    )
    lr_version = log_and_register_model(
        lr_model, lr_accuracy, lr_params, "LogisticRegression-GridSearch", 
        EXPERIMENT_NAME, MODEL_REGISTRY_NAME, "gridsearch-logistic-regression"
    )
    
    # 8. Asignar Champion y Challenger
    assign_champion_and_challenger(
        MODEL_REGISTRY_NAME, [(rf_version, rf_accuracy), (lr_version, lr_accuracy)]
    )



# Ejecutar el pipeline
if __name__ == '__main__':
    main_flow("C:\\Users\\colom\\OneDrive - ITESO\\iteso\\5to semestre\\cienciadatos\\final-exam-pcd2024-autumn\\raw_data\\tickets_classification_eng.json")

