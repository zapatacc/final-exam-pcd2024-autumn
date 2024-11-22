import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
from prefect import task, flow
from sklearn.svm import SVC
import mlflow
import dagshub
from mlflow.tracking import MlflowClient

def preprocess_text(text):
    # minusculas
    text = text.lower()
   
    # Eliminar puntuación
    text = ''.join([char for char in text if char not in string.punctuation])
 
    # Eliminar cadenas con más de dos 'X' consecutivas
    text = re.sub(r'x{2,}', '', text)
 
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
   
    # Tokenización
    words = nltk.word_tokenize(text)
   
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
   
    # Lematización
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
   
    # Unir las palabras preprocesadas
    return ' '.join(words)
 
@task(name="read_data")
def read_csv():
    df = pd.read_csv("../data/clean/processed_data.csv")
    return df
 
@task(name="preprocess")
def apply_preprocess(df):
 
    df['complaint_what_happened'] = df['complaint_what_happened'].apply(preprocess_text)
    df['ticket_classification'] = df['ticket_classification'].apply(preprocess_text)
    X = df['complaint_what_happened']
    y = df['ticket_classification']
 
    return X, y
 
   
 
@task(name="splits")
def vectorizer(X, y):
 
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)
    with open ("LabelEncoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)

    tfidf = TfidfVectorizer()
 
    X = tfidf.fit_transform(X)
    with open ("TfidfVectorizer.pkl", "wb") as file:
        pickle.dump(tfidf, file)
 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
 
    return X_train, X_test, y_train, y_test
 
@task(name="model")
def run_model(X_train, X_test, y_train, y_test):
    models_gridSearch = {
    "SVC": {
        "model": SVC(),
        "params": {
            "C": [1],           # Regularización
            "kernel": ["linear"], # Tipos de kernel
            "gamma": ["auto"]   # Coeficiente del kernel
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(),
        "params": {
            "C": [0.1, 1, 5],          # Regularización
            "solver": ["liblinear"], # Algoritmo de optimización
            "penalty": ["l2"]           # Tipo de penalización
        }
    }
}
   
    model = models_gridSearch["SVC"]["model"]
    params = models_gridSearch["SVC"]["params"]
    grid = GridSearchCV(model, params, scoring="accuracy", cv=3)
    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)
    mlflow.set_experiment("mariapaula-perez-experiments")

    with mlflow.start_run(run_name="mariapaula-perez-SVC"):
        grid.fit(X_train,y_train)
        model = grid.best_estimator_
        y_pred = model.predict(X_test)
    
        accuracy = accuracy_score(y_test,y_pred)
    
        mlflow.log_artifact("LabelEncoder.pkl")
    

        mlflow.log_artifact("TfidfVectorizer.pkl")
    
        ## Log del modelo
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", accuracy)
    
        mlflow.sklearn.log_model(model, artifact_path=f"SVC-maripau")

@task(name="Champion/challenger")
def best_model(experiments: str, model_name: str = "mariapaula-model"):
    client = MlflowClient()
    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)


    all_runs = mlflow.search_runs(
        experiment_names=[experiments]
    )


    best_accuracy = -float("inf")
    second_best_accuracy = -float("inf")
    best_run_id = None
    second_best_run_id = None


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

    if best_run_id:  
        model_uri = f"runs:/{best_run_id}/model"
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        best_model_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version

        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=best_model_version
        )

    if second_best_run_id: 
        model_uri = f"runs:/{second_best_run_id}/model"
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        second_best_model_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version

        client.set_registered_model_alias(
            name=model_name,
            alias="challenger",
            version=second_best_model_version
        )
    return True

@flow
def main_flow():
    df = read_csv()
    X,y = apply_preprocess(df)
    X_train, X_test, y_train, y_test = vectorizer(X,y)
    run_model(X_train, X_test, y_train, y_test)
    exit= best_model("mariapaula-perez-experiments")


main_flow()
