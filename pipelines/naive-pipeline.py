import pickle
import mlflow
import pathlib
import dagshub
import pandas as pd
from hyperopt.pyll import scope
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import task
import nltk
from prefect import flow
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pathlib
from mlflow.tracking import MlflowClient
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "LuisFLopezA"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "37d1c615f665c61af97d8a85683704cd5ca42315"




mlflow.set_tracking_uri("https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow")


@task
def read_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        print(f"Data successfully loaded. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    
    

nltk.download('punkt')
nltk.download('stopwords')

@task
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['cleaned_complaint'] = df['cleaned_complaint'].fillna("").astype(str)
    df['ticket_classification'] = df['ticket_classification'].fillna("").astype(str)
    df['combined_text'] = df['cleaned_complaint'] + " " + df['ticket_classification']
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9+\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=2000)
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    tfidf_df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, tfidf_df], axis=1)

    for col in df.select_dtypes(include=['int']).columns:
        df[col] = df[col].astype(float)

    print(f"Feature engineering complete. New DataFrame shape: {df.shape}")
    return df



@task
def hyperparameter_tuning(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(params):
        model = MultinomialNB(alpha=params['alpha'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return {'loss': -accuracy, 'status': STATUS_OK}  
    search_space = {
        'alpha': hp.uniform('alpha', 0.1, 1.0)
    }

    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials()
    )

    print(f"Best Parameters: {best_params}")
    return best_params


@task
def train_best_model(X_train, X_val, y_train, y_val, best_params: dict) -> None:
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer

    mlflow.set_tracking_uri("https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow")
    experiment_name = "luis-lopez Best Naive Bayes Model"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="luis-lopez Final Model Training"):
        
        mlflow.log_params(best_params)

        X_train["text"] = X_train["text"].fillna("").astype(str)
        X_val["text"] = X_val["text"].fillna("").astype(str)
        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        X_train_vectorized = vectorizer.fit_transform(X_train["text"])
        X_val_vectorized = vectorizer.transform(X_val["text"])
        model = MultinomialNB(alpha=best_params['alpha'])
        model.fit(X_train_vectorized, y_train)
        y_pred = model.predict(X_val_vectorized)
        accuracy = accuracy_score(y_val, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="luis-lopez_naive_bayes_model"
        )
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact("vectorizer.pkl", artifact_path="preprocessing")

    print(f"Model training complete. Accuracy: {accuracy:.4f}")

@task
def label_and_register_models():
    mlflow.set_tracking_uri("https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow")
    experiment_name = "luis-lopez Best Naive Bayes Model"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )
    if len(runs) < 2:
        raise ValueError("Not enough runs to determine champion and challenger models.")
    champion_run = runs[0]
    challenger_run = runs[1]

    champion_model_name = "luis-lopez_champion_naive_bayes"
    champion_model_uri = f"runs:/{champion_run.info.run_id}/model"
    registered_champion = mlflow.register_model(model_uri=champion_model_uri, name=champion_model_name)
    client.set_registered_model_tag(name=champion_model_name, key="model_role", value="champion")
    client.set_registered_model_tag(name=champion_model_name, key="author", value="luis-lopez")
    challenger_model_name = "luis-lopez_challenger_naive_bayes"
    challenger_model_uri = f"runs:/{challenger_run.info.run_id}/model"
    registered_challenger = mlflow.register_model(model_uri=challenger_model_uri, name=challenger_model_name)
    client.set_registered_model_tag(name=challenger_model_name, key="model_role", value="challenger")
    client.set_registered_model_tag(name=challenger_model_name, key="author", value="luis-lopez")

    print(f"Champion model registered: {champion_model_name}")
    print(f"Challenger model registered: {challenger_model_name}")
   

@flow
def pipeline():
    file_path = "../raw_data/final_df.csv"
    df = read_data(file_path)
    df_features = feature_engineering(df)
    target_column = "predicted_topic"
    feature_columns = [col for col in df_features.columns if col not in ["processed_text", "combined_text"] and pd.api.types.is_numeric_dtype(df_features[col])]
    X = df_features[feature_columns]
    y = df_features[target_column]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    best_params = hyperparameter_tuning(X, y)

    train_best_model(X_train, X_val, y_train, y_val, best_params)

    label_and_register_models()


if __name__ == "__main__":
    pipeline()
