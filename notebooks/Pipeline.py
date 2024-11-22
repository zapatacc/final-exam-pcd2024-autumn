import pickle
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import mlflow
from sklearn.preprocessing import LabelEncoder
import re
import contractions
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from mlflow import MlflowClient
import dagshub
from prefect import flow, task
import json



###########################################################
stop_words = set(stopwords.words('english'))


def delete_frequent_words(corpus, threshold=0.75):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    word_counts = X.sum(axis=0).A1 
    word_list = vectorizer.get_feature_names_out()
    threshold_count = len(corpus) * threshold
    frequent_words = {word_list[i] for i, count in enumerate(word_counts) if count > threshold_count}
    filtered_corpus = []
    for doc in corpus:
        filtered_doc = ' '.join([word for word in doc.split() if word not in frequent_words])
        filtered_corpus.append(filtered_doc)
    return filtered_corpus
    
def clean_complaint(complaint):
    complaint = complaint.lower()
    complaint = contractions.fix(complaint)
    #complaint = re.sub(r'xx+', '', complaint)
    complaint = re.sub(r'\W', ' ', complaint)
    complaint_tokens = word_tokenize(complaint)
    complaint = ' '.join([word for word in complaint_tokens if word not in stop_words])
    return complaint

###########################################################

@task(name="loadExperiment")
def loadinit():
    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)
    mlflow.set_experiment("jesus-carbajal-logreg")

@task(name="readJson")
def readJson(file_path):
    with open(file_path, "r") as file:  
        datos = json.load(file)
    df = pd.json_normalize(datos)
    return df

@task(name="wrangling")
def processData(df):

    df = df[['_source.complaint_what_happened', '_source.product', '_source.sub_product']]
    df = df.rename(columns={'_source.complaint_what_happened':'complaint_what_happened','_source.product':'category','_source.sub_product':'sub_product'})
    df['ticket_classification'] = df['category'] + " + " + df['sub_product']
    df = df.drop(['category', 'sub_product'], axis=1)
    df['complaint_what_happened'] = df['complaint_what_happened'].replace("", pd.NA)
    df = df.dropna()
    df = df.reindex()
    df.to_csv('../data/transformed_data/preprocessed.csv', index=False)

    return df


@task(name="cleaning")
def cleanse(df):
    batch_size = 1000
    cleaned_corpus = []
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch_complaints = df['complaint_what_happened'][start:end]
        cleaned_batch = batch_complaints.apply(clean_complaint)
        cleaned_corpus.extend(cleaned_batch)
    cleaned_corpus = delete_frequent_words(cleaned_corpus, threshold=0.80)
    df['complaint_what_happened'] = cleaned_corpus

    counts = df['ticket_classification'].value_counts()
    todelete = counts[counts < 10]
    df = df[~df['ticket_classification'].isin(todelete)]

    return df

@task(name="vectorize")
def vectorizer(df):
    X = df['complaint_what_happened']
    y = df['ticket_classification']

    labelmapping = LabelEncoder()
    y_mapped = labelmapping.fit_transform(y)
    y = y_mapped.tolist()


    text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.20, random_state = 309)

    return text_train, text_test, sent_train, sent_test, labelmapping

@task(name='train')
def trainlogreg(text_train, text_test, sent_train, sent_test, labelmapping):
    
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words = stopwords.words('english'))),
        ("logreg", LogisticRegression(max_iter=600))
    ])
    
    params_grid = {
    'logreg__C': [0.01, 0.1, 1, 0.5],
    'logreg__penalty': ['l2'],
    'logreg__solver': ['lbfgs'],
    }
    
    grid_search = GridSearchCV(pipeline, params_grid, scoring='accuracy', cv=5, n_jobs=1, verbose=2)

    with mlflow.start_run(run_name="Logreg Pipeline"):

        grid_search.fit(text_train, sent_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(text_test)
        
        # calcular métricas
        accuracy = accuracy_score(sent_test, y_pred)
        report = classification_report(sent_test, y_pred, output_dict=True)
        
        # Loggear el mejor modelo
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])
        
        mlflow.sklearn.log_model(best_model, artifact_path="pipeline_model")

        with open("labelmapping.pkl", "wb") as f:
            pickle.dump(labelmapping, f)
        mlflow.log_artifact("labelmapping.pkl")

@task(name="getBestModel")
def getChamp():
    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)
    mlflow.set_experiment("jesus-carbajal-logreg")

    mlflow.set_experiment("jesus-carbajal-logreg-label-encoder")
    client = MlflowClient()

    # Get the experiment ID
    experiment = client.get_experiment_by_name("jesus-carbajal-logreg-label-encoder")
    experiment_id = experiment.experiment_id

    #Get 100 experiments
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=100,
        order_by=["start_time desc"]
    )

    best_run = 0
    best_accuracy = 0

    for run in runs:
        accuracy = run.data.metrics.get('accuracy', None)
        if accuracy is not None and accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run = run

    return best_run.info.run_id

@task(name="SetChamp")
def setChamp(run_id, model_name):
    dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)
    mlflow.set_experiment("jesus-carbajal-logreg")

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    model_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version
    client.set_registered_model_alias(
        name=model_name,
        alias="Champ",
        version=model_version
    )

@flow(name="mainFlow")
def mainFlow(json):
    loadinit()
    data = readJson(json)
    processed = processData(data)
    df = cleanse(processed)
    text_train, text_test, sent_train, sent_test, label_encoder = vectorizer(df)
    trainlogreg(text_train, text_test, sent_train, sent_test, label_encoder)
    champ = getChamp()
    setChamp(champ, "jesus-carbajal-model")

mainFlow("../data/raw_data/tickets_classification_eng.json")










