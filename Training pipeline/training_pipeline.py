import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import dagshub
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import pickle
import pathlib
from mlflow.tracking import MlflowClient
from prefect import task,flow
import json
import contractions
import re
import nltk
from nltk.corpus import stopwords

@task(name = "data wrangling")
def data_wrangling(file_path):
    """Carga el archivo JSON y lo convierte en un DataFrame limpio para modelar."""
    with open(file_path, "r") as file:
        data = json.load(file)
    df = pd.json_normalize(data)

    # Seleccionamos columnas necesarias
    df = df[['_source.complaint_what_happened', '_source.product', '_source.sub_product']]

    # Renombramos columnas
    df.rename(columns={
        '_source.complaint_what_happened': 'complaint_what_happened',
        '_source.product': 'category',
        '_source.sub_product': 'sub_product'
    }, inplace=True)

    # Creamos columna 'ticket_classification'
    df['ticket_classification'] = df['category'] + ' + ' + df['sub_product']

    # Eliminamos las columnas innecesarias
    df.drop(columns=['category', 'sub_product'], inplace=True)

    # Reemplazamos valores vacíos con valore nulos
    df['complaint_what_happened'].replace('', pd.NA, inplace=True)

    # Eliminamos filas con datos faltantes
    df.dropna(subset=['complaint_what_happened', 'ticket_classification'], inplace=True)

    # Reiniciamos el  índice
    df.reset_index(drop=True, inplace=True)

    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        # Expandir contracciones
        text = contractions.fix(text)

        # Eliminar 'X' mayúsculas
        text = re.sub(r'X+', '', text)

        # Convertir a minúsculas
        text = text.lower()

        # Tokenizar texto
        words = nltk.word_tokenize(text)

        # Eliminar stopwords
        words = [word for word in words if word not in stop_words]

        # Unir palabras
        return ' '.join(words)

    df['complaint_what_happened'] = df['complaint_what_happened'].apply(clean_text)

    counts = df['ticket_classification'].value_counts()
    valid_categories = counts[counts >= 50].index
    df = df[df['ticket_classification'].isin(valid_categories)]
    df = df.drop_duplicates()

    return df
@task(name = "split data")
def split_data(df):

