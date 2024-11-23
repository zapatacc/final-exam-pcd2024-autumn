import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pathlib
import pandas as pd
import json
import numpy as np


def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    file_path = "../raw_data/tickets_classification_eng.json"

    with open(file_path, "r", encoding="utf-8") as file:
        datos = json.load(file)

    df = pd.json_normalize(datos)

    columnas = ['_source.complaint_what_happened', '_source.product', '_source.sub_product']
    df = df[columnas]

    # renombramos
    df.columns = ['complaint_what_happened', 'category', 'sub_product']

    # creacion de nueva columna
    df['ticket_classification'] = df['category'] + ' + ' + df['sub_product']

    #dropeamos columnas redundantes
    df = df.drop(columns=['sub_product', 'category'])

    #limpieza de datos
    df['complaint_what_happened'] = df['complaint_what_happened'].replace('', np.nan)

    #eliminacion de datos faltantes
    df = df.dropna(subset=['complaint_what_happened', 'ticket_classification'])

    # Reiniciar Indice
    df = df.reset_index(drop=   True)

    return df
'''
def train_test(df: pd.DataFrame):
    # Dividir características y etiquetas
    X = df['complaint_what_happened']
    y = df['ticket_classification']

    # Vectorización de texto usando TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Guardar el TF-IDF vectorizer en un archivo
    pathlib.Path("models").mkdir(exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f_model:
        pickle.dump(tfidf_vectorizer, f_model)

    # Continuar con el entrenamiento y división de datos
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Convertir las etiquetas en valores numéricos
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    pathlib.Path("models").mkdir(exist_ok=True)
    with open("models/label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)
    y_train = label_encoder.fit_transform(y_train)
    y_test = y_test.map(lambda x: x if x in label_encoder.classes_ else 'unknown')
    label_encoder.classes_ = np.append(label_encoder.classes_, 'unknown')
    y_test = label_encoder.transform(y_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = read_data("../raw_data/tickets_classification_eng.json")
    train_test(df)
'''
import pickle

from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

# Supongamos que `y_train` contiene las etiquetas del conjunto de entrenamiento
label_encoder = LabelEncoder()

file_path = "../raw_data/tickets_classification_eng.json"
df = read_data(file_path)

# Ajustar el LabelEncoder con las etiquetas
y = df['ticket_classification'] # Reemplaza con las etiquetas de tu conjunto de entrenamiento
label_encoder.fit(y)

# Guardar el LabelEncoder ajustado en un archivo .pkl
with open("models/label_encoder2.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

