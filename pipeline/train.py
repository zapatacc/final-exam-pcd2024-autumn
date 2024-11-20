import prefect 
from prefect import flow, task
import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn


def removeX(text:str)-> str: 
    return text.replace("X","")


@task(name="readData")
async def readData(path:str)-> json:
    with open(path, "r") as f:
        data = json.load(f)

    return data

@task(name="normalize")
async def normalize(Json:json) -> pd.DataFrame:
    return pd.json_normalize(Json)


@task(name="preprocess")
async def preprocessData(df:pd.DataFrame) -> None:
    df = df[["_source.complaint_what_happened","_source.product","_source.sub_product"]]

    rename = {"_source.complaint_what_happened":"complaint_what_happened",
    "_source.product":"category",
    "_source.sub_product":"sub_product"}

    df = df.rename(rename, axis=1)

    df["ticket_classification"] = df["category"] + " + " + df["sub_product"]
    df = df.drop(columns=["category","sub_product"],axis=1)
    df["complaint_what_happened"] = df["complaint_what_happened"].replace("",pd.NA)

    df.to_csv("../data/preprocessed_data/preprocessed.csv",index=False)

    return None

@task(name="clean")
async def cleanData() -> pd.DataFrame:
    df = pd.read_csv("../data/preprocessed_data/preprocessed.csv")
    df["complaint_what_happened"] = df["complaint_what_happened"].apply(removeX)
    class_counts = df['ticket_classification'].value_counts()
    valid_classes = class_counts[class_counts >= 100].index
    df = df[df['ticket_classification'].isin(valid_classes)]
    df.to_csv("../data/clean_data/cleaned.csv",index=False)

    return df



@flow(name="mainFlow")
async def mainFlow(path:str)->None:
    # data almacena un tipo de dato json que contiene la informacion actualizada del rawData
    data = readData(path)
    # df almacena un tipo de dato DataFrame que contiene el json normalizado
    df = normalize(data)
    # preprocessData es una funcion sin tipo de dato de retorno que actualiza el csv contenido en data/proprocessed_data/ con los datos limpios y listos para procesarse
    preprocessData(df)
    # cleanData no necesita parametros ya que esta pensado para funcionar remplazando el preprocessed.csv en cada ejecucion de este training pipeline por lo que el path es fijo
    cleanedData = cleanData()


    '''
    Next steps add the model into a MLFLOW repo
    '''



mainFlow("../data/raw_data/tickets_classification_eng.json")
