from pydantic import BaseModel
import mlflow
import pandas as pd
import pickle
from fastapi import FastAPI
from mlflow.tracking import MlflowClient

# se establece la uri de seguimiento para mlflow
mlflow.set_tracking_uri('https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow')

# se inicializa una aplicacion fastapi
app = FastAPI()

# se define el modelo de entrada para la api
class InputData(BaseModel):
    complaint_what_happened: str

# funcion para obtener el run id a partir del nombre del modelo y un alias
def get_run_id_from_alias(model_name: str, alias: str) -> str:
    '''
    entrada de la funcion
    model_name: nombre del modelo en mlflow
    alias: alias asociado a una version del modelo
    
    que hace la funcion
    obtiene el run id de la version del modelo asociada al alias
    
    salida de la funcion
    run id correspondiente al alias
    '''
    client = MlflowClient()
    alias_info = client.get_model_version_by_alias(name=model_name, alias=alias)
    return alias_info.run_id

# funcion para realizar una prediccion usando un modelo registrado en mlflow
def predict(data: dict):
    '''
    entrada de la funcion
    data: diccionario con los datos de entrada para el modelo
    
    que hace la funcion
    carga un modelo de mlflow utilizando un run id, transforma los datos de entrada y genera una prediccion
    
    salida de la funcion
    prediccion decodificada realizada por el modelo
    '''
    model_name = "patricio-model"
    alias = "champion"

    run_id = get_run_id_from_alias(model_name=model_name, alias=alias)

    model_uri = f"runs:/{run_id}/pipeline_model"

    # carga el modelo registrado en mlflow
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

    # descarga el label encoder utilizado para transformar las etiquetas
    label_encoder_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="label_encoder.pkl")
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # crea un dataframe a partir de los datos de entrada
    df = pd.DataFrame([data])

    # genera la prediccion codificada
    prediction_encoded = loaded_model.predict(df["complaint_what_happened"])

    # decodifica la prediccion
    prediction_decoded = label_encoder.inverse_transform(prediction_encoded)

    return prediction_decoded

# endpoint para realizar una prediccion a traves de la api
@app.post("/predict")
def predict_endpoint(input_data: InputData):
    '''
    entrada de la funcion
    input_data: datos de entrada en formato InputData
    
    que hace la funcion
    convierte los datos de entrada en un diccionario, realiza una prediccion y la retorna como respuesta json
    
    salida de la funcion
    respuesta json con la prediccion realizada
    '''
    input_dict = input_data.dict()

    result = predict(input_dict)

    return {
        "prediction": result[0]
    }
