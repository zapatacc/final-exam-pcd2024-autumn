import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient

# MLflow settings
dagshub_repo = "https://dagshub.com/" #!!!!!!!

MLFLOW_TRACKING_URI = "https://dagshub.com/.mlflow" #!!!!!!!

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

run_ = mlflow.search_runs(order_by=['metrics.accuracy'],
                          output_format="list",
                          experiment_names=[""] #!!!!!!!
                          )[0]

run_id = run_.info.run_id

run_uri =  #!!!!!!!

dv = mlflow.pyfunc.load_model(run_uri)

def preprocess(input_data):
    input_dict = {
        'complaint_what_happened':input_data.complaint_what_happened
    }
    return input_dict

def predict(input_data):

    X_pred = vectorize(input_data) #!!!!!!!

    return dv.predict(X_pred)

app = FastAPI()

class InputData(Basemode):
    complaint_what_happened: float


@app.post("/predict")
def predict_endpoint(input_data: InputData):
    result = predict(input_data)[0]

    return{
        "prediction": float(result)
    }
