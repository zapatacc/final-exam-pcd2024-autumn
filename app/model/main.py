import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient
import dagshub
from pydantic import BaseModel


dagshub.init(repo_owner='zapatacc', repo_name='final-exam-pcd2024-autumn', mlflow=True)
mlflow.set_experiment("jesus-carbajal-logreg-rf")
TRACKING_URI = mlflow.get_tracking_uri()

run_uri = 'runs:/8ccf9f0120494d78a04c99aa0b113d82/pipeline_model'

dv = mlflow.pyfunc.load_model(run_uri)

def preprocess(input_data):
    input_dict = {
        'complaint_what_happened':input_data.complaint_what_happened
    }
    return input_dict

def predict(input_data):
    X_pred = preprocess(input_data)
    return dv.predict(X_pred)


app = FastAPI()

class InputData(BaseModel):
    complaint_what_happened: float


@app.post("/predict")
def predict_endpoint(input_data: InputData):
    result = predict(input_data)[0]

    return{
        "prediction": result
    }
