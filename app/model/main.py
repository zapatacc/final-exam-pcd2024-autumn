from pydantic import BaseModel
import mlflow
import pandas as pd
import pickle
from fastapi import FastAPI

app = FastAPI()


class InputData(BaseModel):
    complaint_what_happened:str


def predict():
    ...

@app.post("")
def sendPredictions():
    ...