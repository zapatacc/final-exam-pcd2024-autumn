import pickle
import mlflow
import dagshub
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn"

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

model_name = "arturo-prefect-model"
alias = "champion"

model_uri = f"models:/{model_name}@{alias}"

champion_model = mlflow.pyfunc.load_model(
    model_uri=model_uri
)

with open("Vectorizer/vectorizer.pkl", "rb") as f_in:
    dv = pickle.load(f_in)

with open("LabelEncoder/labelencoder.pkl", "rb") as a_in:
    label_encoder = pickle.load(a_in)


def preprocess(input_data):

    input_dict = {
        'complaint_what_happened': input_data.text,
    }

    return dv.transform(input_dict)


def predict(input_data):

    X_val = preprocess(input_data)

    return champion_model.predict(X_val)

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict_endpoint(input_data: InputData):
    result = predict(input_data)[0]
    predicted_label = label_encoder.inverse_transform(result)
    return {"prediction": predicted_label[0]}