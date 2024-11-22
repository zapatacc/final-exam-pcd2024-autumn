import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow.pyfunc import load_model
from mlflow import MlflowClient
import dagshub

# Initialize the necessary variables
repo_url = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn"
tracking_uri = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow"
mlflow.set_tracking_uri(uri=tracking_uri)

client = MlflowClient(tracking_uri=tracking_uri)

# Load the model directly from the run URI
model_run_uri = 'runs:/6beeb5e8d3dc4fd99c564ba707443bf3/logreg_pipeline_model'
prediction_model = load_model(model_run_uri)

# FastAPI app initialization
app = FastAPI()

# Pydantic model to validate input data
class ComplaintData(BaseModel):
    incident_description: str

# Preprocessing function that extracts relevant text for prediction
def extract_input_data(complaint: ComplaintData):
    return complaint.incident_description

# Prediction function that returns the model prediction
def get_prediction(input_data: ComplaintData):
    extracted_data = extract_input_data(input_data)
    return prediction_model.predict([extracted_data])

# API endpoint for making predictions
@app.post("/predict")
def prediction_endpoint(input_data: ComplaintData):
    prediction_result = get_prediction(input_data)[0]  # Get the first prediction result
    return {
        "prediction": prediction_result
    }
