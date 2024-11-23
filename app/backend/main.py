import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# MLflow settings

MLFLOW_TRACKING_URI = "https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow"

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

run_ = mlflow.search_runs(order_by=['metrics.f1 ASC'],
                          output_format="list",
                          experiment_names=["erick-model-prefect"]
                          )[0]
run_id = run_.info.run_id

model_name = "erick-model-perfect"
alias = "champion" # Se busca el modelo champion

model_uri = f"models:/{model_name}@{alias}"

champion_model = mlflow.pyfunc.load_model(
    model_uri=model_uri
)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Define the preprocessing function
def preprocess(input_data):
    def remove_sensitive_info(text):
        if isinstance(text, str):  # Ensure input is a string
            return re.sub(r'X{2,}', '', text)  # Remove XX, XXX, XXXX
        return text

    def remove_stopwords(text):
        tokens = word_tokenize(text)  # Tokenize the text
        filtered_text = [word for word in tokens if word.lower() not in stop_words and not re.match(r'[^\w\s]', word)]
        return ' '.join(filtered_text)

    # Apply preprocessing steps
    input_data = remove_sensitive_info(input_data)
    input_data = remove_stopwords(input_data)

    return input_data

# Predict function
def predict(input_data):
    # Preprocess input data
    preprocessed_data = preprocess(input_data)

    # The model expects a list or array of inputs
    predictions = champion_model.predict([preprocessed_data])

    return predictions[0]

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict_endpoint(input_text: str):
    prediction = predict(input_text)
    return {"prediction": prediction}