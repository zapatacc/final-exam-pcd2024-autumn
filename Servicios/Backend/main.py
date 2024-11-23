# Servicios/Backend/main.py
from flask import Flask, request, jsonify
import mlflow.pyfunc
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score
import os
from dotenv import load_dotenv
import time

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)

# Variable global para el modelo
model = None


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})


@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        # Cargar el modelo si no está cargado
        if model is None:
            try:
                mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
                model = mlflow.pyfunc.load_model("models:/Champion/1")
            except Exception as e:
                print(f"Error al cargar el modelo: {e}")
                return jsonify({"error": "No se pudo cargar el modelo"}), 503

        # Obtener datos JSON
        data = request.get_json()

        if not data:
            return jsonify({"error": "No se proporcionaron datos"}), 400

        # Convertir a DataFrame
        df = pd.DataFrame(data)

        # Realizar predicciones
        predictions = model.predict(df)

        # Calcular precisión si se proporcionan valores reales
        accuracy = None
        if '_source_issue' in df.columns:
            true_values = df['_source_issue']
            accuracy = accuracy_score(true_values, predictions)

        return jsonify({
            "predictions": predictions.tolist(),
            "accuracy": accuracy if accuracy is not None else "No disponible"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5002))
    app.run(host='0.0.0.0', port=port)