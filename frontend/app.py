import streamlit as st
import requests

st.title("Modelo de Predicción")

# Campos de entrada para el usuario
input_feature = st.text_input("Ingrese el texto para clasificación:")

if st.button("Predecir"):
    # Preparar los datos para enviar a la API
    data = {"data": [input_feature]}
    # Hacer la solicitud POST a la API
    response = requests.post("http://api:8000/predict", json=data)
    prediction = response.json()["predictions"]
    # Mostrar la predicción
    st.write(f"Predicción: {prediction[0]}")
