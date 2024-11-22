import streamlit as st
import json
import requests

st.write("""
# Ticket Classification Predictor
Esta herramienta utiliza un modelo de aprendizaje automático para clasificar tickets basados en las descripciones de problemas.
""")

st.sidebar.header("Parámetros de Entrada")

def user_input_features():
    """
    Captura los datos del usuario desde la barra lateral de Streamlit.
    """
    complaint = st.sidebar.text_area("Descripción del problema (complaint_what_happend)", height=200)

    data = {'complaint_what_happend': complaint}
    return data

# Capturar datos del usuario
input_data = user_input_features()

if st.button("Realizar Predicción"):
    """
    Envía los datos capturados al endpoint de la API y muestra el resultado.
    """
    try:
        response = requests.post(
            url="http://127.0.0.1:8000/predict",  # Cambiar si el endpoint está en otro servidor
            data=json.dumps(input_data),
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            st.write(f"### Clasificación del ticket: **{result['ticket_classification']}**")
        else:
            st.error(f"Error en la predicción: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"No se pudo conectar con la API. Error: {e}")
