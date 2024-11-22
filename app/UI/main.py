import streamlit as st
import requests
import json

st.write("""
# Clasificación de la predicción del ticket""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    inputmodel = st.sidebar.text_area("Incidente:")
    input_dict = {
        'complaint_what_happened': inputmodel
    }
    return input_dict


input_dict = user_input_features()

if st.button('Predict'):
    response = requests.post(
        url="http://dafne-model-container:8000/predict",  
        headers={"Content-Type": "application/json"},
        data=json.dumps(input_dict)  
    )

    if response.status_code == 200:
        st.write(f"The category is: {response.json()['prediction']}")
    else:
        st.write("Error en la predicción.")
        st.write(f"Detalle del error: {response.text}")