import streamlit as st
import requests
import json
import pickle
st.write("""
# TICKET CLASSIFICATION
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    complaint = st.sidebar.text_area("What happened?")
    input_data = {
        'complaint_what_happened': complaint
    }
    return input_data


input_data = user_input_features()


# Ruta del archivo LabelEncoder
label_encoder_path = r"C:\Users\jplv0\PycharmProjects\final-exam-pcd2024-autumn\training_pipeline\models\label_encoder2.pkl"

# Cargar el LabelEncoder desde el archivo
with open(label_encoder_path, "rb") as file:
    label_encoder = pickle.load(file)


# Crear el diccionario con el mapeo del índice al nombre de la clase
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}



if st.button('Predict'):
    response = requests.post(
        url="http://localhost:8000/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(input_data)
    )

    if response.status_code == 200:
        numeric_prediction = int(response.json()['prediction'])
        category_prediction = label_mapping[numeric_prediction]
        st.write(f"The category is", category_prediction)
    else:
        st.write("Error en la predicción.")
        st.write(f"Detalle del error: {response.text}")