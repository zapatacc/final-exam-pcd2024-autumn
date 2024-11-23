import streamlit as st
import requests
import json

st.write("""
# Application to predict the ticket classification
""")

st.sidebar.header('User Input Parameters')


def user_input_features():
    PU = st.sidebar.text_input("Input complaint")

    input_dict = {
        "complaint_what_happened": PU,

    }
    return input_dict


input_dict = user_input_features()

# hacer request al api
if st.button("Predict"):
    response = requests.post(
        url="http://model:8000/predict",
        data=json.dumps(input_dict)
    )

    try:
        response_data = response.json()
        st.write(f"The ticket is: {response_data['ypred'][0]}")
    except Exception as e:
        st.error(f"Error en la respuesta del backend: {e}")
        st.write("Respuesta del backend:", response.text)


