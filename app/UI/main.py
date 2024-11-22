import streamlit as st
import requests
import json

st.write("""
# App that predicts the category of the complain for the Chase bank
""")

st.sidebar.header('User complain')

def user_input_features():
    text = st.sidebar.text_input("Put your complain")

    input_dict = {
        'complaint_what_happened': text,
    }

    return input_dict

input_dict = user_input_features()

if st.button('Predict'):
    response = requests.post(
        #url = "http://chase-complains-model-container:8000/predict",
        url="http://model:8000/predict",
        #url = "http://nyc-taxi-model-container:8000/predict",
        data = json.dumps(input_dict)
    )

    st.write(f"La queja es de la categoría {response.json()['prediction']} .")