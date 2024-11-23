import streamlit as st
import requests
import json

st.write("""
# Application to predict complaint category
""")

st.sidebar.header('User Input Complaint')

def user_input_features():
    complaint = st.sidebar.text_input('Complaint Text', "This service sucks!")

    input_dict = {
        'Complaint': complaint
    }
    return input_dict

input_dict = user_input_features()

if st.button('Predict'):
    response = requests.post(
        #url="http://localhost:8000/predict",
        url="http://erick-examen-backend-container:8000/predict",
        data=json.dumps(input_dict)
    )

    st.write(f"The predicted category is: {response.json()['prediction']}")