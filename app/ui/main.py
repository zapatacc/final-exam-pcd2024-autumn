import streamlit as st
import requests
import json
import numpy as np


#Streamlite app header
st.write("""
# Application to Classify Complaints
         """)

st.sidebar.header('User Input Params')

def convert_int64(obj):
    """Helper function to convert np.int64 to int for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_int64(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj

def user_input_features():
        """Collect user inputs via Streamlit widgets and map them to backend-compatible values."""
        InputComplaint = st.sidebar.text_input("Input Complaint", max_chars=500)
        input_dict = {
             'complaint_what_happened': InputComplaint
        }
        return input_dict

#Collect user inputs
input_dict = user_input_features()
input_dict = convert_int64(input_dict)

# Predict button functionality
if st.button('Predict'):
    try:
        response = requests.post(
            url="http://pcd-car-model-container:4444/predict",
            data=json.dumps(input_dict),
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            prediction = response.json().get('prediction', 'No prediction found')
            st.write(f"El precio estimado de la renta es: {prediction} dólares")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        


