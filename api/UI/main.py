import streamlit as st
import requests
import json
import numpy as np

# Streamlit app header
st.title("Complaint Classification Application")

# Sidebar header
st.sidebar.header('Input Parameters')

def convert_to_native_type(obj):
    """Helper function to convert np.int64 to native int for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_native_type(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_type(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj

def collect_user_input():
    """Collect user inputs via Streamlit widgets and map them to backend-compatible values."""
    complaint_description = st.sidebar.text_area("Enter Complaint Details", max_chars=500)
    return {'complaint_description': complaint_description}

def request_prediction(data):
    """Send input data to the backend for prediction and handle the response."""
    try:
        response = requests.post(
            url="http://renata-pcd-model-container:8000/predict",  # Ensure backend server is running on localhost:8000
            data=json.dumps(data),
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            prediction = response.json().get('prediction', 'No prediction available')
            return f"The predicted complaint category is: {prediction}"
        else:
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        return f"An error occurred while connecting to the backend: {e}"

# Collect user inputs
user_data = collect_user_input()
user_data = convert_to_native_type(user_data)

# Predict button functionality
if st.button('Make Prediction'):
    result_message = request_prediction(user_data)
    st.write(result_message)
