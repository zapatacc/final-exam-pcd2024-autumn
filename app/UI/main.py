import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import json
import requests


st.write("""
# Application to predict the ticket classification
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    PU = st.sidebar.text_input("Input complaint")


    input_dict={
        "complaint_what_happened":PU,
    

    }
    return input_dict

input_dict = user_input_features()




#hacer request al api
if st.button("Predict"):
    response = requests.post(
        url = "http://ticket-prediction:8000/predict",
        data = json.dumps(input_dict))

    st.write(f"el ticket es : {response.json()['ypred'][0]}")