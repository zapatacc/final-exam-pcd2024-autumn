import streamlit as st
import requests

st.title("Clasificador de Quejas")

complaint_text = st.text_area("Ingrese la queja del cliente:")

if st.button("Clasificar"):
    if complaint_text.strip() == "":
        st.warning("Por favor, ingrese una queja válida.")
    else:
        # Realizar la solicitud a la API
        response = requests.post(
            "http://localhost:8000/predict",
            json={"text": complaint_text}
        )
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"La clasificación predicha es: {prediction}")
        else:
            st.error("Error al conectar con la API.")

# ejecutar la app de streamlit
# streamlit run app.py