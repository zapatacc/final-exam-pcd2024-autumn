import streamlit as st
import requests

st.title("Clasificador de Quejas")

complaint_text = st.text_area("Ingrese la queja del cliente:")

if st.button("Clasificar"):
    if complaint_text.strip() == "":
        st.warning("Por favor, ingrese una queja válida.")
    else:
        response = requests.post(
            "http://api:8001/predict",  # Asegúrate de que el puerto es correcto
            json={"text": complaint_text}
        )
        st.write("Estado de la respuesta:", response.status_code)
        st.write("Contenido de la respuesta:", response.json())
        if response.status_code == 200:
            data = response.json()
            if "prediction" in data:
                prediction = data["prediction"]
                st.success(f"La clasificación predicha es: {prediction}")
            else:
                st.error(f"Error en la respuesta de la API: {data}")
        else:
            st.error("Error al conectar con la API.")