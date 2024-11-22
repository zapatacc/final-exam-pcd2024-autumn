import streamlit as st
import requests

# Configurar la URL de la API
API_URL = "http://model:8000/predict"  # Usamos 'model' porque así se llama el servicio en docker-compose

# Título de la aplicación
st.title("Clasificación de Quejas de Clientes")

# Descripción
st.write("""
Ingrese la descripción de la queja del cliente en el campo de texto a continuación, y la aplicación clasificará automáticamente la queja en la categoría correspondiente.
""")

# Campo de entrada de texto
complaint_text = st.text_area("Descripción de la Queja", height=200)

# Botón para realizar la predicción
if st.button("Clasificar Queja"):
    if complaint_text.strip() == "":
        st.warning("Por favor, ingrese la descripción de la queja.")
    else:
        # Datos a enviar a la API
        data = {"complaint_what_happened": complaint_text}

        # Realizar la solicitud POST a la API
        try:
            response = requests.post(API_URL, json=data)
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "No se pudo obtener la predicción.")
                st.success(f"La queja ha sido clasificada como: **{prediction}**")
            else:
                st.error(f"Error en la API: {response.status_code} - {response.reason}")
        except Exception as e:
            st.error(f"Ocurrió un error al comunicarse con la API: {e}")