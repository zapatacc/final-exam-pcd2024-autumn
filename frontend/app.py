import streamlit as st
import requests

# Título de la aplicación
st.title("Complaint classifier")

# Campo de entrada para la queja
complaint_text = st.text_area("Type in client's complaint:")

# Botón para enviar la solicitud a la API
if st.button("Classify"):
    # Validar que el campo no esté vacío
    if complaint_text.strip() == "":
        st.warning("Type a valid complaint, please.")
    else:
        # URL del backend
        api_url = "http://api:8000/predict"  # Cambia "api" por "localhost" si pruebas localmente
        
        try:
            # Enviar solicitud POST a la API
            response = requests.post(api_url, json={"text": complaint_text})
            
            # Manejo de respuestas
            if response.status_code == 200:
                data = response.json()
                if "prediction" in data:
                    prediction = data["prediction"]
                    st.success(f"La clasificación predicha es: {prediction}")
                else:
                    st.error("La respuesta de la API no contiene la predicción.")
            else:
                st.error(f"Error al conectar con la API. Código de estado: {response.status_code}")
                st.write("Detalles:", response.text)
        
        except requests.exceptions.RequestException as e:
            st.error("No se pudo conectar con la API. Verifica que esté en funcionamiento.")
            st.write("Detalles del error:", str(e))



# http://localhost:8501 URL para el streamlit.


# puerto 8051 porque es comúnmente usado por streamlit. funciona
