import streamlit as st
import requests

# Configuración inicial
st.set_page_config(page_title="Sistema de Inferencias", layout="centered")

# Configuración de la API
API_URL = "http://api:8000/predict"

# Título de la aplicación
st.title("Sistema de Inferencias para Clasificación de Tickets")
st.write("Interfaz para interactuar con el modelo Champion.")

# Formulario para entrada de texto
st.header("Entrada de Texto")
complaint_text = st.text_area("Ingrese el texto de la queja:", height=150)

# Botón para realizar la predicción
if st.button("Realizar Predicción"):
    if len(complaint_text.strip()) >= 5:  # Validar que el texto sea suficientemente largo
        # Realizar solicitud a la API
        payload = {"complaint_what_happened": complaint_text}
        try:
            with st.spinner("Realizando predicción, por favor espere..."):
                response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("prediction", "No disponible")
                probability = data.get("probability", "No disponible")
                
                # Mostrar resultados
                st.success("¡Predicción realizada exitosamente!")
                st.header("Resultados de la Predicción")
                st.write(f"**Categoría Predicha:** {prediction}")
                if probability != "No disponible":
                    st.write(f"**Probabilidad:** {probability:.2f}")
                else:
                    st.write("**Probabilidad:** No disponible")
            else:
                st.error(f"Error en la API: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error al conectar con la API: {e}")
    else:
        st.warning("Por favor, ingrese un texto válido y suficientemente largo.")

# Pie de página
st.markdown("---")
st.caption("Desarrollado por: Francisco Gonzalez")
