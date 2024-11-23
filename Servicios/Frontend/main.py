# Servicios/Frontend/main.py
import streamlit as st
import pandas as pd
import json
import requests
import time
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Predictor ML",
    page_icon="🤖",
    layout="wide"
)

# Título y descripción
st.title("🤖 Predictor de ML")
st.markdown("""
    Esta aplicación te permite cargar datos en formato JSON y obtener predicciones usando nuestro modelo de ML.

    ### Instrucciones:
    1. Sube tu archivo JSON con los datos
    2. Revisa la vista previa de los datos
    3. Haz clic en 'Realizar Predicción' para obtener resultados
""")


# Función para verificar el estado del backend
def check_backend_health():
    try:
        response = requests.get("http://backend:5000/health")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# Cargar archivo JSON
uploaded_file = st.file_uploader("Sube tu archivo JSON", type='json')

if uploaded_file is not None:
    try:
        # Cargar y mostrar datos
        data = json.load(uploaded_file)
        df = pd.DataFrame(data)

        # Mostrar información del dataset
        st.subheader("📊 Vista Previa de los Datos")
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"Número de registros: {len(df)}")
        with col2:
            st.info(f"Número de características: {len(df.columns)}")

        # Mostrar los primeros registros
        st.dataframe(df.head())

        # Botón para hacer la predicción
        if st.button("🎯 Realizar Predicción"):
            # Verificar el estado del backend
            if not check_backend_health():
                st.error(
                    "❌ El servicio de predicción no está disponible en este momento. Por favor, intenta más tarde.")
            else:
                # Mostrar spinner mientras se procesa
                with st.spinner("Procesando predicción..."):
                    try:
                        # Hacer la petición al backend
                        start_time = time.time()
                        response = requests.post(
                            "http://backend:5000/predict",
                            json=data,
                            timeout=30
                        )
                        end_time = time.time()

                        if response.status_code == 200:
                            result = response.json()
                            predictions = result['predictions']
                            accuracy = result.get('accuracy', 'No disponible')

                            # Mostrar resultados
                            st.success("✅ ¡Predicción completada exitosamente!")

                            # Métricas
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Tiempo de Procesamiento",
                                    f"{(end_time - start_time):.2f} segundos"
                                )
                            with col2:
                                if isinstance(accuracy, (int, float)):
                                    st.metric("Precisión del Modelo", f"{accuracy:.2%}")
                                else:
                                    st.metric("Precisión del Modelo", accuracy)
                            with col3:
                                st.metric("Predicciones Realizadas", len(predictions))

                            # Mostrar predicciones
                            st.subheader("🎯 Resultados de la Predicción")
                            pred_df = pd.DataFrame({"Predicción": predictions})
                            st.dataframe(pred_df)

                            # Opción para descargar resultados
                            st.download_button(
                                label="📥 Descargar Predicciones",
                                data=pred_df.to_csv(index=False).encode('utf-8'),
                                file_name=f'predicciones_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                                mime='text/csv'
                            )
                        else:
                            st.error(f"❌ Error en la predicción: {response.text}")

                    except requests.exceptions.Timeout:
                        st.error(
                            "❌ La solicitud ha excedido el tiempo de espera. Por favor, intenta con menos datos o más tarde.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Error de conexión: {str(e)}")

    except json.JSONDecodeError:
        st.error("❌ Error: El archivo JSON no tiene un formato válido")
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")


