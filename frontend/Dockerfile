FROM python:3.9-slim

WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY frontend.py /app
COPY requirements.txt /app

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto para Streamlit
EXPOSE 8501

# Comando para iniciar el frontend
CMD ["streamlit", "run", "frontend.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
