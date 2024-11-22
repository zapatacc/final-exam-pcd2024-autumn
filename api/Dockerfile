FROM python:3.9-slim

WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY app.py /app
COPY requirements.txt /app
#COPY ../artifacts /app/artifacts

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto para la API
EXPOSE 8000

# Comando para iniciar la API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
