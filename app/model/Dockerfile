# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar paquetes de Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requisitos
COPY requirements.txt .

# Actualizar pip a la última versión
RUN pip install --upgrade pip

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Descargar recursos de NLTK
RUN python -m nltk.downloader stopwords wordnet

# Copiar el resto del código fuente
COPY . .

# Exponer el puerto en el que corre la API
EXPOSE 8000

# Comando para ejecutar la aplicación con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]