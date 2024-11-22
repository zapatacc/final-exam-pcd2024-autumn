# se usa la imagen base de python 3.11 en su version slim
FROM python:3.11-slim

# se establece el directorio de trabajo dentro del contenedor
WORKDIR /code

# se actualizan los paquetes, se instala procps, se actualiza pip y se configura la zona horaria a mexico city
RUN apt-get update && apt-get install -y procps && pip install -U pip && rm /etc/localtime && ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime

# se copia el archivo requirements.txt al directorio de trabajo
COPY ./requirements.txt /code/requirements.txt

# se instalan las dependencias definidas en el archivo requirements.txt
RUN pip install -r ./requirements.txt

# se copia el archivo main.py al directorio de trabajo
COPY ./main.py /code/

# se expone el puerto 8501 para que la aplicacion sea accesible desde el exterior
EXPOSE 8501

# se define el comando para ejecutar la aplicacion con streamlit
CMD ["streamlit","run", "main.py"]
