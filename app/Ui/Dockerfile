FROM python:3.11-slim

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

# Instalamos dependencias necesarias y configuramos la zona horaria
RUN apt-get update && apt-get install -y procps && pip install -U pip && \
    rm /etc/localtime && ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime && \
    pip install -r ./requirements.txt

COPY ./main.py /code/
EXPOSE 8501
CMD ["streamlit", "run", "main.py"]
