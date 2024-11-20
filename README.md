# Examen Final - Proyecto en Ciencia de Datos

## Información General

Este repositorio contiene todos los archivos e información esenciales necesarios para prepararse para el examen final del curso **'Proyecto en Ciencia de Datos'** en la Universidad ITESO para el semestre de Otoño de 2024.

## Instrucciones

Antes de comenzar el examen, por favor, lea cuidadosamente las siguientes indicaciones:

- El objetivo de este examen es evaluar sus conocimientos como Científico de Datos en el desarrollo de un proyecto de principio a fin.
- Como líder de este proyecto, tiene la autoridad para tomar decisiones y asumir lo que considere necesario, **siempre justificando sus elecciones**.
- Se le presentará un escenario de la vida real con un conjunto de datos para abordar el problema.
- Clone este repositorio en su local:  
  ```bash
  git clone git@github.com:zapatacc/final-exam-pcd2024-autumn.git
  ```
- Después de clonar, cree una rama con su nombre y apellido. Ejemplo: `cristian-zapata`
- Trabaje en dicha rama.
- Se va a utilizar `mlflow` para realizar experimentos y registrar modelos, por lo que debe enlazar su repositorio de `github` con `dagshub`
- La solución del problema debe abordar, como mínimo, los aspectos básicos de un proyecto de ciencia de datos, incluyendo:
  - **Análisis Exploratorio de Datos.**
  - **Ingeniería de Características - Data Wrangling.**
  - **Entrenamiento, validación, evaluación y selección del modelo usando `mlflow`:**
    - Realice **tracking de experimentos** con al menos **dos modelos diferentes**.
    - Implemente **hyperparameter tuning** para cada modelo.
    - Asigne nombres a los experimentos siguiendo el formato:  
      `nombre-nombre-del-algoritmo` (Ejemplo: `cristian-zapata-xgboost`).
    - Registre el modelo con mejor desempeño como **Champion** y el segundo mejor como **Challenger** en el Model Registry.
    - Cree un nuevo modelo en el Model Registry con un nombre siguiendo el formato:  
      `nombre-modelo` (Ejemplo: `cristian-modelo`).
    - **Orquestación del flujo de entrenamiento:**
      - Use un **cuaderno Jupyter** para documentar sus experimentos iniciales y hacer el setting con el servidor de `mlflow`.
      - Cree un **script con Prefect** que orqueste el flujo de entrenamiento y registre los modelos.
      - El flujo debe ser ejecutable en local, realizar el tracking en `mlflow`, y asegurar el registro de los modelos.
  - **Microservicio (API) para servir el modelo:**
    - Utilice `fastapi` (o cualquier framework con el que se sienta cómodo trabajando) para crear una API y servir el modelo previamente entrenado.
    - La API debe usar el modelo **Champion** para realizar inferencias.
  -  **Frontend:**
    - Cree una interfaz de usuario sencilla para conectarla con la API del modelo y poder servir las inferencias
    - Utilice `streamlit` o cualquier otro framework con el que se sienta cómodo trabajando
  - **Creación de contenedores**
    - Utilice un archivo `docker-compose.yaml` para ejecutar los contenedores de la API y el frontend.

- **Estructura del Proyecto:**
  - Diseñe una estructura clara y lógica para los archivos y carpetas del proyecto.  
  - Asegúrese de incluir los archivos esenciales:  
    - `requirements.txt`
    - `Dockerfile`
    - `docker-compose.yaml`
    - Experimentos y flujo de orquestación
    - `ExamenFinal.ipynb` (en la raíz del repositorio).
    
- **Formato:**
  - La presentación del examen debe tener un formato adecuado, con tamaños de letra, colores y etiquetas apropiados.
  - No se responderán dudas por parte del profesor o cualquier otro docente.

- **Entregables:**
  - Haga `commit` y `push` regularmente en su rama correspondiente.
  - Incluya interpretaciones y conclusiones de los resultados obtenidos.

## Rúbrica de Evaluación

| Aspecto a Evaluar                                                                                                           | Porcentaje |
|-----------------------------------------------------------------------------------------------------------------------------|------------|
| **Repositorio de Github/Dagshub debidamente configurado y debidamente estructurado con commits explícitos y profesionales** | 10         |
| **Análisis Exploratorio de Datos**                                                                                          | 5%         |
| **Ingeniería de Características - Data Wrangling**                                                                          | 5%         |
| **Entrenamiento de Modelos con mlflow**                                                                                     |            |
| - Tracking de experimentos con al menos dos modelos con tuning de hyper-parametros                                          | 10%        |
| - Registro del modelo Champion y Challenger en el Model Registry                                                            | 10%        |
| - Creación de script con Prefect que orqueste y registre los modelos                                                        | 20%        |
| **Microservicio (API) para servir el modelo Champion**                                                                      | 10%        |
| **Frontend para conectar con la API**                                                                                       | 10%        |
| **Creación de Contenedores**                                                                                                | 10%        |
| **Conclusiones**                                                                                                            | 10%        |

