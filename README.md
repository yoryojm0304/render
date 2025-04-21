# Proyecto_final
#  Proyecto Final - Predicción de Diabetes (MLOps)
Participantes:

David Mora Vargas: Software Engineer

Esteban Gutiérrez Saborío: Data Scientist

Jorge Arturo Jiménez Madrigal: MLops Engineer

Este proyecto implementa un sistema de predicción de diabetes utilizando un modelo de machine learning. Está diseñado siguiendo principios de MLOps con enfoque en despliegue, automatización, validación y versionamiento.

---
##  Objetivos del proyecto

- Aplicar un flujo completo de MLOps (entrenamiento, despliegue, versionamiento y validación).
- Exponer el modelo a través de una API con FastAPI.
- Contenerizar la aplicación con Docker.
- Automatizar análisis estático del código con DeepSource.
- Facilitar el futuro reentrenamiento y despliegue continuo del modelo.

 Cómo funciona internamente
El archivo train_model.py entrena un modelo RandomForestClassifier con datos del dataset diabetes.csv
El modelo es guardado como model.pkl dentro de la carpeta /models
La API FastAPI carga este modelo al iniciar (joblib.load)
Al llamar al endpoint /predict, los datos se transforman a numpy.array y se pasan al modelo
La predicción se devuelve al usuario

##  Tecnologías utilizadas

- Python 3.10
- FastAPI (API REST)
- Scikit-learn (modelo)
- Docker (contenedorización)
- GitHub Actions (CI/CD)
- DeepSource (análisis estático)
- DVC (pendiente)
- AWS S3 / EC2 (pendiente)

---

##  Cómo ejecutar la API

###  Opción 1: Local (entorno virtual)

```bash
# Clonar repositorio
git clone https://github.com/DavidMoraV/Proyecto_final.git
cd Proyecto_final

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar API
uvicorn api.main:app --reload

