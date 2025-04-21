from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib # type: ignore
import numpy as np

app = FastAPI()

# Cargar modelo entrenado
model = joblib.load("models/model.pkl")


class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20, description="Número de embarazos")
    Glucose: float = Field(..., ge=1, le=300, description="Nivel de glucosa")
    BloodPressure: float = Field(..., ge=1, le=200, description="Presión arterial")
    SkinThickness: float = Field(..., ge=0, le=100, description="Espesor de piel")
    Insulin: float = Field(..., ge=0, le=1000, description="Nivel de insulina")
    BMI: float = Field(..., ge=0, le=80, description="Índice de masa corporal")
    DiabetesPedigreeFunction: float = Field(
        ..., ge=0.0, le=2.5, description="Historial familiar"
    )
    Age: int = Field(..., ge=0, le=120, description="Edad en años")


@app.post("/predict")
def predict(data: DiabetesInput):
    input_data = np.array(
        [
            [
                data.Pregnancies,
                data.Glucose,
                data.BloodPressure,
                data.SkinThickness,
                data.Insulin,
                data.BMI,
                data.DiabetesPedigreeFunction,
                data.Age,
            ]
        ]
    )
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
