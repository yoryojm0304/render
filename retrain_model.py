import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ruta al CSV
ruta_csv = os.path.join("mlops_diabetes", "diabetes.csv")
df = pd.read_csv(ruta_csv, sep=";", engine="python", encoding="latin1")

# Si solo hay una columna, significa que los valores están juntos
if df.shape[1] == 1:
    df = df.iloc[:, 0].astype(str).str.split(";", expand=True)

# Nombres de columnas
df.columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# Convertir a numérico
df = df.apply(pd.to_numeric)

# Reemplazar ceros con NaN en columnas clave y luego llenar con media
columnas_invalidas = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[columnas_invalidas] = df[columnas_invalidas].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Preparar datos
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo (esto sobreescribe el existente si ya hay uno)
output_path = os.path.join("models", "model.pkl")
joblib.dump(model, output_path)

print(f"✅ Modelo reentrenado y guardado en: {output_path}")
