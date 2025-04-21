import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Cargar datos
df = pd.read_csv("mlops_diabetes/diabetes.csv", sep=";", encoding="latin1")

# 2. Renombrar columnas a nombres compatibles con el modelo
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

# 3. Separar variables predictoras y variable objetivo
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. Separar conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Crear y entrenar modelo de regresión logística
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# 6. Predecir
y_pred = modelo.predict(X_test)

# 7. Evaluación
accuracy = accuracy_score(y_test, y_pred)
print("🔍 Accuracy del modelo (Regresión Logística):", round(accuracy, 4))

print("\n📉 Matriz de confusión:")
matriz = confusion_matrix(y_test, y_pred)
print(matriz)

print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# 8. Visualizar matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# Después de evaluar el modelo (luego del reporte)
joblib.dump(modelo, "models/model.pkl")
print(" Modelo de Regresión Logística guardado como models/model.pkl")
