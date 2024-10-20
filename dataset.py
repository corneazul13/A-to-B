import pandas as pd
import numpy as np

# Simular datos
np.random.seed(42)
estaciones = ['A', 'B', 'C', 'D', 'E', 'F']
clima = ['soleado', 'lluvioso', 'nublado']
horario = ['mañana', 'tarde', 'noche']
dia_semana = ['laborable', 'fin de semana']

# Generar el dataset
data = {
    'origen': np.random.choice(estaciones, 100),
    'destino': np.random.choice(estaciones, 100),
    'tiempo_viaje_min': np.random.randint(5, 60, size=100),
    'hora_del_dia': np.random.choice(horario, 100),
    'dia_semana': np.random.choice(dia_semana, 100),
    'clima': np.random.choice(clima, 100)
}

df = pd.DataFrame(data)
# Filtrar para que origen y destino no sean iguales
df = df[df['origen'] != df['destino']]

print(df.head())
# Guardar el dataset en un archivo CSV
df.to_csv('datos_transporte.csv', index=False)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib

# Cargar el dataset
df = pd.read_csv('datos_transporte.csv')

# Codificar las variables categóricas
label_enc = LabelEncoder()
df['origen'] = label_enc.fit_transform(df['origen'])
df['destino'] = label_enc.fit_transform(df['destino'])
df['hora_del_dia'] = label_enc.fit_transform(df['hora_del_dia'])
df['dia_semana'] = label_enc.fit_transform(df['dia_semana'])
df['clima'] = label_enc.fit_transform(df['clima'])

# Separar las características (X) y la variable objetivo (y)
X = df.drop('tiempo_viaje_min', axis=1)
y = df['tiempo_viaje_min']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Crear el modelo
modelo = DecisionTreeRegressor(random_state=42)

# Entrenar el modelo
modelo.fit(X_train, y_train)
# Predecir en el conjunto de prueba
predicciones = modelo.predict(X_test)

# Calcular el error medio absoluto
error = mean_absolute_error(y_test, predicciones)
print(f"Error Medio Absoluto: {error}")


# Guardar el modelo entrenado
joblib.dump(modelo, 'modelo_transporte.pkl')
