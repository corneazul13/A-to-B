import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Configuración inicial
np.random.seed(42)

# 1. Generar el dataset y guardarlo en un archivo CSV
estaciones = ['A', 'B', 'C', 'D', 'E', 'F']
clima = ['soleado', 'lluvioso', 'nublado']
horario = ['mañana', 'tarde', 'noche']
dia_semana = ['laborable', 'fin de semana']

data = {
    'estacion': np.random.choice(estaciones, 100),
    'hora_del_dia': np.random.choice(horario, 100),
    'dia_semana': np.random.choice(dia_semana, 100),
    'numero_pasajeros': np.random.randint(50, 500, size=100),
    'clima': np.random.choice(clima, 100)
}

df = pd.DataFrame(data)
df.to_csv('datos_transporte_no_supervisado.csv', index=False)

# 2. Cargar el archivo CSV
df = pd.read_csv('datos_transporte_no_supervisado.csv')

# 3. Preprocesamiento de datos
# Codificación de variables categóricas
label_enc = LabelEncoder()
df['estacion'] = label_enc.fit_transform(df['estacion'])
df['hora_del_dia'] = label_enc.fit_transform(df['hora_del_dia'])
df['dia_semana'] = label_enc.fit_transform(df['dia_semana'])
df['clima'] = label_enc.fit_transform(df['clima'])

# 4. Aplicar el modelo de K-means para el aprendizaje no supervisado
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['estacion', 'hora_del_dia', 'dia_semana', 'numero_pasajeros', 'clima']])

# 5. Visualización del resultado
# Reducción de la dimensionalidad con PCA para visualizar los clusters en 2D
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[['estacion', 'hora_del_dia', 'dia_semana', 'numero_pasajeros', 'clima']])

plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['cluster'], cmap='viridis', marker='o')
plt.title('Clustering de estaciones de transporte')
plt.xlabel('PCA Componente 1')
plt.ylabel('PCA Componente 2')
plt.colorbar(label='Cluster')
plt.show()

# 6. Guardar el DataFrame con los clusters en un nuevo archivo CSV
df.to_csv('datos_transporte_clusterizados.csv', index=False)

# 7. Resumen de los resultados
print("Número de pasajeros promedio por cluster:")
print(df.groupby('cluster')['numero_pasajeros'].mean())

print("\nCantidad de muestras por cluster:")
print(df['cluster'].value_counts())