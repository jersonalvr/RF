import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Título y descripción de la aplicación
st.title('Modelo de Random Forest para Pesca Artesanal en Coishco')
st.write("""
Esta aplicación permite visualizar la importancia de las características en un modelo de Random Forest para la predicción del volumen de captura.
""")

# Cargar los datos
df = pd.read_excel('data.xlsx')

# Convertir la columna 'Inicio_Faena' a datetime
df['Inicio_Faena'] = pd.to_datetime(df['Inicio_Faena'], format='%d %m %Y %H:%M')

# Verificar la estructura de los datos
st.write("### Vista previa de los datos")
st.write(df.head())

# Agrupar por especie y aparejo, sumando los kilos
df_agrupado = df.groupby(['Especie', 'Aparejo'])['Volumen_Kg'].sum().unstack()

# Graficar la captura total por especie
st.subheader('Captura total por especie')
fig, ax = plt.subplots(figsize=(12, 7))
df_agrupado.plot(kind='bar', ax=ax, color='skyblue')
ax.set_title('Captura total por especie')
ax.set_xlabel('Especie')
ax.set_ylabel('Kilos')
plt.xticks(rotation=45)
st.pyplot(fig)

# Graficar en escala logarítmica
st.subheader('Captura total por especie (Escala Logarítmica)')
fig, ax = plt.subplots(figsize=(12, 7))
df_agrupado.plot(kind='bar', ax=ax, color='skyblue', logy=True)
ax.set_title('Captura total por especie (Escala Logarítmica)')
ax.set_xlabel('Especie')
ax.set_ylabel('Kilos')
plt.xticks(rotation=45)
st.pyplot(fig)

# Agrupar por Marca de Motor y Caballos de fuerza, sumando los kilos
df_agrupado = df.groupby(['Marca_Motor', 'Caballos_Motor'])['Volumen_Kg'].sum().unstack()

# Graficar captura total por Motor y Caballos de fuerza
st.subheader('Captura total por Motor y Caballos de fuerza')
fig, ax = plt.subplots(figsize=(12, 7))
df_agrupado.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
ax.set_title('Captura total por Motor y Caballos de fuerza')
ax.set_xlabel('Motor')
ax.set_ylabel('Kilos')
plt.xticks(rotation=45)
st.pyplot(fig)

# Graficar en escala logarítmica
st.subheader('Captura total por Motor y Caballos de fuerza (Escala Logarítmica)')
fig, ax = plt.subplots(figsize=(12, 7))
df_agrupado.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', logy=True)
ax.set_title('Captura total por Motor y Caballos de fuerza (Escala Logarítmica)')
ax.set_xlabel('Motor')
ax.set_ylabel('Kilos')
plt.xticks(rotation=45)
st.pyplot(fig)

# Agrupar las ventas por fecha de faena
df_agrupado = df.groupby('Inicio_Faena')['Venta'].sum()

# Graficar la distribución de ventas por fecha de faena
st.subheader('Distribución de Ventas por Fecha de Faena')
fig, ax = plt.subplots(figsize=(12, 7))
df_agrupado.plot(ax=ax, legend=True)
ax.set_title('Distribución de Ventas por Fecha de Faena')
ax.set_xlabel('Inicio_Faena')
ax.set_ylabel('Ventas')
plt.xticks(rotation=45)
st.pyplot(fig)

# Crear un nuevo DataFrame y agregar la columna Hora_Flotante
df_horas = df.drop(columns=['Inicio_Faena'])
df_horas['Hora'] = df['Inicio_Faena'].dt.time
df_horas['Hora_Flotante'] = df_horas['Hora'].apply(lambda x: x.hour + x.minute/60)

# Graficar la distribución de las ventas por hora del día
st.subheader('Distribución de las Ventas por Hora del Día')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_horas, x='Hora_Flotante', weights='Venta', bins=24, kde=True)
ax.set_xlabel('Hora del Día')
ax.set_ylabel('Distribución de las Ventas')
ax.set_title('Distribución de las Ventas por Hora del Día')
st.pyplot(fig)

# Calcular y graficar la matriz de correlación
st.subheader('Matriz de Correlación')
selected_columns = ['Hora_Flotante', 'Horas_Faena', 'Volumen_Kg', 'Talla_cm', 'Precio_Kg', 'Venta', 'Caballos_Motor', 'Tripulantes']
correlation_matrix = df_horas[selected_columns].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=correlation_matrix, annot=True, mask=np.triu(np.ones_like(correlation_matrix, dtype=bool)),
            cmap='magma', center=0, linewidths=0.3, annot_kws={"fontsize":12}, cbar_kws={"shrink": .7})
plt.title("Correlación entre variables", fontsize=17, y=1.02)
plt.tight_layout()
st.pyplot(fig)

# Descargar la matriz de correlación
st.subheader('Descargar Matriz de Correlación')
csv = correlation_matrix.to_csv(index=True)
st.download_button(label="Descargar Matriz de Correlación como CSV",
                   data=csv,
                   file_name='matriz_correlacion.csv',
                   mime='text/csv')

# Normalizar los datos
scaler = MinMaxScaler()
columns_to_normalize = [
    'Horas_Faena', 'Volumen_Kg', 'Talla_cm',
    'Precio_Kg', 'Venta', 'Caballos_Motor', 
    'Tripulantes', 'Hora_Flotante'
]
df_horas_normalizado = pd.DataFrame(scaler.fit_transform(df_horas[columns_to_normalize]), columns=columns_to_normalize)

# Mostrar las primeras filas del DataFrame normalizado
st.write("Primeras filas del DataFrame normalizado:")
st.write(df_horas_normalizado.head())

# Definir las variables independientes y dependientes
X = df_horas_normalizado[['Horas_Faena', 'Volumen_Kg', 'Talla_cm', 'Precio_Kg', 'Venta', 'Caballos_Motor', 'Tripulantes', 'Hora_Flotante']]
X = pd.get_dummies(X, drop_first=True)
y = df_horas_normalizado['Venta']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones y evaluar el modelo
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.write(f"Coeficiente de determinación (R^2): {r2}")
st.write(f"Error cuadrático medio (MSE): {mse}")

# Visualización de la importancia de las características
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

st.subheader('Importancia de las Características - Random Forest')
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
ax.set_title('Importancia de las Características - Random Forest')
ax.set_xlabel('Importancia')
ax.set_ylabel('Características')
st.pyplot(fig)

# Visualización de los resultados
st.subheader('Predicción vs Realidad')
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([0, 1], [0, 1], color='red', linewidth=2)
ax.set_xlabel('Valores Reales de Venta')
ax.set_ylabel('Valores Predichos de Venta')
ax.set_title('Random Forest - Predicción de Venta')
st.pyplot(fig)
