import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Título y descripción de la aplicación
st.title('Modelo de Random Forest para Pesca Artesanal en Coishco')
st.write("""
Esta aplicación permite visualizar la importancia de las características en un modelo de Random Forest para la predicción del volumen de captura.
""")

# Cargar los datos
df = pd.read_excel('data.xlsx')

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

# Convertir la columna 'Inicio_Faena' a datetime
df['Inicio_Faena'] = pd.to_datetime(df['Inicio_Faena'], format='%d %m %Y %H:%M')

# Agregar una nueva columna 'Hora' con solo la hora de 'Inicio_Faena'
df['Hora'] = df['Inicio_Faena'].dt.time

# Transformar la columna 'Hora' en un valor flotante (hora + minutos/60)
df['Hora_Float'] = df['Hora'].apply(lambda x: x.hour + x.minute/60)

# Crear un nuevo DataFrame eliminando las columnas 'Inicio_Faena' y 'Hora'
df_ = df.drop(columns=['Inicio_Faena', 'Hora'])

# Graficar la distribución de las ventas por hora del día
st.subheader('Distribución de las Ventas por Hora del Día')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_, x='Hora_Float', weights='Venta', bins=24, kde=True)
ax.set_xlabel('Hora del Día')
ax.set_ylabel('Distribución de las Ventas')
ax.set_title('Distribución de las Ventas por Hora del Día')
st.pyplot(fig)

# Seleccionamos las columnas numéricas
numeric_columns = df_.select_dtypes(include=['int64', 'float64']).columns

# Crear el escalador
scaler = MinMaxScaler()

# Aplicar la normalización
df_normalized = df_.copy()
df_normalized[numeric_columns] = scaler.fit_transform(df_[numeric_columns])

st.write("### Datos normalizados")
st.write(df_normalized.head())

# Calcular y graficar la matriz de correlación
st.subheader('Matriz de Correlación')
selected_columns = ['Hora_Float', 'Horas_Faena', 'Volumen_Kg', 'Talla_cm', 'Precio_Kg', 'Venta', 'Caballos_Motor', 'Tripulantes']
correlation_matrix = df_normalized[selected_columns].corr()

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

# Seleccionar la especie
especie_especifica = st.selectbox("Seleccionar la especie", df_normalized['Especie'].unique())

# Procesar los datos del DataFrame
def procesar_datos(df, especie):
    df_especie = df[df['Especie'] == especie]
    df_especie = pd.get_dummies(df_especie, columns=['Aparejo', 'Origen', 'Marca_Motor', 'Modelo_Motor'], drop_first=True)
    x = df_especie.drop(columns=['Especie', 'Volumen_Kg', 'Talla_cm', 'Precio_Kg', 'Venta'])  # Se excluye 'Venta'
    y = df_especie['Volumen_Kg']
    return x.apply(pd.to_numeric, errors='coerce').fillna(0), y.fillna(0)

# Entrenar el modelo de Random Forest
def entrenar_modelo(x_train, y_train):
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(x_train, y_train)
    return modelo

# Mostrar la importancia de las características
def mostrar_importancia_caracteristicas(modelo, x):
    importancia = pd.Series(modelo.feature_importances_, index=x.columns)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importancia.nlargest(10).values, y=importancia.nlargest(10).index)
    plt.xlabel('Importancia de la característica')
    plt.ylabel('Característica')
    plt.title('Importancia de las características en el modelo de Random Forest')
    st.pyplot(plt.gcf())

# Mostrar gráfico de valores reales vs. valores predichos
def mostrar_valores_vs_predicciones(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Línea de referencia
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.title('Valores Reales vs. Valores Predichos')
    plt.tight_layout()  # Ajustar diseño para evitar recortes
    st.pyplot(plt)

# Procesar datos y entrenar modelo
x, y = procesar_datos(df_normalized, especie_especifica)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
modelo_rf = entrenar_modelo(x_train, y_train)

# Evaluar modelo
y_pred = modelo_rf.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f'Error cuadrático medio: {mse:.4f}')

# Mostrar importancia de las características
mostrar_importancia_caracteristicas(modelo_rf, x)

# Mostrar gráfico de valores reales vs. valores predichos
mostrar_valores_vs_predicciones(y_test, y_pred)
