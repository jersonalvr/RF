import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configuración de Streamlit
st.title('Modelo de Random Forest para Predicción de Kilos de Pesca')
st.write("""
Esta aplicación permite construir y evaluar un modelo de Random Forest para predecir el volumen de captura de pesca basado en diversas características.
""")

@st.cache
def cargar_datos():
    try:
        data = pd.read_excel('data.xlsx', engine='openpyxl')
        data['Inicio_Faena'] = pd.to_datetime(data['Inicio_Faena'], format='%d %m %Y %H:%M')
        data['Hora_Día'] = data['Inicio_Faena'].dt.hour
        return data
    except Exception as e:
        st.error(f"Error al leer el archivo Excel: {e}")
        return None

def procesar_datos(data, especie_especifica):
    data_especie = data[data['Especie'] == especie_especifica]
    data_especie = pd.get_dummies(data_especie, columns=['Aparejo', 'Origen', 'Motor', 'HP'])
    x = data_especie.drop(columns=['Especie', 'Kilos', 'Talla', 'Precio_Kilo'])
    y = data_especie['Kilos']

    # Convertir a numérico y manejar valores nulos
    x = x.apply(pd.to_numeric, errors='coerce')
    x = x.fillna(0)
    y = y.fillna(0)

    return x, y

def entrenar_modelo(x_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    return rf

def mostrar_importancia_caracteristicas(rf, x):
    importancia_caracteristicas = pd.Series(rf.feature_importances_, index=x.columns)
    fig, ax = plt.subplots()
    sns.barplot(x=importancia_caracteristicas.nlargest(10).values, y=importancia_caracteristicas.nlargest(10).index, ax=ax)
    ax.set_xlabel('Importancia de la característica')
    ax.set_ylabel('Característica')
    ax.set_title('Importancia de las características en el modelo de Random Forest')
    st.pyplot(fig)

# Cargar datos
data = cargar_datos()
if data is not None:
    st.write(data.head())

    # Seleccionar la especie
    especie_especifica = st.selectbox("Seleccionar la especie", data['Especie'].unique())

    # Procesar datos
    x, y = procesar_datos(data, especie_especifica)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    rf = entrenar_modelo(x_train, y_train)

    # Predecir y evaluar el modelo
    y_pred = rf.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Error cuadrático medio: {mse}')

    # Mostrar importancia de las características
    mostrar_importancia_caracteristicas(rf, x)
