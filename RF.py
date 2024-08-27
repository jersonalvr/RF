import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

fig, ax = plt.subplots()
df['Millas_Recorridas'].hist(bins=20, ax=ax)
ax.set_title('Distribución de Millas Recorridas')
ax.set_xlabel('Millas Recorridas')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Especie', y='Precio_Kg', data=df, ax=ax)
ax.set_title('Distribución de Precio por Kg por Especie')
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

fig, ax = plt.subplots()
ax.scatter(df['Millas_Recorridas'], df['Ganancia'])
ax.set_title('Millas Recorridas vs Ganancia')
ax.set_xlabel('Millas Recorridas')
ax.set_ylabel('Ganancia')
st.pyplot(fig)

# Agrupar las ganancias por fecha de faena
df_agrupado = df.groupby('Inicio_Faena')['Ganancia'].sum()

# Graficar la distribución de ganancias por fecha de faena
st.subheader('Distribución de ganancias por Fecha de Faena')
fig, ax = plt.subplots(figsize=(12, 7))
df_agrupado.plot(ax=ax, legend=True)
ax.set_title('Distribución de ganancias por Fecha de Faena')
ax.set_xlabel('Inicio_Faena')
ax.set_ylabel('Ganancia')
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
selected_columns = ['Caballos_Motor', 'Millas_Recorridas', 'Volumen_Kg', 'Precio_Kg', 'Talla_cm', 'Venta', 'Costo_Combustible', 'Ganancia', 'Tripulantes', 'Hora_Float']
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

# Seleccionar la opción (especie o embarcación)
opcion = st.selectbox("Seleccionar el enfoque", ["Embarcación", "Especie"], key="enfoque_selectbox")

# Clave única para cada selección
if opcion == "Embarcación":
    seleccion = st.selectbox("Seleccionar la embarcación", df_normalized['Embarcacion'].unique(), key="embarcacion_selectbox")
else:
    seleccion = st.selectbox("Seleccionar la especie", df_normalized['Especie'].unique(), key="especie_selectbox")

def procesar_datos(df, seleccion, es_embarcacion=True):
    if es_embarcacion:
        df_seleccion = df[df['Embarcacion'] == seleccion]
    else:
        df_seleccion = df[df['Especie'] == seleccion]
    
    if df_seleccion.empty:
        st.error(f"No se encontraron datos para la {'embarcación' if es_embarcacion else 'especie'}: {seleccion}")
        return None, None
    
    df_seleccion = pd.get_dummies(df_seleccion, columns=['Marca_Motor', 'Modelo_Motor', 'Aparejo'], drop_first=True)
    X = df_seleccion.drop(columns=['Embarcacion', 'Especie', 'Volumen_Kg', 'Talla_cm', 'Precio_Kg', 'Venta', 'Ganancia'])
    y = df_seleccion['Volumen_Kg'].fillna(0)
    
    return X.apply(pd.to_numeric, errors='coerce').fillna(0), y

def entrenar_modelo_con_curvas(X_train, y_train, X_val, y_val, n_estimators=100):
    train_errors = []
    val_errors = []
    modelo = RandomForestRegressor(warm_start=True, random_state=42)
    
    for i in range(1, n_estimators + 1):
        modelo.set_params(n_estimators=i)
        modelo.fit(X_train, y_train)
        
        train_errors.append(mean_squared_error(y_train, modelo.predict(X_train)))
        val_errors.append(mean_squared_error(y_val, modelo.predict(X_val)))
    
    return modelo, train_errors, val_errors

# Procesar los datos
X, y = procesar_datos(df_normalized, seleccion, es_embarcacion=(opcion == "Embarcación"))

if X is not None and y is not None:
    if len(X) > 1:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo_rf, train_errors, val_errors = entrenar_modelo_con_curvas(X_train, y_train, X_val, y_val)

        # Curvas de entrenamiento y validación
        st.subheader(f'Curvas de Entrenamiento y Validación - {seleccion} ({opcion})')
        st.markdown("""
        Estas curvas muestran cómo de bien nuestro modelo está aprendiendo a predecir el volumen de captura. Si el error de validación es cercano al error de entrenamiento, significa que el modelo es bastante preciso y no se está sobreajustando a los datos de entrenamiento.
        """)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(train_errors) + 1), train_errors, label='Error de Entrenamiento')
        ax.plot(range(1, len(val_errors) + 1), val_errors, label='Error de Validación')
        ax.set_xlabel('Número de Árboles')
        ax.set_ylabel('Error Cuadrático Medio')
        ax.legend()
        st.pyplot(fig)

        # Importancia de características
        st.subheader(f'Importancia de Características - {seleccion} ({opcion})')
        st.markdown("""
        La importancia de características nos ayuda a entender cuáles variables son más influyentes en la predicción del volumen de captura. Estas son como los ingredientes principales de una receta, donde algunos tienen un mayor impacto en el resultado final.
        """)
        importances = modelo_rf.feature_importances_
        indices = X_train.columns
        feature_importances = pd.Series(importances, index=indices).sort_values(ascending=False)
        
        if 'Ganancia' in feature_importances.index:
            feature_importances = feature_importances.drop('Ganancia')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importances.plot(kind='bar', ax=ax)
        ax.set_title(f'Importancia de Características - {seleccion} ({opcion})')
        ax.set_ylabel('Importancia')
        st.pyplot(fig)

        # Valores reales vs predichos
        st.subheader(f'Valores Reales vs Predichos - {seleccion} ({opcion})')
        st.markdown("""
        Este gráfico compara nuestras predicciones con los valores reales observados. Si los puntos se alinean bien con la línea diagonal, significa que nuestro modelo está haciendo un buen trabajo prediciendo el volumen de captura.
        """)
        y_val_pred = modelo_rf.predict(X_val)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_val, y_val_pred, alpha=0.5)
        ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        ax.set_xlabel('Valores Reales')
        ax.set_ylabel('Valores Predichos')
        ax.set_title(f'Valores Reales vs Predichos - {seleccion} ({opcion})')
        st.pyplot(fig)

        # Mostrar métricas del modelo
        st.subheader('Métricas del Modelo')
        st.markdown("""
        Aquí se muestran algunas métricas clave que nos indican cuán bien está funcionando nuestro modelo:
        - **MSE (Error Cuadrático Medio):** Indica qué tan lejos están, en promedio, nuestras predicciones de los valores reales.
        - **MAE (Error Absoluto Medio):** Muestra el promedio de las diferencias absolutas entre las predicciones y los valores reales.
        - **R2 (Coeficiente de Determinación):** Nos dice qué tan bien las variables explican la variabilidad del resultado.
        """)
        mse = mean_squared_error(y_val, y_val_pred)
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        
        st.write(f"MSE (Error Cuadrático Medio): {mse:.4f}")
        st.write(f"MAE (Error Absoluto Medio): {mae:.4f}")
        st.write(f"R2 (Coeficiente de Determinación): {r2:.4f}")
        
    else:
        st.error("No hay suficientes datos para dividir en conjuntos de entrenamiento y validación.")
