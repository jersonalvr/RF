import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import folium_static
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# Título y descripción de la aplicación
st.title('Modelo de Random Forest para Pesca Artesanal en Coishco')
st.write("""
Esta aplicación permite visualizar la importancia de las características en un modelo de Random Forest para la predicción del volumen de captura.
""")

# Cargar los datos
df = pd.read_excel('data.xlsx')

st.write("### Vista previa de los datos")
st.write(df.head())

st.write("""
Comencemos visualizando algunos gráficos estadisticos referentes a la actividad pesquera de la zona
""")

# Agrupar por especie y aparejo, sumando los kilos
df_agrupado_kilos = df.groupby(['Especie', 'Aparejo'])['Volumen_Kg'].sum().unstack()

# Ordenar los datos por la suma total de kilos para cada especie (de menor a mayor)
df_agrupado_kilos = df_agrupado_kilos.loc[df_agrupado_kilos.sum(axis=1).sort_values().index]

# Seleccionar el tipo de gráfico para los kilos
opcion_kilos = st.radio("Selecciona el tipo de gráfico para visualizar la distribución de la captura", ('Escala Normal', 'Escala Logarítmica'), key='kilos')

# Crear el gráfico interactivo con Plotly
if opcion_kilos == 'Escala Normal':
    st.subheader('Captura total por especie')
    fig = px.bar(df_agrupado_kilos, 
                 title='Captura total por especie',
                 labels={'value': 'Kilos', 'index': 'Especie'},
                 text_auto=True)
    fig.update_layout(yaxis_title='Kilos', xaxis_title='Especie', xaxis_tickangle=-45)
else:
    st.subheader('Captura total por especie (Escala Logarítmica)')
    fig = px.bar(df_agrupado_kilos, 
                 title='Captura total por especie (Escala Logarítmica)',
                 labels={'value': 'Kilos', 'index': 'Especie'},
                 log_y=True,
                 text_auto=True)
    fig.update_layout(yaxis_title='Kilos (Logarítmico)', xaxis_title='Especie', xaxis_tickangle=-45)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Agrupar los datos por 'Especie' y sumar las ganancias
ventas_por_especie = df.groupby('Especie')['Ganancia'].sum().sort_values()

# Convertir el resultado en un DataFrame para Plotly
df_ventas = ventas_por_especie.reset_index()
df_ventas.columns = ['Especie', 'Ganancia']

# Seleccionar el tipo de gráfico para las ganancias
opcion_ganancia = st.radio("Selecciona el tipo de gráfico para visualizar las ganancias según la especie", ('Escala Normal', 'Escala Logarítmica'), key='escala_ganancia')

# Crear el gráfico interactivo con Plotly
if opcion_ganancia == 'Escala Normal':
    fig = px.bar(df_ventas, 
                 x='Especie', 
                 y='Ganancia', 
                 title='Ganancia por Especie (Escala Normal)', 
                 labels={'Ganancia': 'Ganancia (Suma Total)', 'Especie': 'Especie'},
                 color='Ganancia',
                 text='Ganancia')
    fig.update_layout(xaxis_title='Especie', yaxis_title='Ganancia (Suma Total)', xaxis_tickangle=-45)
else:
    fig = px.bar(df_ventas, 
                 x='Especie', 
                 y='Ganancia', 
                 title='Ganancia por Especie (Escala Logarítmica)', 
                 labels={'Ganancia': 'Ganancia (Suma Total)', 'Especie': 'Especie'},
                 color='Ganancia',
                 text='Ganancia',
                 log_y=True)
    fig.update_layout(xaxis_title='Especie', yaxis_title='Ganancia (Suma Total) (Logarítmica)', xaxis_tickangle=-45)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Selección de la especie
especie_seleccionada = st.selectbox('Selecciona la especie', df['Especie'].unique())

# Filtrar los datos según la especie seleccionada
df_filtrado = df[df['Especie'] == especie_seleccionada]

# Crear el gráfico de barras utilizando matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Crear el mapa centrado en la ubicación media
mapa = folium.Map(location=[df_filtrado['Origen_Latitud'].mean(), df_filtrado['Origen_Longuitud'].mean()], zoom_start=6)

# Añadir marcadores al mapa
for idx, row in df_filtrado.iterrows():
    folium.Marker(
        location=[row['Origen_Latitud'], row['Origen_Longuitud']],
        popup=row['Origen']
    ).add_to(mapa)

# Mostrar el mapa en Streamlit
folium_static(mapa)

# Título de la aplicación
st.title('Distribución de Precio por Kg ponderada por Volumen en Kg')

# Crear el histograma ponderado
fig = px.histogram(df, x='Precio_Kg', y='Volumen_Kg', 
                   histfunc='sum', 
                   nbins=24, 
                   title='Distribución de Precio por Kg ponderada por Volumen en Kg')

# Actualizar etiquetas del gráfico
fig.update_layout(xaxis_title='Precio por Kg', 
                  yaxis_title='Volumen en Kg', 
                  bargap=0.1)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Título de la aplicación
st.title('Distribución de Precio por Kg ponderada por Talla en cm')

# Crear el histograma ponderado
fig = px.histogram(df, x='Talla_cm', y='Volumen_Kg', 
                   histfunc='sum', 
                   nbins=24, 
                   title='Distribución de Precio por Kg ponderada por Talla en cm')

# Actualizar etiquetas del gráfico
fig.update_layout(xaxis_title='Talla en cm', 
                  yaxis_title='Volumen en Kg', 
                  bargap=0.1)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Título de la aplicación
st.title('Distribución de Precio por Kg ponderada por Millas Recorridas')

# Crear el histograma ponderado
fig = px.histogram(df, x='Millas_Recorridas', y='Volumen_Kg', 
                   histfunc='sum', 
                   nbins=24, 
                   title='Distribución de Precio por Kg ponderada por Millas Recorridas')

# Actualizar etiquetas del gráfico
fig.update_layout(xaxis_title='Millas Recorridas', 
                  yaxis_title='Volumen en Kg', 
                  bargap=0.1)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Agrupar por Marca de Motor y Caballos de fuerza, sumando los kilos
df_agrupado = df.groupby(['Marca_Motor', 'Caballos_Motor'])['Volumen_Kg'].sum().unstack()

# Convertir el DataFrame a formato largo para Plotly
df_agrupado_long = df_agrupado.reset_index().melt(id_vars='Marca_Motor', var_name='Caballos_Motor', value_name='Volumen_Kg')

# Crear el selector de gráficos
opcion = st.radio('Selecciona el tipo de gráfico para captura por caballos de motor:', ['Escala Normal', 'Escala Logarítmica'])

# Crear el gráfico interactivo con Plotly
if opcion == 'Escala Normal':
    st.subheader('Captura total por caballos de motor')
    fig = px.bar(df_agrupado_long, 
                 x='Marca_Motor', 
                 y='Volumen_Kg', 
                 color='Caballos_Motor', 
                 title='Captura total por Motor y Caballos de fuerza',
                 labels={'Marca_Motor': 'Motor', 'Volumen_Kg': 'Kilos'},
                 color_continuous_scale='viridis',
                 text='Volumen_Kg')
    fig.update_layout(xaxis_title='Motor', yaxis_title='Kilos', xaxis_tickangle=-45)
else:
    st.subheader('Captura total por caballos de motor (Escala Logarítmica)')
    fig = px.bar(df_agrupado_long, 
                 x='Marca_Motor', 
                 y='Volumen_Kg', 
                 color='Caballos_Motor', 
                 title='Captura total por caballos de motor (Escala Logarítmica)',
                 labels={'Marca_Motor': 'Motor', 'Volumen_Kg': 'Kilos'},
                 color_continuous_scale='viridis',
                 text='Volumen_Kg',
                 log_y=True)
    fig.update_layout(xaxis_title='Motor', yaxis_title='Kilos (Logarítmico)', xaxis_tickangle=-45)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Agrupar los datos
ventas_por_embarcacion = df.groupby('Embarcacion')['Ganancia'].sum()
millas_por_embarcacion = df.groupby('Embarcacion')['Millas_Recorridas'].sum()
volumen_por_embarcacion = df.groupby('Embarcacion')['Volumen_Kg'].sum()

# Crear un DataFrame combinado
datos_combinados = pd.DataFrame({
    'Ganancia': ventas_por_embarcacion,
    'Millas Recorridas': millas_por_embarcacion,
    'Volumen de Capturas': volumen_por_embarcacion
})

# Convertir el DataFrame combinado a formato largo para Plotly
datos_combinados_long = datos_combinados.reset_index().melt(id_vars='Embarcacion', var_name='Métrica', value_name='Valor')

# Crear botones para seleccionar el gráfico
opcion = st.radio('Selecciona el tipo de gráfico:', 
                  ['Ganancia por Embarcación', 
                   'Millas Recorridas por Embarcación', 
                   'Volumen de Capturas por Embarcación',
                   'Barras Apiladas: Ganancia, Millas, Volumen'])

# Mostrar el gráfico correspondiente
if opcion == 'Ganancia por Embarcación':
    st.subheader('Ganancia por Embarcación')
    fig = px.bar(ventas_por_embarcacion.reset_index(), 
                 x='Embarcacion', 
                 y='Ganancia', 
                 title='Ganancia por Embarcación',
                 labels={'Ganancia': 'Ganancia', 'Embarcacion': 'Embarcación'},
                 color='Ganancia',
                 text='Ganancia')
    fig.update_layout(xaxis_title='Embarcación', yaxis_title='Ganancia', xaxis_tickangle=-45)

elif opcion == 'Millas Recorridas por Embarcación':
    st.subheader('Millas Recorridas por Embarcación')
    fig = px.bar(millas_por_embarcacion.reset_index(), 
                 x='Embarcacion', 
                 y='Millas_Recorridas', 
                 title='Millas Recorridas por Embarcación',
                 labels={'Millas_Recorridas': 'Millas Recorridas', 'Embarcacion': 'Embarcación'},
                 color='Millas_Recorridas',
                 text='Millas_Recorridas')
    fig.update_layout(xaxis_title='Embarcación', yaxis_title='Millas Recorridas', xaxis_tickangle=-45)

elif opcion == 'Volumen de Capturas por Embarcación':
    st.subheader('Volumen de Capturas por Embarcación')
    fig = px.bar(volumen_por_embarcacion.reset_index(), 
                 x='Embarcacion', 
                 y='Volumen_Kg', 
                 title='Volumen de Capturas por Embarcación',
                 labels={'Volumen_Kg': 'Volumen (Kg)', 'Embarcacion': 'Embarcación'},
                 color='Volumen_Kg',
                 text='Volumen_Kg')
    fig.update_layout(xaxis_title='Embarcación', yaxis_title='Volumen (Kg)', xaxis_tickangle=-45)

elif opcion == 'Barras Apiladas: Ganancia, Millas, Volumen':
    st.subheader('Barras Apiladas: Ganancia, Millas, Volumen por Embarcación')
    fig = px.bar(datos_combinados_long, 
                 x='Embarcacion', 
                 y='Valor', 
                 color='Métrica', 
                 title='Barras Apiladas: Ganancia, Millas, Volumen por Embarcación',
                 labels={'Valor': 'Valores', 'Embarcacion': 'Embarcación'},
                 text='Valor')
    fig.update_layout(xaxis_title='Embarcación', yaxis_title='Valores', xaxis_tickangle=-45)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Agrupar las ganancias por fecha de faena
df_agrupado = df.groupby('Inicio_Faena')['Ganancia'].sum().reset_index()

# Crear el gráfico interactivo con Plotly
st.subheader('Distribución de ganancias por Fecha de Faena')
fig = px.line(df_agrupado, 
              x='Inicio_Faena', 
              y='Ganancia', 
              title='Distribución de Ganancias por Fecha de Faena',
              labels={'Inicio_Faena': 'Fecha de Faena', 'Ganancia': 'Ganancia'},
              markers=True)

# Personalizar el gráfico
fig.update_layout(xaxis_title='Fecha de Faena', yaxis_title='Ganancia', xaxis_tickformat='%Y-%m-%d')

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Agrupar las ganancias por fecha de faena
df_agrupado = df.groupby('Inicio_Faena')['Costo_Combustible'].sum().reset_index()

# Crear el gráfico interactivo con Plotly
st.subheader('Distribución de Costo Combustible por Fecha de Faena')
fig = px.line(df_agrupado, 
              x='Inicio_Faena', 
              y='Costo_Combustible', 
              title='Distribución de Costo Combustible por Fecha de Faena',
              labels={'Inicio_Faena': 'Fecha de Faena', 'Costo_Combustible': 'Costo_Combustible'},
              markers=True)

# Personalizar el gráfico
fig.update_layout(xaxis_title='Fecha de Faena', yaxis_title='Costo Combustible', xaxis_tickformat='%Y-%m-%d')

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Convertir la columna 'Inicio_Faena' y 'Fecha_Venta' a datetime
df['Inicio_Faena'] = pd.to_datetime(df['Inicio_Faena'], format='%d %m %Y %H:%M')
df['Inicio_Venta'] = pd.to_datetime(df['Inicio_Venta'], format='%d %m %Y %H:%M')

# Agregar una nueva columna 'Hora_Faena' y 'Hora_Venta' con solo la hora de 'Inicio_Faena' y 'Fecha_Venta'
df['Hora_Faena'] = df['Inicio_Faena'].dt.time
df['Hora_Venta'] = df['Inicio_Venta'].dt.time

# Transformar la columna 'Hora_Faena' y 'Hora_Venta' en un valor flotante (hora + minutos/60)
df['HFloat_Faena'] = df['Hora_Faena'].apply(lambda x: x.hour + x.minute/60)
df['HFloat_Venta'] = df['Hora_Venta'].apply(lambda x: x.hour + x.minute/60)

# Crear un nuevo DataFrame eliminando las columnas 'Inicio_Faena', 'Inicio_Venta', 'Hora_Faena' y 'Hora_Venta'
df_ = df.drop(columns=['Inicio_Faena', 'Inicio_Venta', 'Hora_Faena', 'Hora_Venta'])

# Crear una tabla de frecuencias ponderada
hist_data = df_.groupby('HFloat_Faena').apply(lambda x: (x['Volumen_Kg'] * len(x)).sum())
hist_data = hist_data.reset_index(name='Volumen_Total')

# Graficar la distribución de las faenas por hora del día
st.subheader('Distribución de las faenas por Hora del Día')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_, x='HFloat_Faena', weights='Volumen_Kg', bins=24, kde=True)
ax.set_xlabel('Hora del Día')
ax.set_ylabel('Distribución de las faenas')
ax.set_title('Distribución de las faenas por Hora del Día')
st.pyplot(fig)

# Graficar la distribución de las ventas por hora del día
st.subheader('Distribución de las Ventas por Hora del Día')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_, x='HFloat_Venta', weights='Venta', bins=24, kde=True)
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

# Seleccionar las columnas para la matriz de correlación
selected_columns = ['Caballos_Motor', 'Millas_Recorridas', 'Volumen_Kg', 'Precio_Kg', 
                    'Talla_cm', 'Venta', 'Costo_Combustible', 'Ganancia', 'Tripulantes', 
                    'HFloat_Faena', 'HFloat_Venta', 'Origen_Latitud', 'Origen_Longuitud']

# Calcular la matriz de correlación
correlation_matrix = df_normalized[selected_columns].corr()

# Crear el gráfico interactivo con Plotly
fig = px.imshow(correlation_matrix,
                labels={'x': 'Variables', 'y': 'Variables', 'color': 'Correlación'},
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale='Magma',
                aspect='auto')

# Personalizar el diseño del gráfico
fig.update_layout(
    title='Matriz de Correlación entre Variables',
    coloraxis_showscale=True,
    xaxis={'side': 'bottom'},
    yaxis={'side': 'left'}
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

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

# Mostrar la imagen si la opción es "Especie"
if opcion == "Especie":
    especie_seleccionada = seleccion
    ruta_imagen = f"resources/{especie_seleccionada}.png"
    
    try:
        st.image(ruta_imagen, caption=f"Especie: {especie_seleccionada}", use_column_width=True)
    except FileNotFoundError:
        st.error(f"No se encontró la imagen para la especie: {especie_seleccionada}")

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

        fig = go.Figure()

        # Añadir líneas para los errores de entrenamiento y validación
        fig.add_trace(go.Scatter(x=list(range(1, len(train_errors) + 1)), y=train_errors, mode='lines', name='Error de Entrenamiento'))
        fig.add_trace(go.Scatter(x=list(range(1, len(val_errors) + 1)), y=val_errors, mode='lines', name='Error de Validación'))

        fig.update_layout(
            xaxis_title='Número de Árboles',
            yaxis_title='Error Cuadrático Medio',
            title='Curvas de Entrenamiento y Validación',
            template='plotly_dark'
        )

        st.plotly_chart(fig)

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

        fig = go.Figure()

        # Añadir las barras de importancia
        fig.add_trace(go.Bar(x=feature_importances.index, y=feature_importances.values, marker_color='magenta'))

        fig.update_layout(
            xaxis_title='Características',
            yaxis_title='Importancia',
            title=f'Importancia de Características - {seleccion} ({opcion})',
            template='plotly_dark'
        )

        st.plotly_chart(fig)

        # Valores reales vs predichos
        st.subheader(f'Valores Reales vs Predichos - {seleccion} ({opcion})')
        st.markdown("""
        Este gráfico compara nuestras predicciones con los valores reales observados. Si los puntos se alinean bien con la línea diagonal, significa que nuestro modelo está haciendo un buen trabajo prediciendo el volumen de captura.
        """)

        y_val_pred = modelo_rf.predict(X_val)

        fig = go.Figure()

        # Añadir puntos para valores reales vs predichos
        fig.add_trace(go.Scatter(x=y_val, y=y_val_pred, mode='markers', name='Valores Reales vs Predichos', marker=dict(color='cyan', opacity=0.5)))

        # Añadir línea de referencia
        fig.add_trace(go.Scatter(x=[y_val.min(), y_val.max()], y=[y_val.min(), y_val.max()], mode='lines', name='Línea de Referencia', line=dict(color='red', dash='dash')))

        fig.update_layout(
            xaxis_title='Valores Reales',
            yaxis_title='Valores Predichos',
            title=f'Valores Reales vs Predichos - {seleccion} ({opcion})',
            template='plotly_dark'
        )

        st.plotly_chart(fig)

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
