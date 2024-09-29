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
from sklearn.model_selection import learning_curve
import plotly.express as px
import plotly.graph_objects as go

# Título y descripción de la aplicación
st.title('Modelo de Random Forest para Pesca Artesanal en Coishco')
st.write("""
Esta aplicación permite visualizar la importancia de las características en un modelo de Random Forest para la predicción del volumen de captura.
""")

# Cargar los datos
df = pd.read_excel('data.xlsx')

# Función para categorizar horas en intervalos de 2 horas
def categorize_hour(hour):
    period = "A.M." if hour < 12 else "P.M."
    hour_12 = hour % 12
    hour_12 = 12 if hour_12 == 0 else hour_12
    start_hour = hour_12
    end_hour = (hour_12 + 2) % 12
    end_hour = 12 if end_hour == 0 else end_hour
    return f"{start_hour:02d} - {end_hour:02d} {period}"

# Aplicar la función para categorizar horas en 'Inicio_Faena' y 'Inicio_Venta'
df['Hora_Faena'] = df['Inicio_Faena'].dt.hour.apply(categorize_hour)

# Crear la columna 'Mes_Faena' y mapear los meses a abreviaturas
meses = {1: 'ENE', 2: 'FEB', 3: 'MAR', 4: 'ABR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AGO', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DIC'}
df['Mes_Faena'] = df['Inicio_Faena'].dt.month.map(meses)

# Crear rangos para 'Precio_Kg' y 'Talla_cm'
bins_precio = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
labels_precio = ["S/ (0 - 5)", "S/ (5 - 10)", "S/ (10 - 15)", "S/ (15 - 20)", "S/ (20 - 25)", "S/ (25 - 30)", "S/ (30 - 35)", "S/ (35 - 40)", "S/ (40 - 45)", "S/ (45 - 50)", "S/ (50 - 55)"]
df['Precio_Float'] = pd.cut(df['Precio_Kg'], bins=bins_precio, labels=labels_precio, right=False)

bins_talla = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
labels_talla = ["(10 - 20) cm", "(20 - 30) cm", "(30 - 40) cm", "(40 - 50) cm", "(50 - 60) cm", "(60 - 70) cm", "(70 - 80) cm", "(80 - 90) cm", "(90 - 100) cm", "(100 - 110) cm", "(110 - 120) cm", "(120 - 130) cm", "(130 - 140) cm", "(140 - 150) cm"]
df['Talla_Float'] = pd.cut(df['Talla_cm'], bins=bins_talla, labels=labels_talla, right=False)

# Vista previa de los datos con nuevas columnas
st.write("### Vista previa de los datos")
st.write(df.head())

# Descripción de las variables
st.write("### Descripción de las Variables")
st.markdown("""
- **Temperatura del Agua (°C):** Varía entre 15 y 30 °C.
- **Profundidad (m):** Varía entre 20 y 100 metros.
- **Salinidad (PSU):** Rango entre 30 y 35 PSU.
- **Velocidad del Viento (m/s):** Rango de 1 a 10 m/s.
- **Corriente Marina (m/s):** Entre 0.1 y 1.5 m/s.
- **Índice de Captura por Unidad de Esfuerzo (CPUE):** Calculado en función del volumen de captura, las millas recorridas y el número de tripulantes.
""")

# Descripción de los Cálculos
st.write("### Descripción de los Cálculos")

st.markdown("""
**1. Cálculo de las Millas Recorridas:**

Se utiliza la siguiente fórmula basada en la ley del coseno esférico para determinar la distancia entre el origen (lugar de pesca) y el destino (muelle):
""")
st.latex(r"""
\text{Millas\_Recorridas} = 2 \times 3958.8 \times \arcsin\left(\sqrt{\left(\sin\left(\frac{\text{radians}(\Delta \text{Lat})}{2}\right)\right)^2 + \cos(\text{radians}(\text{OR\_Lat})) \times \cos(\text{radians}(\text{Dest\_Lat})) \times \left(\sin\left(\frac{\text{radians}(\Delta \text{Lon})}{2}\right)\right)^2}\right)
""")
st.markdown("""
Donde:
""")
st.latex(r"""
\Delta \text{Lat} = \text{Latitud de Destino} - \text{Latitud de Origen}
""")
st.latex(r"""
\Delta \text{Lon} = \text{Longitud de Destino} - \text{Longitud de Origen}
""")

# Fórmula para el cálculo de los costos de combustible
st.markdown("""
**2. Cálculo de los Costos de Combustible:**

El costo de combustible se calcula en función de las millas recorridas, el volumen de pesca, y el precio del galón de combustible. La fórmula es:
""")
st.latex(r"""
\text{Costo\_Combustible} = \left(\text{Millas\_Recorridas} \times 0.05 \div 2 \times \left(1 + 1 + \text{Volumen} \times 0.0001\right)\right) \times \text{Precio\_Galón}
""")

# Fórmula para el cálculo de la ganancia
st.markdown("""
**3. Cálculo de la Ganancia:**

Finalmente, la ganancia se obtiene restando los costos de combustible del valor total de las ventas:
""")
st.latex(r"""
\text{Ganancia} = \text{Venta} - \text{Costo\_Combustible}
""")
# Gráficos Estadísticos Descriptivos
st.write("### Análisis Estadístico Descriptivo")

# Selección de variables para el análisis
numerical_columns = ['Temperatura_Agua_°C', 'Profundidad_m', 'Salinidad_PSU',
                    'Velocidad_Viento_m_s', 'Corriente_Marina_m_s', 'CPUE',
                    'Caballos_Motor', 'Millas_Recorridas', 'Precio_Kg',
                    'Talla_cm', 'Costo_Combustible', 'Ganancia', 'Tripulantes']

selected_var = st.multiselect("Selecciona las variables para visualizar", numerical_columns, default=numerical_columns[:3])

# Selección del tipo de gráfico
plot_type = st.selectbox("Selecciona el tipo de gráfico", ["Histograma", "Barra", "Boxplot", "Scatter"])

# Selección de escala
scale_option = st.selectbox("Selecciona la escala del eje Y", ["Lineal", "Logarítmica"])

# Función para generar gráficos
def generar_grafico(var, tipo, escala):
    if tipo == "Histograma":
        fig = px.histogram(df, x=var, nbins=30, title=f'Histograma de {var}')
    elif tipo == "Barra":
        fig = px.bar(df, x=var, title=f'Barra de {var}')
    elif tipo == "Boxplot":
        fig = px.box(df, y=var, title=f'Boxplot de {var}')
    elif tipo == "Scatter":
        # Si se selecciona scatter, pedir otra variable
        var_y = st.selectbox(f"Selecciona la variable Y para {var}", numerical_columns, index=1)
        fig = px.scatter(df, x=var, y=var_y, title=f'Scatter de {var} vs {var_y}')
    else:
        fig = {}
    
    if escala == "Logarítmica":
        fig.update_yaxes(type='log')
    
    return fig

# Generar y mostrar gráficos
for var in selected_var:
    fig = generar_grafico(var, plot_type, scale_option)
    st.plotly_chart(fig)

# Gráficos de Tendencia
st.write("### Gráficos de Tendencia")
st.markdown("Selecciona una variable para ver su tendencia a lo largo del tiempo.")

# Selección de variable para tendencia
trend_var = st.selectbox("Selecciona la variable para la tendencia", numerical_columns, index=0)

# Asegurarse de que 'Inicio_Faena' esté ordenado
df_sorted = df.sort_values('Inicio_Faena')

# Crear gráfico de tendencia
fig_trend = px.line(df_sorted, x='Inicio_Faena', y=trend_var, title=f'Tendencia de {trend_var} a lo largo del tiempo')
st.plotly_chart(fig_trend)

# Gráfico de Dispersión Interactivo
st.write("### Gráfico de Dispersión Interactivo")
st.markdown("Selecciona las variables para los ejes y una opción para colorear los puntos.")

# Selección de variables para el gráfico de dispersión
scatter_x = st.selectbox("Selecciona la variable para el eje X", numerical_columns, index=0)
scatter_y = st.selectbox("Selecciona la variable para el eje Y", numerical_columns, index=1)

# Opcional: Selección de una variable para colorear
color_option = st.selectbox("Selecciona una variable para colorear los puntos (opcional)", 
                            ["Ninguna"] + numerical_columns + ['Especie', 'Embarcacion'], index=0)

# Crear el gráfico de dispersión
if color_option != "Ninguna":
    fig_scatter = px.scatter(df, x=scatter_x, y=scatter_y, color=color_option,
                             title=f'Gráfico de Dispersión: {scatter_x} vs {scatter_y}',
                             hover_data=['Especie', 'Embarcacion'])
else:
    fig_scatter = px.scatter(df, x=scatter_x, y=scatter_y,
                             title=f'Gráfico de Dispersión: {scatter_x} vs {scatter_y}',
                             hover_data=['Especie', 'Embarcacion'])

# Mostrar el gráfico
st.plotly_chart(fig_scatter)

# Matriz de Correlación
st.write("### Matriz de Correlación entre Variables")
st.markdown("Visualiza las correlaciones lineales entre las variables numéricas seleccionadas.")

# Selección de variables para la matriz de correlación
corr_variables = st.multiselect(
    "Selecciona las variables para incluir en la matriz de correlación",
    options=numerical_columns,
    default=numerical_columns
)

# Calcular la matriz de correlación
if corr_variables:
    corr_matrix = df[corr_variables].corr(method='pearson')
    
    # Crear el mapa de calor interactivo usando Plotly
    fig_corr = px.imshow(corr_matrix,
                         text_auto=True,
                         aspect="auto",
                         color_continuous_scale='RdBu_r',
                         title='Matriz de Correlación de Pearson')
    st.plotly_chart(fig_corr)
    
    # Opcional: Descargar la matriz de correlación
    csv_corr = corr_matrix.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="Descargar Matriz de Correlación como CSV",
        data=csv_corr,
        file_name='matriz_correlacion.csv',
        mime='text/csv',
    )
else:
    st.warning("Por favor, selecciona al menos una variable para mostrar la matriz de correlación.")

# Selección del enfoque para las predicciones
opcion = st.selectbox("Seleccionar el enfoque", ["Embarcación", "Especie"], key="enfoque_selectbox")

# Filtrar según la opción seleccionada
if opcion == "Embarcación":
    seleccion = st.selectbox("Seleccionar la embarcación", df['Embarcacion'].unique(), key="embarcacion_selectbox")
    df_seleccion = df[df['Embarcacion'] == seleccion]
else:
    seleccion = st.selectbox("Seleccionar la especie", df['Especie'].unique(), key="especie_selectbox")
    df_seleccion = df[df['Especie'] == seleccion]

# Mostrar el mapa centrado en la ubicación de las capturas
st.subheader(f"Mapa de capturas para la selección: {seleccion}")
mapa = folium.Map(location=[df_seleccion['Origen_Latitud'].mean(), df_seleccion['Origen_Longuitud'].mean()], zoom_start=6)
for idx, row in df_seleccion.iterrows():
    folium.Marker(
        location=[row['Origen_Latitud'], row['Origen_Longuitud']],
        popup=row['Especie'] if opcion == "Especie" else row['Embarcacion']
    ).add_to(mapa)
folium_static(mapa)

# Definir las características y la variable objetivo
selected_columns = ['Caballos_Motor', 'Millas_Recorridas', 'Precio_Kg', 'Talla_cm', 'Costo_Combustible', 'Ganancia',
                    'Temperatura_Agua_°C', 'Profundidad_m', 'Salinidad_PSU', 'Velocidad_Viento_m_s',
                    'Corriente_Marina_m_s', 'CPUE']

# Normalizar los datos
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[selected_columns] = scaler.fit_transform(df[selected_columns])

X = df_normalized[selected_columns]
y = df_normalized['Volumen_Kg']

# Entrenar modelo de Random Forest
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Gráfico Interactivo: Curvas de Entrenamiento y Validación
st.write("### Evaluación del Modelo")

# Predicción en el conjunto de validación
y_pred = modelo_rf.predict(X_val)

# Cálculo de las métricas
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# Mostrar las métricas
st.markdown("**Métricas de Evaluación del Modelo:**")
st.write(f"**Error Cuadrático Medio (MSE):** {mse:.4f}")
st.write(f"**Error Absoluto Medio (MAE):** {mae:.4f}")
st.write(f"**Coeficiente de Determinación (R²):** {r2:.4f}")

# Explicación de las métricas
st.markdown("""
- **Error Cuadrático Medio (MSE):** Mide el promedio de los errores al cuadrado entre los valores reales y los predichos. Un valor menor indica un mejor desempeño del modelo.
- **Error Absoluto Medio (MAE):** Representa el promedio de los errores absolutos entre los valores reales y los predichos. Es útil para entender el error promedio en las predicciones.
- **Coeficiente de Determinación (R²):** Indica la proporción de la variabilidad de la variable dependiente que es explicada por el modelo. Un valor cercano a 1 sugiere un buen ajuste.
""")

# Gráfico de Valores Reales vs Predichos
st.write("#### Valores Reales vs Predichos")
fig_real_vs_pred = px.scatter(x=y_val, y=y_pred, labels={'x': 'Valores Reales', 'y': 'Valores Predichos'},
                              title='Valores Reales vs Predichos')
fig_real_vs_pred.add_shape(type="line",
                           x0=y_val.min(), y0=y_val.min(),
                           x1=y_val.max(), y1=y_val.max(),
                           line=dict(color="Red", dash="dash"))
st.plotly_chart(fig_real_vs_pred)

# Gráfico de Curvas de Aprendizaje
st.write("#### Curvas de Aprendizaje")
st.markdown("Selecciona el número de árboles para visualizar cómo afecta al desempeño del modelo.")

# Selección del rango de n_estimators
n_estimators_range = st.slider("Selecciona el rango de n_estimators", 10, 200, (10, 100), step=10)

# Preparar los rangos
n_estimators_list = list(range(n_estimators_range[0], n_estimators_range[1]+1, 10))
train_scores = []
val_scores = []

for n in n_estimators_list:
    modelo = RandomForestRegressor(n_estimators=n, random_state=42)
    modelo.fit(X_train, y_train)
    train_score = modelo.score(X_train, y_train)
    val_score = modelo.score(X_val, y_val)
    train_scores.append(train_score)
    val_scores.append(val_score)

# Crear el gráfico de curvas de aprendizaje
fig_learning = go.Figure()
fig_learning.add_trace(go.Scatter(x=n_estimators_list, y=train_scores, mode='lines+markers', name='Entrenamiento'))
fig_learning.add_trace(go.Scatter(x=n_estimators_list, y=val_scores, mode='lines+markers', name='Validación'))
fig_learning.update_layout(title='Curvas de Aprendizaje',
                           xaxis_title='Número de Árboles (n_estimators)',
                           yaxis_title='Puntuación R²',
                           yaxis=dict(range=[0,1]))
st.plotly_chart(fig_learning)

# Predicción con el formulario
st.subheader("Formulario para predicciones personalizadas")
caballos_motor = st.slider("Caballos de Motor", 10, 50, 20)
millas_recorridas = st.slider("Millas Recorridas", 1, 100, 10)
precio_kg = st.slider("Precio por Kg", 1.0, 50.0, 15.0)
talla_cm = st.slider("Talla del Pescado (cm)", 10, 100, 30)
costo_combustible = st.slider("Costo de Combustible (S/.)", 10.0, 1000.0, 100.0)
ganancia = st.slider("Ganancia (S/.)", 50.0, 5000.0, 500.0)
temperatura_agua = st.slider("Temperatura del Agua (°C)", 15.0, 30.0, 25.0)
profundidad = st.slider("Profundidad del Mar (m)", 10.0, 100.0, 50.0)
salinidad = st.slider("Salinidad del Agua (PSU)", 30.0, 40.0, 35.0)
velocidad_viento = st.slider("Velocidad del Viento (m/s)", 1.0, 20.0, 5.0)
corriente_marina = st.slider("Corriente Marina (m/s)", 0.1, 2.0, 0.5)
cpue = st.slider("Índice CPUE", 0.1, 5.0, 1.0)

# Crear un nuevo conjunto de datos con los valores ingresados
nuevos_datos = np.array([[caballos_motor, millas_recorridas, precio_kg, talla_cm, costo_combustible, ganancia, 
                         temperatura_agua, profundidad, salinidad, velocidad_viento, corriente_marina, cpue]])

# Normalizar los nuevos datos
nuevos_datos_normalizados = scaler.transform(nuevos_datos)

# Predicción del modelo con los datos ingresados
prediccion_nueva = modelo_rf.predict(nuevos_datos_normalizados)
st.write(f"**Predicción del Volumen Capturado (Kg) basado en los datos ingresados:** {prediccion_nueva[0]:.2f}")

# Gráfico de importancia de características
importances = modelo_rf.feature_importances_
indices = X.columns
feature_importances = pd.Series(importances, index=indices).sort_values(ascending=False)

st.subheader('Importancia de Características')
fig = px.bar(feature_importances, x=feature_importances.index, y=feature_importances.values,
             title="Importancia de las Características para Predicción")
st.plotly_chart(fig)
