#Creamos el archivo de la APP en el interprete principal (Phyton)

#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
from streamlit_echarts import st_echarts
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score

# Definimos la instancia
@st.cache_resource
def load_data():
    # Lectura de archivos CSV
    Naples = pd.read_csv("Datos_limpios_Naples.csv").drop(['Unnamed: 0'], axis=1)
    Rio = pd.read_csv("Rio de Janeiro sin atipicos.csv")
    Berlin = pd.read_csv("Datos_limpios_Berlin.csv").drop(['Unnamed: 0'], axis=1)
    Mexico = pd.read_csv("México sin atipicos.csv").drop(['Unnamed: 0'], axis=1)

    # Columnas numéricas
    numeric_Naples = Naples.select_dtypes(['float', 'int'])
    numeric_Rio = Rio.select_dtypes(['float', 'int'])
    numeric_Berlin = Berlin.select_dtypes(['float', 'int'])
    numeric_Mexico = Mexico.select_dtypes(['float', 'int'])

    # Columnas de texto
    text_Naples = Naples.select_dtypes(['object'])
    text_Rio = Rio.select_dtypes(['object'])
    text_Berlin = Berlin.select_dtypes(['object'])
    text_Mexico = Mexico.select_dtypes(['object'])

    # Columnas categóricas (ejemplo)
    unique_categories_host = Naples['host_is_superhost'].unique()

    return (
        Naples, Rio, Berlin, Mexico,
        numeric_Naples, numeric_Rio, numeric_Berlin, numeric_Mexico,
        text_Naples, text_Rio, text_Berlin, text_Mexico,
        unique_categories_host
    )

# Cargar datos  
( Naples, Rio, Berlin, Mexico,
numeric_Naples, numeric_Rio, numeric_Berlin, numeric_Mexico,
text_Naples, text_Rio, text_Berlin, text_Mexico,
unique_categories_host) = load_data()

############# CREACIÓN DEL DASHBOARD Vista principal

#Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("ᯓ ✈︎ Datos")
st.sidebar.subheader("Presentación de los datos")

# Checkbox para mostrar dataset (para verificar que carga bien los datos)
# check_box_Naples = st.sidebar.checkbox(label="📂 Mostrar Dataset Naples")
check_box_Mexico = st.sidebar.checkbox(label="📂 Mostrar Dataset México")

# Condicional para que aparezca el dataframe
# if check_box_Naples:
#     st.header("📊 Dataset Completo")
#     st.write(Naples)

#     st.subheader("🔠 Columnas del Dataset")
#     st.write(Naples.columns)

#     st.subheader("📈 Estadísticas Descriptivas")
#     st.write(Naples.describe())

# Condicional para que aparezca el dataframe
if check_box_Mexico:
    st.header("📊 Dataset Completo")
    st.write(Mexico)

    st.subheader("🔠 Columnas del Dataset")
    st.write(Mexico.columns)

    st.subheader("📈 Estadísticas Descriptivas")
    st.write(Mexico.describe())

# Checkbox para mostrar etapas
etapas_checkbox = st.sidebar.checkbox(label="📌 Mostrar Etapas del Análisis")

# Si se activa el checkbox, mostramos el selectbox
if etapas_checkbox:
    st.sidebar.subheader("Etapas")
    View = st.sidebar.selectbox(
        label="🔽 Selecciona una etapa del análisis:",
        options=[
            "Etapa I. Modelado explicativo", 
            "Etapa II. Modelado predictivo"]
    )

    if View == "Etapa I. Modelado explicativo":
        st.sidebar.title("🧠 Etapa I – Modelado Explicativo")
        st.sidebar.header("Exploración de características importantes de los datos")

    elif View == "Etapa II. Modelado predictivo":
        st.sidebar.title("🤖 Etapa II – Modelado Predictivo")
        st.sidebar.header("Predicción de tendencias y patrones")

        st.sidebar.subheader("Tipo de Regresión")
        tipo_regresion = st.sidebar.selectbox(
            label="📊 Selecciona el tipo de regresión:",
            options=[
                "Regresión Lineal Simple", 
                "Regresión Lineal Múltiple", 
                "Regresión Logística",
            ]
        )

        if tipo_regresion == "Regresión Logística":
            st.sidebar.subheader("Variables para regresión logística")

            # Seleccionar variables sobre todas las disponibles
            variables_comunes = list(
                set(numeric_Naples.columns) &
                set(numeric_Rio.columns) &
                set(numeric_Berlin.columns) &
                set(numeric_Mexico.columns)
            )

            categorias_comunes = list(
                set(text_Naples.columns) &
                set(text_Rio.columns) &
                set(text_Berlin.columns) &
                set(text_Mexico.columns)
            )

            x_vars = st.sidebar.multiselect("Variables independientes (X):", options=variables_comunes)
            y_var = st.sidebar.selectbox("Variable dependiente categórica (Y):", options=categorias_comunes)

            if x_vars and y_var:
                    st.subheader(f"📊 Comparación de regresión logística entre países para predecir: `{y_var}`")

                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import LabelEncoder
                    from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    
                    resultados = []
                    individuales = []  # Guardaremos los datos de cada país para mostrarlos después

                    for nombre, df in [("Naples", Naples), ("Rio", Rio), ("Berlin", Berlin), ("Mexico", Mexico)]:
                        try:
                            if all(var in df.columns for var in x_vars + [y_var]):
                                X = df[x_vars]
                                y = df[y_var]

                                # Codificar variable dependiente
                                encoder = LabelEncoder()
                                y_encoded = encoder.fit_transform(y)

                                model = LogisticRegression(max_iter=200)
                                model.fit(X, y_encoded)
                                y_pred = model.predict(X)

                                # Métricas
                                precision = precision_score(y_encoded, y_pred, average='binary' if len(np.unique(y_encoded)) == 2 else 'macro')
                                accuracy = accuracy_score(y_encoded, y_pred)
                                recall = recall_score(y_encoded, y_pred, average='binary' if len(np.unique(y_encoded)) == 2 else 'macro')

                                resultados.append({
                                    "País": nombre,
                                    "Precisión": precision,
                                    "Exactitud": accuracy,
                                    "Sensibilidad": recall
                                })

                                # Guardamos los datos para mostrar después
                                individuales.append((nombre, model, encoder, x_vars, y_encoded, y_pred))

                            else:
                                st.warning(f"⚠️ Las variables seleccionadas no están disponibles en el dataset de {nombre}.")
                        except Exception as e:
                            st.error(f"❌ Error al procesar {nombre}: {e}")

                    # 🔼 MOSTRAR PRIMERO LA GRÁFICA Y LA TABLA
                    if resultados:
                        st.subheader("📊 Comparación entre países")                        
                        comparacion_df = pd.DataFrame(resultados)
                        comparacion_df[["Precisión", "Exactitud", "Sensibilidad"]] = comparacion_df[["Precisión", "Exactitud", "Sensibilidad"]].applymap(lambda x: round(x, 4))
                        st.dataframe(comparacion_df)

                        # Calcular el promedio de métricas
                        comparacion_df["Promedio"] = comparacion_df[["Precisión", "Exactitud", "Sensibilidad"]].mean(axis=1)

                        # Obtener el país con mejor desempeño general
                        mejor_pais_row = comparacion_df.loc[comparacion_df["Promedio"].idxmax()]
                        mejor_pais = mejor_pais_row["País"]
                        mejor_score = mejor_pais_row["Promedio"]

                        # Obtener métricas individuales
                        mejor_precision = mejor_pais_row["Precisión"]
                        mejor_exactitud = mejor_pais_row["Exactitud"]
                        mejor_sensibilidad = mejor_pais_row["Sensibilidad"]

                        # Mostrar tarjeta visual en Streamlit
                        st.markdown(f"""
                            <div style="
                                background-color: #e6f9f0;
                                padding: 20px;
                                border-radius: 12px;
                                border: 2px solid #34c38f;
                                width: 420px;
                                font-family: 'Segoe UI', sans-serif;
                                box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
                                margin-top: 20px;
                                color: #1a202c;  /* Color del texto */
                            ">
                                <h3 style="color: #2f855a;">🏆 País con mejor desempeño general</h3>
                                <p><strong>🌍 País:</strong> {mejor_pais}</p>
                                <p><strong>📊 Promedio de métricas:</strong> {mejor_score:.2f}</p>
                                <ul style="list-style-type: none; padding-left: 0;">
                                    <li><strong>✔️ Precisión:</strong> {mejor_precision:.2f}</li>
                                    <li><strong>✔️ Exactitud:</strong> {mejor_exactitud:.2f}</li>
                                    <li><strong>✔️ Sensibilidad:</strong> {mejor_sensibilidad:.2f}</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)

                        st.subheader("📈 Comparación visual de métricas por país")
                        melted_df = comparacion_df.melt(id_vars="País", var_name="Métrica", value_name="Valor")
                        fig = px.bar(melted_df, 
                                    x='País', 
                                    y='Valor', 
                                    color='Métrica', 
                                    barmode='group',
                                    title='Métricas de Regresión Logística por País')
                        st.plotly_chart(fig, use_container_width=True)


                        # Calcular el promedio de métricas
                        comparacion_df["Promedio"] = comparacion_df[["Precisión", "Exactitud", "Sensibilidad"]].mean(axis=1)

                        # Obtener el país con mejor desempeño general
                        mejor_pais_row = comparacion_df.loc[comparacion_df["Promedio"].idxmax()]
                        mejor_pais = mejor_pais_row["País"]
                        mejor_score = mejor_pais_row["Promedio"]

                        # Obtener métricas individuales
                        mejor_precision = mejor_pais_row["Precisión"]
                        mejor_exactitud = mejor_pais_row["Exactitud"]
                        mejor_sensibilidad = mejor_pais_row["Sensibilidad"]

                        # Mostrar tarjeta visual en Streamlit
                        st.markdown(f"""
                            <div style="
                                background-color: #e6f9f0;
                                padding: 20px;
                                border-radius: 12px;
                                border: 2px solid #34c38f;
                                width: 420px;
                                font-family: 'Segoe UI', sans-serif;
                                box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
                                margin-top: 20px;
                                color: #1a202c;  /* Color del texto */
                            ">
                                <h3 style="color: #2f855a;">🏆 País con mejor desempeño general</h3>
                                <p><strong>🌍 País:</strong> {mejor_pais}</p>
                                <p><strong>📊 Promedio de métricas:</strong> {mejor_score:.2f}</p>
                                <ul style="list-style-type: none; padding-left: 0;">
                                    <li><strong>✔️ Precisión:</strong> {mejor_precision:.2f}</li>
                                    <li><strong>✔️ Exactitud:</strong> {mejor_exactitud:.2f}</li>
                                    <li><strong>✔️ Sensibilidad:</strong> {mejor_sensibilidad:.2f}</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)


                    # 🔽 Luego mostramos los detalles individuales por país
                    for nombre, model, encoder, x_vars, y_encoded, y_pred in individuales:
                        st.markdown(f"### 🌍 Resultados para {nombre}")
                        coef_data = pd.DataFrame({
                            "Variable": x_vars,
                            "Coeficiente": model.coef_[0]
                        })
                        st.write("🔢 Coeficientes")
                        st.dataframe(coef_data)

                        conf_matrix = confusion_matrix(y_encoded, y_pred)
                        conf_df = pd.DataFrame(conf_matrix,
                                            index=[f"Real {label}" for label in encoder.classes_],
                                            columns=[f"Predicho {label}" for label in encoder.classes_])
                        st.write("🧩 Matriz de Confusión")
                        st.dataframe(conf_df)

                        fig, ax = plt.subplots()
                        sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_title(f"Matriz de Confusión - {nombre}")
                        ax.set_xlabel("Predicción")
                        ax.set_ylabel("Real")
                        st.pyplot(fig)

                        






                    

                    
