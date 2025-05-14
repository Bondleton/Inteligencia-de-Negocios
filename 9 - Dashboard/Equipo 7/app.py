#Creamos el archivo de la APP en el interprete principal (Phyton)

#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
from streamlit_echarts import st_echarts
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score

# Configuración inicial
st.set_page_config(layout="wide", page_title="Comparación entre países - Airbnb")

# Aplicamos estilos con CSS


# Definimos la instancia
@st.cache_resource
def load_data():
    # Lectura de archivos CSV
    Naples = pd.read_csv("Datos_limpios_Naples.csv").drop(['Unnamed: 0'], axis=1)
    Rio = pd.read_csv("Rio de Janeiro sin atipicos.csv")
    Berlin = pd.read_csv("Datos_limpios_Berlin.csv").drop(['Unnamed: 0'], axis=1)
    Mexico = pd.read_csv("México sin atipicos.csv").drop(['Unnamed: 0'], axis=1)

    #Extracción de caracteristicas
    Rio1 = pd.read_csv("Rio.csv")
    Naples1 = pd.read_csv("Naples.csv")
    Berlin1 = pd.read_csv("Berlin.csv")
    Mexico1 = pd.read_csv("Mexico.csv")

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
        Naples, Rio, Berlin, Mexico, Naples1, Rio1, Berlin1, Mexico1,
        numeric_Naples, numeric_Rio, numeric_Berlin, numeric_Mexico,
        text_Naples, text_Rio, text_Berlin, text_Mexico,
        unique_categories_host
    )

# Cargar datos  
( Naples, Rio, Berlin, Mexico, Naples1, Rio1, Berlin1, Mexico1,
numeric_Naples, numeric_Rio, numeric_Berlin, numeric_Mexico,
text_Naples, text_Rio, text_Berlin, text_Mexico,
unique_categories_host) = load_data()

############# CREACIÓN DEL DASHBOARD Vista principal

#Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("ᯓ ✈︎ Dashboard")
st.sidebar.subheader("Presentación de los datos")

# Checkbox para mostrar info
info = st.sidebar.checkbox(label="🍃 Mostrar Información")


# Si se activa el checkbox, mostramos el selectbox
if info:
    st.sidebar.subheader("Paises")
    st.image("Equipo 7.png", use_container_width=True)

# Checkbox para mostrar etapas
etapas_checkbox = st.sidebar.checkbox(label="📌 Mostrar Etapas del Análisis")
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

        import streamlit as st
        import pandas as pd
        import plotly.express as px
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pydeck as pdk

        @st.cache_data
        def load_data():
            return {
                "Rio1": pd.read_csv("Rio.csv"),
                "Naples1": pd.read_csv("Naples.csv"),
                "Berlin1": pd.read_csv("Berlin.csv"),
                "Mexico1": pd.read_csv("Mexico.csv")
            }

        data = load_data()
        colores = {
            "Rio1": "green",
            "Naples1": "gold",
            "Berlin1": "black",
            "Mexico1": "red"
        }       

        colores = {
            "Rio1": "green",
            "Naples1": "gold",
            "Berlin1": "black",
            "Mexico1": "red"
        }

        # Conversión de moneda a pesos mexicanos (MXN)
        conversion_monedas = {
            "Rio1": 3.5,   # BRL a MXN
            "Naples1": 18.0,          # EUR a MXN
            "Berlin1": 18.0,          # EUR a MXN
            "Mexico1": 1.0            # Ya está en MXN
        }

        # Convertir columna 'price' en cada DataFrame
        for ciudad, df in data.items():
            df['price_mxn'] = pd.to_numeric(df['price'], errors='coerce') * conversion_monedas[ciudad]
            df['price_mxn'] = df['price_mxn'].round(2)


        # Variables clasificadas
        variables_numericas = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
        variables_categoricas = ['host_response_time', 'host_verifications', 'room_type', 'property_type', 'host_acceptance_rate']
        variables_scores = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                            'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                            'review_scores_value']
        variables_binarias = ['instant_bookable', 'has_availability', 'host_is_superhost', 'host_has_profile_pic',
                            'host_identity_verified']
        variable_precio=['price_mxn']
        todas_las_variables = variables_numericas + variables_categoricas + variables_scores + variables_binarias + variable_precio

        # Sidebar
        st.sidebar.title("Panel de control")

        # Variable a visualizar
        selected_var = st.sidebar.selectbox("Selecciona una variable:", todas_las_variables)
        show_table = st.sidebar.checkbox("Mostrar tabla")


        # Tamaño de gráfico
        st.sidebar.subheader("Tamaño del gráfico")
        width = st.sidebar.slider("Ancho", 4, 20, 10)
        height = st.sidebar.slider("Alto", 1, 15, 6)

        # Título principal
        st.title("Exploración Comparativa")


        # Visualización según el tipo de variable
        st.header(f"Visualización para: {selected_var}")

        # Variables numéricas: diagrama de puntos
        if selected_var in variables_numericas:
            fig, ax = plt.subplots(figsize=(width, height))
            for ciudad, df in data.items():
                sns.stripplot(x=[ciudad]*len(df), y=df[selected_var], color=colores[ciudad], alpha=0.5, ax=ax)
            ax.set_title(f"Distribución de {selected_var}")
            st.pyplot(fig)


        # Variables categóricas: barras por país en cuadrícula 2x2
        elif selected_var in variables_categoricas:
            
            ciudades = list(data.keys())
            
            for i in range(0, len(ciudades), 2):
                cols = st.columns(2)
                for j, ciudad in enumerate(ciudades[i:i+2]):
                    with cols[j]:
                        df = data[ciudad]

                        # Preprocesamiento específico para 'host_acceptance_rate'
                        if selected_var == "host_acceptance_rate":
                            df = df.dropna(subset=[selected_var])
                            df['category'] = pd.qcut(df[selected_var], 5, duplicates='drop')
                            counts = df['category'].value_counts().sort_index()
                        else:
                            counts = df[selected_var].value_counts().nlargest(5)

                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.bar(counts.index.astype(str), counts.values, color=colores[ciudad])
                        ax.set_title(ciudad)
                        ax.set_xlabel(selected_var)
                        ax.set_ylabel("Frecuencia")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)


        # Scores: polígonos de frecuencia
        elif selected_var in variables_scores:
            fig, ax = plt.subplots(figsize=(width, height))
            for ciudad, df in data.items():
                sns.kdeplot(df[selected_var].dropna(), label=ciudad, color=colores[ciudad], ax=ax)
            ax.set_title(f"Densidad de {selected_var}")
            ax.legend()
            st.pyplot(fig)

        # Variables binarias: pastel por ciudad (en columnas)
        elif selected_var in variables_binarias:

            col1, col2 = st.columns(2)
            ciudades = list(data.keys())

            for i in range(0, len(ciudades), 2):
                col_a = col1 if i % 4 == 0 else col2
                col_b = col2 if i % 4 == 0 else col1

                for col, ciudad in zip([col1, col2], ciudades[i:i+2]):
                    with col:
                        fig, ax = plt.subplots(figsize=(4, 4))  # Tamaño pequeño
                        df = data[ciudad]
                        df[selected_var] = df[selected_var].astype(str)
                        df[selected_var].value_counts().plot.pie(autopct='%1.1f%%', colors=[colores[ciudad], 'lightgray'], ax=ax)
                        ax.set_ylabel('')
                        ax.set_title(ciudad)
                        st.pyplot(fig)

        #Precio
        elif selected_var in variable_precio:
            fig, axs = plt.subplots(2, 2, figsize=(width, height + 2))
            fig.tight_layout(pad=10)
            ciudades = list(data.keys())

            for i, ciudad in enumerate(ciudades):
                df = data[ciudad].dropna(subset=['price_mxn'])
                df['precio_categoria'] = pd.qcut(df['price_mxn'], q=5, duplicates='drop')

                conteo = df['precio_categoria'].value_counts().sort_index()

                fila = i // 2
                col = i % 2

                axs[fila, col].bar(conteo.index.astype(str), conteo.values, color=colores[ciudad])
                axs[fila, col].set_title(f"Categorías de precio en {ciudad}")
                axs[fila, col].tick_params(axis='x', rotation=45)

            st.pyplot(fig)

            # ------------------ Mapa con control de intervalo de precios -------------------
            st.subheader("Mapa interactivo de precios (MXN)")

            ciudad_mapa = st.sidebar.selectbox("Selecciona una ciudad para el mapa", ciudades)
            df_mapa = data[ciudad_mapa].dropna(subset=['latitude', 'longitude', 'price_mxn'])

            if not df_mapa.empty:
                precio_min, precio_max = int(df_mapa['price_mxn'].min()), int(df_mapa['price_mxn'].max())

                rango = st.sidebar.slider(
                    "Rango de precios a visualizar en el mapa (MXN)",
                    min_value=precio_min,
                    max_value=precio_max,
                    value=(precio_min, precio_max)
                )

                df_filtrado = df_mapa[(df_mapa['price_mxn'] >= rango[0]) & (df_mapa['price_mxn'] <= rango[1])].copy()

                if df_filtrado.empty:
                    st.warning("No hay alojamientos en ese rango de precios.")
                else:
                    max_price = df_filtrado['price_mxn'].max()
                    df_filtrado['color'] = df_filtrado['price_mxn'].apply(
                        lambda x: [255, max(0, 255 - int((x / max_price) * 255)), 0]
                    )

                    st.pydeck_chart(pdk.Deck(
                        initial_view_state=pdk.ViewState(
                            latitude=df_filtrado["latitude"].mean(),
                            longitude=df_filtrado["longitude"].mean(),
                            zoom=11,
                            pitch=45,
                        ),
                        layers=[
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=df_filtrado,
                                get_position='[longitude, latitude]',
                                get_color='color',
                                get_radius=200,
                                pickable=True
                            ),
                        ],
                        tooltip={"text": "Precio: {price_mxn} MXN"}
                    ))
            else:
                st.warning(f"No hay datos geográficos disponibles para {ciudad_mapa}.")


        # Mostrar tabla resumen solo si se activa el checkbox
        if show_table:
            st.markdown("---")
            st.subheader("Tabla resumen por país")

            pais_seleccionado = st.sidebar.selectbox("Selecciona un país para ver su tabla", list(data.keys()))

            df = data[pais_seleccionado]
            resumen = df[selected_var].value_counts(dropna=False).reset_index()
            resumen.columns = [selected_var, "Frecuencia"]

            st.markdown(f"**{pais_seleccionado}**")
            st.dataframe(resumen)



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

        if tipo_regresion == "Regresión Lineal Múltiple":            
            st.sidebar.header("🔧 Panel de Control")
            st.sidebar.subheader("Variables para Regresión Lineal Múltiple")

            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
            import numpy as np
            import plotly.express as px
            import plotly.graph_objects as go
                        

            # Interfaz Streamlit
            st.title("Regresión Lineal Múltiple por Ciudad")
            
            # Cargar datos
            data = load_data()
            sample_df = data[0]  # Naples

            data_dict = {
                "Naples": data[0],
                "Rio": data[1],
                "Berlin": data[2],
                "Mexico": data[3]
            }

            # Selección de variables
            st.sidebar.header("Selecciona las variables")
            numeric_cols = sample_df.select_dtypes(include=['float', 'int']).columns.tolist()

            target = st.sidebar.selectbox("Selecciona la variable dependiente (Y)", numeric_cols)
            features = st.sidebar.multiselect("Selecciona las variables independientes (X)", [col for col in numeric_cols if col != target])

            # Verifica si se han seleccionado variables
            if target and features:
                

                # Selección de la variable X para graficar antes del ciclo de las ciudades
                eje_x = st.selectbox("Selecciona una variable X para graficar:", options=features,  key="variable_x")

                for city, df in data_dict.items():
                    st.subheader(f"📊 Resultados para {city}")
                    df = df.dropna(subset=features + [target])
                    X = df[features]
                    y = df[target]

                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)

                    # Gráficos de relación entre cada X e Y
                    st.markdown("**Relación entre variables independientes y dependiente**")
                    cols = st.columns(2)  # Máximo 2 gráficos por fila

                    for i, feature in enumerate(features):
                        with cols[i % 2]:
                            fig, ax = plt.subplots(figsize=(5, 4))
                            sns.scatterplot(x=df[feature], y=y, ax=ax, color="green")
                            ax.set_title(f"{feature} vs {target}", fontsize=11)
                            ax.set_xlabel(feature, fontsize=9)
                            ax.set_ylabel(target, fontsize=9)
                            plt.tight_layout()
                            st.pyplot(fig)
                        if (i + 1) % 2 == 0:
                            cols = st.columns(2)  # Nueva fila después de cada 2

                    # Métricas: R² y r
                    r2 = r2_score(y, y_pred)
                    r = np.corrcoef(y, y_pred)[0, 1]
                    st.markdown(f"**Coeficiente de determinación (R²)**: {r2:.3f}")
                    st.markdown(f"**Coeficiente de correlación (r)**: {r:.3f}")

                    # Genera el gráfico comparativo con la variable X seleccionada
                    st.subheader("📉 Comparación de valores reales vs. predichos")
                    fig = px.scatter(
                        df,
                        x=eje_x,
                        y=target,
                        labels={'x': eje_x, 'y': target},
                        title=f"Comparación de {eje_x} vs {target}"
                    )

                    fig.add_scatter(
                        x=df[eje_x],
                        y=y_pred,
                        mode='markers',
                        name='Valores predichos',
                        marker=dict(color='Orange')
                    )

                    st.plotly_chart(fig)

            else:
                st.warning("Selecciona una variable dependiente y al menos una independiente para ver los resultados.")

        elif tipo_regresion == "Regresión Lineal Simple":
            st.sidebar.subheader("Variables para regresión lineal simple")

            # Solo se permiten variables que existan en todos los países
            variables_comunes = list(
                set(numeric_Naples.columns) &
                set(numeric_Rio.columns) &
                set(numeric_Berlin.columns) &
                set(numeric_Mexico.columns)
            )

            x_var = st.sidebar.selectbox("Variable independiente (X):", options=variables_comunes)
            y_var = st.sidebar.selectbox("Variable dependiente (Y):", options=variables_comunes)

            if x_var and y_var:
                st.subheader(f"📈 Comparación de regresión lineal simple entre países para: `{y_var}` vs `{x_var}`")

                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                from scipy.stats import pearsonr

                resultados = []
                graficas = []

                for nombre, df in [("Naples", Naples), ("Rio", Rio), ("Berlin", Berlin), ("Mexico", Mexico)]:
                    try:
                        if all(var in df.columns for var in [x_var, y_var]):
                            X = df[[x_var]]
                            y = df[y_var]
                            model = LinearRegression()
                            model.fit(X, y)
                            y_pred = model.predict(X)

                            r2 = r2_score(y, y_pred)
                            r, _ = pearsonr(df[x_var], df[y_var])
                            coef = model.coef_[0]
                            intercepto = model.intercept_

                            resultados.append({
                                "País": nombre,
                                "Coeficiente (pendiente)": coef,
                                "Intercepto": intercepto,
                                "R²": r2,
                                "r": r
                            })

                            # Guardar gráfica individual
                            fig = px.scatter(x=X.squeeze(), y=y, labels={'x': x_var, 'y': y_var},
                                            title=f"{nombre}: {y_var} vs {x_var}")
                            fig.add_scatter(x=X.squeeze(), y=y_pred, mode='lines', name='Línea de regresión', line=dict(color='firebrick'))
                            graficas.append((nombre, fig))

                        else:
                            st.warning(f"⚠️ Las variables seleccionadas no están disponibles en el dataset de {nombre}.")
                    except Exception as e:
                        st.error(f"❌ Error al procesar {nombre}: {e}")

                if resultados:
                    comparacion_df = pd.DataFrame(resultados)
                    comparacion_df[["Coeficiente (pendiente)", "Intercepto", "R²", "r"]] = comparacion_df[["Coeficiente (pendiente)", "Intercepto", "R²", "r"]].applymap(lambda x: round(x, 4))

                    st.markdown("### 📊 Comparación de métricas entre países")
                    st.dataframe(comparacion_df)

                    # Graficar métricas por país
                    melted = comparacion_df.melt(id_vars="País", var_name="Métrica", value_name="Valor")
                    fig_bar = px.bar(melted, x="País", y="Valor", color="Métrica", barmode="group", title="Métricas de Regresión Lineal Simple por País")
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Mostrar gráficas individuales
                    st.markdown("## 📌 Gráficas individuales por país")
                    for nombre, fig in graficas:
                        st.markdown(f"### 🌍 {nombre}")
                        st.plotly_chart(fig)



        elif tipo_regresion == "Regresión Logística":
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
                        # st.subheader("📊 Comparación entre países")                        
                        comparacion_df = pd.DataFrame(resultados)
                        comparacion_df[["Precisión", "Exactitud", "Sensibilidad"]] = comparacion_df[["Precisión", "Exactitud", "Sensibilidad"]].applymap(lambda x: round(x, 4))
                        # st.dataframe(comparacion_df)
                        
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

                        

                        # Crear dos columnas: una para la tarjeta y otra para la gráfica
                        col1, col2 = st.columns([1, 2])  # Puedes ajustar el ancho relativo

                        with col1:
                            # Tarjeta visual
                            st.markdown(f"""
                                <div style="
                                    background-color: #e6f9f0;
                                    padding: 20px;
                                    border-radius: 12px;
                                    border: 2px solid #34c38f;
                                    font-family: 'Segoe UI', sans-serif;
                                    box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
                                    color: #1a202c;
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

                        with col2:
                            # Gráfica
                            melted_df = comparacion_df.melt(id_vars="País", var_name="Métrica", value_name="Valor")
                            fig = px.bar(melted_df, 
                                        x='País', 
                                        y='Valor', 
                                        color='Métrica', 
                                        barmode='group',
                                        title='Métricas de Regresión Logística por País')
                            st.plotly_chart(fig, use_container_width=True)

                        import itertools

                        # 🔽 Crear lista para almacenar las figuras de cada país
                        figuras = []

                        for nombre, model, encoder, x_vars, y_encoded, y_pred in individuales:
                            coef_data = pd.DataFrame({
                                "Variable": x_vars,
                                "Coeficiente": model.coef_[0]
                            })

                            conf_matrix = confusion_matrix(y_encoded, y_pred)
                            conf_df = pd.DataFrame(conf_matrix,
                                                index=[f"Real {label}" for label in encoder.classes_],
                                                columns=[f"Predicho {label}" for label in encoder.classes_])

                            # 🔹 Crear figura de la matriz de confusión
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                            ax.set_title(f"Matriz de Confusión - {nombre}")
                            ax.set_xlabel("Predicción")
                            ax.set_ylabel("Real")

                            # 🔹 Guardar la figura con su nombre y datos de coeficientes
                            figuras.append((nombre, fig, coef_data, conf_df))

                        # 🔽 Mostrar en 2 filas de 2 columnas
                        iterator = iter(figuras)
                        for _ in range(2):  # dos filas
                            cols = st.columns(2)
                            for col in cols:
                                try:
                                    nombre, fig, coef_data, conf_df = next(iterator)
                                    with col:
                                        st.markdown(f"### 🌍 Resultados para {nombre}")
                                        # st.write("🔢 Coeficientes")
                                        # st.dataframe(coef_data)
                                        # st.write("🧩 Matriz de Confusión")
                                        # st.dataframe(conf_df)
                                        st.pyplot(fig)
                                except StopIteration:
                                    break

                            
    
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

                        
