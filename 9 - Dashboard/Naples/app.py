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
    # Lectura de archivo CSV
    Naples = pd.read_csv("Datos_limpios_Naples.csv")
    Naples = Naples.drop(['Unnamed: 0'], axis=1)

    # Columnas numéricas
    numeric_Naples = Naples.select_dtypes(['float', 'int'])
    numeric_cols = numeric_Naples.columns

    # Columnas de texto
    text_Naples = Naples.select_dtypes(['object'])
    text_cols = text_Naples.columns

    # Columnas categóricas
    categorical_column_host = Naples['host_is_superhost']
    unique_categories_host = categorical_column_host.unique()

    return Naples, numeric_cols, text_cols, unique_categories_host, numeric_Naples

# Cargar datos
Naples, numeric_cols, text_cols, unique_categories_host, numeric_Naples = load_data()

############# CREACIÓN DEL DASHBOARD Vista principal

# Títulos principales
# st.title("ᯓ ✈︎ Análisis de Datos")
# st.header("Descubriendo Nápoles, Italia con datos de Airbnb")
# st.subheader("Presentación de los datos")

# Agregar imagen al sidebar (por ejemplo, un logotipo o imagen representativa)
st.sidebar.image("Napoles.jpg", use_container_width=True)

#Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("ᯓ ✈︎ Análisis de Datos Nápoles, Italia")
# st.sidebar.title("🔍 Menú")
st.sidebar.subheader("Presentación de los datos")

# Checkbox para mostrar dataset
check_box = st.sidebar.checkbox(label="📂 Mostrar Dataset Napolés")

# Condicional para que aparezca el dataframe
if check_box:
    st.header("📊 Dataset Completo")
    st.write(Naples)

    st.subheader("🔠 Columnas del Dataset")
    st.write(Naples.columns)

    st.subheader("📈 Estadísticas Descriptivas")
    st.write(Naples.describe())

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

    # Contenido de la Etapa I
    if View == "Etapa I. Modelado explicativo":
        st.sidebar.title("🧠 Etapa I – Modelado Explicativo")
        st.sidebar.header("Exploración de características importantes de los datos")

        Etapa1 = st.sidebar.selectbox(
        label="📊 Selecciona un tipo de análisis",
        options=[
            "Relación entre variable", 
            "Relación entre diversas variables",
            "Gráfica de pastel",
            "Gráfica de líneas",
            "Líneas múltiples",
            ]
        )

        if Etapa1 == "Relación entre variable":        
            # Sidebar informativo
            st.sidebar.header("🔧 Panel de Control")
            st.sidebar.subheader("Relación entre variable categorica y numerica")
            #st.sidebar.subheader("🗂️ Visualización del dataset")

            #Menus desplegables de opciones de la variables seleccionadas
            Variable_cat= st.sidebar.selectbox(label= "Variable Categórica", options= text_cols)
            Variable_num= st.sidebar.selectbox(label= "Variable Numérica", options= numeric_cols)

            # Botón para mostrar la gráfica
            Button1 = st.sidebar.button(label="📊 Mostrar gráfica")

            # Condicional para mostrar la gráfica solo cuando se presione el botón
            if Button1:
                # Generar un título descriptivo para el gráfico1
                st.subheader(" ༘ ⋆｡˚🍃 Relación entre " + Variable_cat + " y " + Variable_num + " en datos de Airbnb en Nápoles")  
                # titulo_grafico = f"༘ ⋆｡˚🍃 Relación entre {Variable_cat} y {Variable_num} en datos de Airbnb en Nápoles"
                
                # Crear la figura solo cuando se presiona el botón
                figure1 = px.bar(
                    data_frame=Naples, 
                    x=Naples[Variable_cat], 
                    y=Naples[Variable_num], 
                    # title=titulo_grafico
                )
                figure1.update_xaxes(automargin=True)
                figure1.update_yaxes(automargin=True)
                
                # Mostrar la gráfica
                st.plotly_chart(figure1)

        elif Etapa1 == "Relación entre diversas variables":
            # Sidebar informativo
            st.sidebar.title("🔧 Panel de Control")
            st.sidebar.header("Relación entre diversas variables")

            # Selección de variables numéricas
            numerics_vars_selected = st.sidebar.multiselect(
                label="Variables numéricas a graficar", 
                options=numeric_cols
            )

            # Selección de categoría (agrupador)
            category_col = st.sidebar.selectbox(
                label="Categoría para agrupar", 
                options=text_cols
            )

            # Botón para mostrar la gráfica
            Button2 = st.sidebar.button(label="📊 Mostrar gráfica")

            # Condicional para mostrar la gráfica solo cuando se presione el botón
            if Button2:
                st.subheader("༘ ⋆｡˚🍃 Comparativa de variables numéricas agrupadas por " + category_col)

                # Mostramos un gráfico de barras para cada variable numérica seleccionada
                for var in numerics_vars_selected:
                    st.markdown(f"### 📌 {var}")
                    fig = px.bar(
                        data_frame=Naples, 
                        x=category_col, 
                        y=var, 
                        color=category_col, 
                        title=f"{var} por {category_col}", 
                        height=400
                    )
                    fig.update_layout(xaxis_title=category_col, yaxis_title=var)
                    st.plotly_chart(fig)

        elif Etapa1 == "Gráfica de pastel":
                st.sidebar.title("🔧 Panel de Control")
                st.sidebar.header("Gráfica de pastel")

                # Selección de la categoría y valores numéricos
                Variable_cat = st.sidebar.selectbox("Selecciona la categoría (nombres)", options=text_cols)
                Variable_val = st.sidebar.selectbox("Selecciona el valor numérico (valores)", options=numeric_cols)

                # Botón para mostrar la gráfica
                Button_pie = st.sidebar.button(label="📊 Mostrar gráfica")

                if Button_pie:
                    st.subheader(f"༘ ⋆｡˚🍃 Gráfica de pastel: {Variable_val} por {Variable_cat}")
                    
                    # Agrupar datos para evitar repeticiones (por ejemplo, por promedio o suma)
                    grouped_data = Naples.groupby(Variable_cat)[Variable_val].sum().reset_index()

                    fig_pie = px.pie(
                        data_frame=grouped_data, 
                        names=Variable_cat, 
                        values=Variable_val, 
                        title=f"{Variable_val} por {Variable_cat}"
                    )
                    st.plotly_chart(fig_pie)  

        elif Etapa1 == "Gráfica de líneas":
            st.sidebar.title("🔧 Panel de Control")
            st.sidebar.header("📈 Gráfica de líneas")

            # Selección de variables
            eje_x = st.sidebar.selectbox("Eje X (categoría o fecha)", options=text_cols)
            eje_y = st.sidebar.selectbox("Eje Y (numérica)", options=numeric_cols)

            # Botón para mostrar gráfica
            Button_line = st.sidebar.button("📊 Mostrar gráfica")

            if Button_line:
                st.subheader(f"༘ ⋆｡˚🍃 Evolución de {eje_y} por {eje_x}")

                # Agrupamos para evitar muchos puntos repetidos
                grouped_line = Naples.groupby(eje_x)[eje_y].mean().reset_index()

                fig_line = px.line(
                    data_frame=grouped_line,
                    x=eje_x,
                    y=eje_y,
                    markers=True,
                    title=f"{eje_y} por {eje_x}"
                )
                st.plotly_chart(fig_line)
            
        elif Etapa1 == "Líneas múltiples":
            # Sidebar informativo
            st.sidebar.title("🔧 Panel de Control")
            st.sidebar.header("📈 Selección de variables para línea múltiple")

            eje_x = st.sidebar.selectbox("Eje X (Categoría)", options=text_cols)
            vars_seleccionadas = st.sidebar.multiselect("Variables a comparar (Eje Y)", options=numeric_cols)

            mostrar_lineplot = st.sidebar.button("📊 Mostrar gráfica de líneas múltiples")

            if mostrar_lineplot:
                st.subheader("༘ ⋆｡˚🍃 Comparación de múltiples variables numéricas según categoría")

                # Agrupar y preparar los datos
                df_grouped = Naples.groupby(eje_x)[vars_seleccionadas].mean().reset_index()

                # Reestructurar para gráfica
                df_melted = df_grouped.melt(id_vars=eje_x, var_name="Variable", value_name="Valor")

                # Crear el gráfico de líneas
                fig_line = px.line(df_melted, x=eje_x, y="Valor", color="Variable", markers=True)
                fig_line.update_layout(title="Relación entre múltiples variables por " + eje_x)

                st.plotly_chart(fig_line)

        # numerics_vars_selected= st.sidebar.multiselect(label="Variables graficadas", options= numeric_cols)
        # category_selected= st.sidebar.selectbox(label= "Categorias", options= unique_categories_host)


    # Contenido de la Etapa II
    elif View == "Etapa II. Modelado predictivo":
        st.sidebar.title("🤖 Etapa II – Modelado Predictivo")
        st.sidebar.header("Predicción de tendencias y patrones")

        
        
        # Checkbox para mostrar HEATMAP
        heatmap = st.sidebar.checkbox(label="📌 Mostrar Heatmap de Napolés")

        if heatmap:
            st.subheader("༘ ⋆｡˚🔥 Mapa de calor de correlaciones entre variables numéricas")

            selected_vars = st.sidebar.multiselect(
                "Selecciona las variables numéricas a incluir en el mapa de calor:",
                options=numeric_cols,
                default=list(numeric_cols)
            )

            if len(selected_vars) < 2:
                st.warning("Selecciona al menos dos variables para generar el mapa de calor.")
            else:
                # Calcular matriz de correlación
                corr_matrix = Naples[selected_vars].corr().fillna(0)

                # Cálculo simétrico para que el 0 quede centrado
                zmin = float(corr_matrix.replace(1.0, np.nan).min().min())
                zmax = float(corr_matrix.replace(1.0, np.nan).max().max())

            
                # Preparar los datos para el heatmap
                data = []
                for i, row_var in enumerate(selected_vars):
                    for j, col_var in enumerate(selected_vars):
                        value = round(corr_matrix.iloc[i, j], 2)
                        data.append([j, i, value if i != j else "-"])  # "-" en la diagonal (opcional)

                # Crear el option para ECharts
                option = {
                    "tooltip": {"position": "top"},
                    "grid": {"height": "60%", "top": "10%"},
                    "xAxis": {
                        "type": "category",
                        "data": selected_vars,
                        "splitArea": {"show": True}
                    },
                    "yAxis": {
                        "type": "category",
                        "data": selected_vars,
                        "splitArea": {"show": True}
                    },
                    "visualMap": {
                        "min": zmin,
                        "max": zmax,
                        "calculable": True,
                        "orient": "horizontal",
                        "left": "center",
                        "bottom": "5%",
                        "inRange": {
                            "color": [                            
                                # "#053061", "#195696", "#2F79B5", "#4E9AC6", "#86BDDA","#B6D7E8","#DBEAF2",                                 
                                "#053061", "#195696", "#2F79B5", "#4E9AC6","#86BDDA",
                                "#f7f7f7",  # ← neutro para 0
                                "#EF9B7A", "#DA6954", "#C13639", "#9C1127", "#67001F" 
                                # "#FBE3D4", "#F9C3A9", "#EF9B7A", "#DA6954", "#C13639", "#9C1127", "#67001F"                                
                            ]
                        }
                    },
                    "series": [
                        {
                            "name": "Correlación",
                            "type": "heatmap",
                            "data": data,
                            "label": {"show": True},
                            "emphasis": {
                                "itemStyle": {
                                    "shadowBlur": 10,
                                    "shadowColor": "rgba(0, 0, 0, 0.5)"
                                }
                            },
                        }
                    ],
                }

                st_echarts(option, height="500px")

        st.sidebar.subheader("Tipo de Regresión")
        tipo_regresion = st.sidebar.selectbox(
            label="📊 Selecciona el tipo de regresión:",
            options=[
                "Regresión Lineal Simple", 
                "Regresión Lineal Múltiple", 
                "Regresión Logística",
                "Mapa de calor de correlaciones",
            ]
        )

        # Selección de variables
        if tipo_regresion == "Regresión Lineal Simple":
            # Sidebar informativo
            st.sidebar.header("🔧 Panel de Control")
            st.sidebar.subheader("Variables para regresión lineal simple")

            #Menus desplegables de opciones de la variables seleccionadas
            x_var = st.sidebar.selectbox("Variable independiente (X):", options=numeric_cols)
            y_var = st.sidebar.selectbox("Variable dependiente (Y):", options=numeric_cols)

            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split

            X = Naples[[x_var]]
            y = Naples[y_var]
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            st.subheader(f"˚.*☁️ Regresión Lineal Simple: {y_var} vs {x_var}")
            st.write(f"**Coeficiente:** {model.coef_[0]:.4f}")
            st.write(f"**Intercepto:** {model.intercept_:.4f}")

            fig = px.scatter(x=Naples[x_var], y=Naples[y_var], labels={'x': x_var, 'y': y_var})
            fig.add_scatter(x=Naples[x_var], y=y_pred, mode='lines', name='Línea de regresión')
            st.plotly_chart(fig)
        
        elif tipo_regresion == "Regresión Lineal Múltiple":
            st.sidebar.header("🔧 Panel de Control")
            st.sidebar.subheader("Variables para regresión lineal múltiple")

            # Selección de variables independientes (X)
            x_vars = st.sidebar.multiselect("Variables independientes (X):", options=numeric_cols)

            # Selección de variable dependiente (Y)
            y_var = st.sidebar.selectbox("Variable dependiente (Y):", options=numeric_cols)

            if st.sidebar.button("📊 Realizar Regresión"):
                if not x_vars:
                    st.warning("Selecciona al menos una variable independiente.")
                else:
                    # Datos
                    X = Naples[x_vars]
                    y = Naples[y_var]

                    # Entrenar modelo
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)

                    # Resultados
                    st.subheader(f"˚.*☁️  Resultados de Regresión Lineal Múltiple para predecir '{y_var}'")
                    # st.write("Coeficientes:", dict(zip(x_vars, model.coef_)))
                    # st.write("Intercepto:", model.intercept_)
                    # st.write("R² Score:", model.score(X, y))
                    
                    # Mostrar tabla de coeficientes
                    coef_table = pd.DataFrame({
                        "Variable": x_vars,
                        "Coeficiente": model.coef_
                    })
                    st.write("**📈 Coeficientes del Modelo:**")
                    st.dataframe(coef_table.style.format({"Coeficiente": "{:.4f}"}), use_container_width=True)

                    # Mostrar intercepto y R² score
                    summary_table = pd.DataFrame({
                        "Intercepto": [model.intercept_],
                        "R² Score": [model.score(X, y)]
                    })
                    st.write("**📊 Resumen del Modelo:**")
                    st.table(summary_table.style.format({"Intercepto": "{:.4f}", "R² Score": "{:.4f}"}))
                    

                    # Gráfico real vs predicho
                    # df_pred = pd.DataFrame({y_var: y, 'Predicción': y_pred})
                    # fig_pred = px.scatter(df_pred, x=y_var, y='Predicción', title="Valor real vs predicho")
                    # fig_pred.add_shape(
                    #     type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                    #     line=dict(color='red', dash='dash')
                    # )
                    # st.plotly_chart(fig_pred)

                    # Gráfico real vs predicho con colores personalizados
                    df_pred = pd.DataFrame({y_var: y, 'Predicción': y_pred})
                    fig_pred = px.scatter(
                        df_pred,
                        x=y_var,
                        y='Predicción',
                        title="Valor real vs. predicho",
                        labels={y_var: "Valor Real", 'Predicción': "Valor Predicho"}, 
                    )

                    # Personalizar color de los puntos
                    fig_pred.update_traces(marker=dict(color='rgba(100, 149, 237, 0.8)', size=8))  # Azul con transparencia

                    # Agregar línea de identidad (roja punteada)
                    fig_pred.add_shape(
                        type='line',
                        x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                        line=dict(color='crimson', dash='dash'),
                    )

                    # Mostrar gráfico
                    st.plotly_chart(fig_pred)

        
        elif tipo_regresion == "Regresión Logística":
            st.sidebar.subheader("Variables para regresión logística")

            # Filtrar columnas categóricas con al menos dos clases distintas
            valid_categorical_cols = [
                col for col in text_cols if Naples[col].dropna().nunique() >= 2
            ]

            if not valid_categorical_cols:
                st.warning("No hay variables categóricas con al menos dos clases distintas.")

            x_vars = st.sidebar.multiselect("Variables independientes (X):", options=numeric_cols)
            y_var = st.sidebar.selectbox(
                "Variable dependiente categórica (Y):",
                options=valid_categorical_cols
            )

            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder
            import pandas as pd
            # from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score

            if x_vars and y_var:
                # Codificar variable categórica
                encoder = LabelEncoder()
                y_encoded = encoder.fit_transform(Naples[y_var])
                X = Naples[x_vars]

                model = LogisticRegression(max_iter=200)
                model.fit(X, y_encoded)
                score = model.score(X, y_encoded)

                st.subheader(f"˚.*☁️ Regresión Logística para predecir: {y_var}")

                # Crear dataframe con coeficientes
                coef_data = pd.DataFrame({
                    "Variable": x_vars,
                    "Coeficiente": model.coef_[0]
                })

                # # Mostrar tabla de coeficientes
                # st.write("**📈 Coeficientes del Modelo:**")
                # st.dataframe(coef_data.style.format({"Coeficiente": "{:.4f}"}), use_container_width=True)

                # Predicción
                y_pred = model.predict(X)

                # Calcular métricas
                conf_matrix = confusion_matrix(y_encoded, y_pred)
                precision = precision_score(y_encoded, y_pred, average='binary' if len(np.unique(y_encoded)) == 2 else 'macro')
                accuracy = accuracy_score(y_encoded, y_pred)
                recall = recall_score(y_encoded, y_pred, average='binary' if len(np.unique(y_encoded)) == 2 else 'macro')

                # Mostrar la matriz de confusión
                st.subheader("🧩 Matriz de Confusión")
                conf_df = pd.DataFrame(conf_matrix,
                                    index=[f"Real {label}" for label in encoder.classes_],
                                    columns=[f"Predicho {label}" for label in encoder.classes_])
                st.dataframe(conf_df)
        
                # Mostrar métricas como tabla
                st.subheader("📋 Métricas del Modelo")
                metrics_df = pd.DataFrame({
                    "Métrica": ["Precisión", "Exactitud", "Sensibilidad"],
                    "Valor": [precision, accuracy, recall]
                })
                metrics_df["Valor"] = metrics_df["Valor"].apply(lambda x: f"{x:.4f}")
                st.table(metrics_df)

                import seaborn as sns
                import matplotlib.pyplot as plt

                st.subheader("🔍 Matriz de Confusión (Heatmap)")
                fig, ax = plt.subplots()
                sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', ax=ax)
                labels = ["True Neg","False Pos","False Neg","True Pos"]
                ax.set_xlabel("Predicción")
                ax.set_ylabel("Real")
                st.pyplot(fig)


                




        elif tipo_regresion == "Mapa de calor de correlaciones":
            st.sidebar.title("🔧 Panel de Control")
            st.sidebar.header("Mapa de calor")

            st.subheader("༘ ⋆｡˚🔥 Mapa de calor de correlaciones entre variables numéricas")

            # Multiselect para seleccionar variables numéricas
            selected_vars = st.sidebar.multiselect(
                "Selecciona las variables numéricas a incluir en el mapa de calor:",
                options=numeric_cols,
                default=list(numeric_cols)
            )

            # Mostrar automáticamente el mapa de calor si hay al menos dos variables
            if len(selected_vars) < 2:
                st.warning("Selecciona al menos dos variables para generar el mapa de calor.")
            else:
                corr_matrix = Naples[selected_vars].corr()

                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale='RdBu_r',
                    title="Mapa de Calor de Correlación",
                    width=800,  # Ancho en píxeles
                    height=600  # Alto en píxeles
                )
                st.plotly_chart(fig_heatmap)


