#Creamos el archivo de la APP en el interprete principal (Phyton)

#####################################################
#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd

######################################################
#Definimos la instancia
@st.cache_resource

######################################################
#Creamos la función de carga de datos
def load_data():
   #Lectura del archivo csv
   df=pd.read_csv("titanic.csv", index_col= 'Name')
   #df=pd.read_csv("Datos_Limpios_Naples.csv")

   #Selecciono las columnas tipo numericas del dataframe
   numeric_df = df.select_dtypes(['float','int'])  #Devuelve Columnas
   numeric_cols= numeric_df.columns                #Devuelve lista de Columnas

   #Selecciono las columnas tipo texto del dataframe
   text_df = df.select_dtypes(['object'])  #Devuelve Columnas
   text_cols= text_df.columns              #Devuelve lista de Columnas
   
   #Selecciono algunas columnas categoricas de valores para desplegar en diferentes cuadros
   categorical_column_sex= df['Sex']
   #Obtengo los valores unicos de la columna categórica seleccionada
   unique_categories_sex= categorical_column_sex.unique()

   return df, numeric_cols, text_cols, unique_categories_sex, numeric_df

###############################################################################
#Cargo los datos obtenidos de la función "load_data"
df, numeric_cols, text_cols, unique_categories_sex, numeric_df = load_data()
###############################################################################
#CREACIÓN DEL DASHBOARD
#Generamos las páginas que utilizaremos en el diseño
#Widget 1: Selectbox
#Menu desplegable de opciones de laa páginas seleccionadas
View= st.selectbox(label= "View", options= ["View 1", "View 2", "View 3", "View 4"])
# CONTENIDO DE LA VISTA 1
if View == "View 1":
#Generamos los encabezados para el dashboard
    st.title("TITANIC")
    st.header("Panel Principal")
    st.subheader("Line Plot")
##############################################################################
#Generamos los encabezados para la barra lateral (sidebar)
    st.sidebar.title("DASHBOARD")
    st.sidebar.header("Sidebar")
    st.sidebar.subheader("Panel de selección")
###############################################################################
#Widget 2: Checkbox
#Generamos un cuadro de selección (Checkbox) en una barra lateral (sidebar) para mostrar dataset
    check_box = st.sidebar.checkbox(label= "Mostrar Dataset")
    
    #Condicional para que aparezca el dataframe
    if check_box:
   #Mostramos el dataset
        st.write(df)
        st.write(df.columns)
        st.write(df.describe())
        
##############################################################################
#Widget 3: Multiselect box
#Generamos un cuadro de multi-selección (Y) para seleccionar variables a graficar
    numerics_vars_selected= st.sidebar.multiselect(label="Variables graficadas", options= numeric_cols)
#Widget 3: Selectbox
#Menu desplegable de opciones de la variable categórica seleccionada
    category_selected= st.sidebar.selectbox(label= "Categorias", options= unique_categories_sex)
#Widget 4: Button
#Generamos un button (Button) en la barra lateral (sidebar) para mostrar las variables tipo texto
    Button = st.sidebar.button(label= "Mostrar variables STRING")

#Condicional para que aparezca el dataframe
    if Button:
   #Mostramos el dataset
        st.write(text_cols)
        
###############################################################################
#GRAPH 1: LINEPLOT
#Despliegue de un line plot, definiendo las variables "X categorias" y "Y numéricas" 
    data= df[df['Sex']==category_selected]
    data_features= data[numerics_vars_selected]
    figure1 = px.line(data_frame=data_features, x=data_features.index, 
                  y= numerics_vars_selected, title= str('Features of Passengers'), 
                  width=1600, height=600)
    
#Generamos un button (Button) en la barra lateral (sidebar) para mostrar el lineplot
    Button2 = st.sidebar.button(label= "Mostrar grafica tipo LINEPLOT")

#Condicional para que aparezca la grafica tipo Line Plot
    if Button2:
   #Mostramos el lineplot
        st.plotly_chart(figure1)

###############################################################################
# CONTENIDO DE LA VISTA 2
elif View == "View 2":
#Generamos los encabezados para el dashboard
    st.title("TITANIC")
    st.header("Panel Principal")
    st.subheader("Scatter Plot")

#GRAPH 2: SCATTERPLOT
    x_selected= st.sidebar.selectbox(label= "x", options= numeric_cols)
    y_selected= st.sidebar.selectbox(label= "y", options= numeric_cols)
    figure2 = px.scatter(data_frame=numeric_df, x=x_selected, y= y_selected, 
                     title= 'Dispersiones')
    st.plotly_chart(figure2)
    
###############################################################################
# CONTENIDO DE LA VISTA 3
elif View == "View 3":
#Generamos los encabezados para el dashboard
    st.title("TITANIC")
    st.header("Panel Principal")
    st.subheader("Pie Plot")
    
    #Menus desplegables de opciones de la variables seleccionadas
    Variable_cat= st.sidebar.selectbox(label= "Variable Categórica", options= text_cols)
    Variable_num= st.sidebar.selectbox(label= "Variable Numérica", options= numeric_cols)

#GRAPH 3: PIEPLOT
#Despliegue de un pie plot, definiendo las variables "X categorias" y "Y numéricas" 
    figure3 = px.pie(data_frame=df, names=df[Variable_cat], 
                  values= df[Variable_num], title= str('Features of')+' '+'Passengers', 
                  width=1600, height=600)
    st.plotly_chart(figure3)
    
    
# CONTENIDO DE LA VISTA 4
elif View == "View 4":
#Generamos los encabezados para el dashboard
    st.title("TITANIC")
    st.header("Panel Principal")
    st.subheader("Bar Plot")
    
    #Menus desplegables de opciones de la variables seleccionadas
    Variable_cat= st.sidebar.selectbox(label= "Variable Categórica", options= text_cols)
    Variable_num= st.sidebar.selectbox(label= "Variable Numérica", options= numeric_cols)
    
#GRAPH 4: BARPLOT
#Despliegue de un bar plot, definiendo las variables "X categorias" y "Y numéricas" 
    figure4 = px.bar(data_frame=df, x=df[Variable_cat], 
                  y= df[Variable_num], title= str('Features of')+' '+'Passengers')
    figure4.update_xaxes(automargin=True)
    figure4.update_yaxes(automargin=True)
    st.plotly_chart(figure4)
