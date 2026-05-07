import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title='Deserción Escolar', page_icon='🎓', layout='centered')

@st.cache_resource
def cargar_modelo():
    with open('modelo.pkl', 'rb') as f:
        return pickle.load(f)

modelo = cargar_modelo()

st.title('🎓 Predicción de Deserción Escolar')
st.write('Ingrese los datos del estudiante para predecir el riesgo de deserción.')
st.divider()

st.subheader('📋 Datos del estudiante')
col1, col2 = st.columns(2)

with col1:
    edad          = st.slider('Edad', 11, 19, 15)
    grado         = st.slider('Grado escolar', 6, 11, 9)
    promedio      = st.slider('Promedio académico', 1.0, 5.0, 3.0, step=0.1)
    inasistencias = st.number_input('Días de inasistencia', 0, 100, 10)
    repitencia    = st.number_input('Años repitidos', 0, 3, 0)

with col2:
    nivel_se        = st.selectbox('Estrato socioeconómico', [1,2,3,4,5,6])
    distancia_km    = st.slider('Distancia al colegio (km)', 0.5, 30.0, 3.0, step=0.5)
    trabaja         = st.radio('¿Trabaja?', ['No','Sí'])
    acceso_internet = st.radio('¿Tiene internet en casa?', ['No','Sí'])
    apoyo_familiar  = st.slider('Apoyo familiar (1=bajo, 5=alto)', 1, 5, 3)

datos_nuevo = pd.DataFrame([{
    'edad':                edad,
    'grado':               grado,
    'promedio':            promedio,
    'inasistencias':       inasistencias,
    'repitencia':          repitencia,
    'nivel_socioeconomico':nivel_se,
    'distancia_km':        distancia_km,
    'trabaja':             1 if trabaja=='Sí' else 0,
    'acceso_internet':     1 if acceso_internet=='Sí' else 0,
    'apoyo_familiar':      apoyo_familiar
}])

st.divider()
if st.button('🔍 Predecir', use_container_width=True):

    prediccion   = modelo.predict(datos_nuevo)[0]
    probabilidad = modelo.predict_proba(datos_nuevo)[0]

    st.subheader('📊 Resultado')
    col_res, col_prob = st.columns(2)

    with col_res:
        if prediccion == 1:
            st.error('⚠️ Alto riesgo de deserción')
        else:
            st.success('✅ Bajo riesgo de deserción')

    with col_prob:
        st.metric('Probabilidad de desertar',  f'{probabilidad[1]:.1%}')
        st.metric('Probabilidad de continuar', f'{probabilidad[0]:.1%}')

    st.divider()
    st.subheader('📈 Probabilidades')
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(['No deserta','Deserta'], probabilidad,
            color=['#1D9E75','#E24B4A'], height=0.5)
    for i, v in enumerate(probabilidad):
        ax.text(v+0.01, i, f'{v:.1%}', va='center', fontsize=12)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Probabilidad')
    st.pyplot(fig)

    st.divider()
    st.subheader('🔍 Variables más importantes')
    features = ['edad','grado','promedio','inasistencias','repitencia',
                'nivel_socioeconomico','distancia_km','trabaja',
                'acceso_internet','apoyo_familiar']
    importancias = pd.Series(
        modelo.feature_importances_, index=features
    ).sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(importancias.index, importancias.values, color='#378ADD', height=0.6)
    ax2.set_xlabel('Importancia')
    ax2.set_title('Importancia de variables')
    st.pyplot(fig2)

st.divider()
st.caption('Universidad de San Buenaventura — Fundamentos de Ciencia de Datos · 2025')