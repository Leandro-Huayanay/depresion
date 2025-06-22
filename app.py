import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    return joblib.load("depresion_tree_model.pkl")

model = load_model()

# Título de la app
st.title("Predicción de Depresión con Árbol de Decisión")

st.markdown("""
Esta aplicación predice si una persona **puede tener depresión (DEP_ACT)** o **no tiene depresión**, 
basada en características personales y respuestas a preguntas P1 a P9.
""")

# Entradas del usuario
edadc = st.number_input("Edad categórica (1 a N)", min_value=1, max_value=10)
sexo = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Mujer" if x == 0 else "Hombre")
educacion = st.number_input("Nivel educativo (1 a N)", min_value=1, max_value=10)
estado_civil = st.number_input("Estado civil (1 a N)", min_value=1, max_value=10)
idioma_materno = st.selectbox("Idioma materno", options=[0, 1], format_func=lambda x: "Español" if x == 0 else "Otro")

# Preguntas P1 a P9 (ajusta los rangos si tu modelo usa valores distintos)
P1 = st.slider("P1", 0, 3)
P2 = st.slider("P2", 0, 3)
P3 = st.slider("P3", 0, 3)
P4 = st.slider("P4", 0, 3)
P5 = st.slider("P5", 0, 3)
P6 = st.slider("P6", 0, 3)
P7 = st.slider("P7", 0, 3)
P8 = st.slider("P8", 0, 3)
P9 = st.slider("P9", 0, 3)

# Crear dataframe de entrada
input_df = pd.DataFrame([{
    "edadc": edadc,
    "sexo": sexo,
    "educacion": educacion,
    "estado_civil": estado_civil,
    "idioma_materno": idioma_materno,
    "P1": P1,
    "P2": P2,
    "P3": P3,
    "P4": P4,
    "P5": P5,
    "P6": P6,
    "P7": P7,
    "P8": P8,
    "P9": P9
}])

# Botón para predecir
if st.button("Predecir DEP_ACT"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.error("⚠️ Resultado: Puede tener depresión (DEP_ACT = 1)")
    else:
        st.success("✅ Resultado: No tiene depresión (DEP_ACT = 0)")
