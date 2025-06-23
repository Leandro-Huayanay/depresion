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
***Esta aplicación predice si una persona **puede tener depresión (DEPRESION)** o **no tiene depresión**, 
basada en características personales y respuestas a preguntas P1 a P9.
P1. ¿Pocas ganas o interés en hacer las cosas? (Es decir, no disfruta sus actividades cotidianas)
P2. ¿Sentirse desanimada(o), deprimida(o), triste o sin esperanza?
P3. ¿Problemas para dormir o mantenerse dormida(o), o en dormir demasiado?
P4. ¿Sentirse cansada(o) o tener poca energía sin motivo que lo justifique?
P5. ¿Poco apetito o comer en exceso?
P6. ¿Dificultad para poner atención o concentrarse en las cosas que hace? (Como leer el periódico, ver televisión, escuchar atentamente la radio o conversar con otras personas)
P7. ¿Moverse mas lento o hablar más lento de lo normal o sentirse más inquieta(o) o intranquila(o) de lo normal?
P8. ¿Pensamientos de que sería mejor estar muerta(o) o que quisiera hacerse daño de alguna forma buscando morir?
P9. ¿Sentirse mal acerca de si misma(o) o sentir que es una(un) fracasada(o) o que se ha fallado a sí misma(o) o a su familia?
***Pueden ser respondidas:
0 "Para_nada"  1 "de_1_a_6_días" 2 "de_7_a_11_dias" 3 "de_12_a_mas_días"

""")

# Entradas del usuario
edadc = st.number_input("Edad categórica (1 a 4)", min_value=1, max_value=4)
sexo = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Mujer" if x == 0 else "Hombre")
educacion = st.number_input("Nivel educativo (0 a 5)", min_value=0, max_value=5)
estado_civil = st.number_input("Estado civil (0 a 5)", min_value=0, max_value=5)
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
if st.button("Predecir DEPRESION"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.error("⚠️ Resultado: Puede tener depresión (DEPRESION = 1)")
    else:
        st.success("✅ Resultado: No tiene depresión (DEPRESION = 0)")
