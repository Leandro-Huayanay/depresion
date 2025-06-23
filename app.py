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
Esta aplicación predice si una persona **puede tener depresión (DEPRESIÓN)** o **no tiene depresión**, 
basándose en características personales y respuestas a las preguntas P1 a P9, referidas a los últimos 14 días.
""")

st.markdown("**P1.** ¿Ha tenido pocas ganas o interés en hacer las cosas (es decir, no disfruta sus actividades cotidianas)?")
st.markdown("**P2.** ¿Se ha sentido desanimado(a), deprimido(a), triste o sin esperanza?")
st.markdown("**P3.** ¿Ha tenido problemas para dormir, mantenerse dormido(a) o ha dormido demasiado?")
st.markdown("**P4.** ¿Se ha sentido cansado(a) o con poca energía sin un motivo claro?")
st.markdown("**P5.** ¿Ha tenido poco apetito o ha comido en exceso?")
st.markdown("**P6.** ¿Ha tenido dificultad para poner atención o concentrarse en lo que hace (por ejemplo, leer, ver televisión, escuchar la radio o conversar)?")
st.markdown("**P7.** ¿Se ha movido o hablado más lento de lo normal, o se ha sentido más inquieto(a) o intranquilo(a) de lo habitual?")
st.markdown("**P8.** ¿Ha tenido pensamientos de que estaría mejor muerto(a) o ha querido hacerse daño de alguna forma?")
st.markdown("**P9.** ¿Se ha sentido mal consigo mismo(a), como si fuera un(a) fracasado(a) o que le ha fallado a su familia?")

st.markdown("""
**Respuestas posibles para P1 a P9:**  
0 = "Para nada"  
1 = "De 1 a 6 días"  
2 = "De 7 a 11 días"  
3 = "De 12 o más días"
""")


# Entradas del usuario
edadc = st.number_input("Edad categórizada (1:Joven 2:Adulto joven 3 Adulto  4 Adulto mayor)", min_value=1, max_value=4)
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
